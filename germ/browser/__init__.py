import asyncio
import logging
import os
import re
from bs4 import BeautifulSoup, Tag
from bs4.element import AttributeValueList, Comment, NavigableString
from concurrent.futures import ThreadPoolExecutor
from playwright.async_api import Page

from germ.browser.base import BaseBrowser

logger = logging.getLogger(__name__)

KEEP_HTML_MARKER = "__html__"
MD_PROCESSED_MARKER = "__md__"
UNICODE_TRANSLATIONS = str.maketrans({
    0x00A0: 0x20,   # NBSP -> space
    0x00AD: None,   # soft hyphen -> delete
})


class PageScrapingWebBrowser(BaseBrowser):
    def __init__(self, max_contexts: int = 24):
        super().__init__(max_contexts)
        workers = min(os.cpu_count() or 4, 8)
        self._thread_pool = ThreadPoolExecutor(max_workers=workers)
        self._thread_pool_semaphore = asyncio.Semaphore(workers)

    async def fetch_url(
            self, filtered_messages: list[dict[str, str]], url: str,
            user_id: int, user_agent: str, extra_headers: dict[str, str],
            timeout_ms: int = 15000,
    ) -> str:
        time_budget, resp, page = await self.goto_url(url, user_id, user_agent, extra_headers, timeout_ms=timeout_ms)

        resp_status = None
        resp_content_type = ""
        if resp is not None:
            resp_status = resp.status
            resp_content_type = resp.headers.get("content-type", "").lower()

        if resp_status is None or resp_status >= 400:
            await page.close()
            raise RuntimeError(f"Failed to load page: {url}")
        elif "application/xhtml+xml" not in resp_content_type and "text/html" not in resp_content_type:
            await page.close()
            raise RuntimeError(f"Unsupported content_type: {resp_content_type}")

        remaining_ms = time_budget.remaining_ms()
        if remaining_ms <= 0:
            return await self.page_to_md(page)

        try:
            remaining_ms = time_budget.remaining_ms()
            if remaining_ms >= 1000:
                # Wait for page to settle if time allows
                await self.wait_for_page_to_settle(page, max_ms=(remaining_ms - 200))  # Leave room

            await _expand_interactive_docs(page)

            return await self.page_to_md(page)
        except Exception:
            await page.close()
            raise

    async def page_to_md(self, page: Page) -> str:
        html = await page.content()
        async with self._thread_pool_semaphore:
            return await asyncio.get_running_loop().run_in_executor(
                self._thread_pool, html_to_md, html
            )

    async def stop(self):
        await super().stop()
        self._thread_pool.shutdown(wait=False, cancel_futures=True)


def decompose_non_content_elements(root_tag: Tag):
    """Strip out things that are not needed or not content related."""
    tags_to_decompose = [
        "[class*=breadcrumb]",
        "[class*=nocontent]",
        "[class*=sidebar]",
        "[role*=form]",
        "[role*=navigation]",
        "[role*=presentation]",
        "button",
        "head",
        "nav",
        "noscript",
        "script",
        "style",
        "template",
    ]
    ## Framework specifics
    if root_tag.select_one("devsite-content"):  # Google devsite
        tags_to_decompose.extend([
            "cloud-free-trial",
            "[class*=devsite-code-buttons-container]",
            "[class*=devsite-code-cta-popout]",
            "[class*=devsite-nav-list]",
        ])
    for tag in root_tag.select(','.join(tags_to_decompose)):
        tag.decompose()
    for comment in root_tag.find_all(string=lambda s: isinstance(s, Comment)):
        comment.decompose()


def escape_markdown(text):
    """Escapes all common Markdown special characters."""
    special_chars = r'''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    return re.sub(f"([{re.escape(special_chars)}])", r"\\\1", text)


async def _expand_interactive_docs(page: Page):
    # Open <details>, reveal hidden tabpanels, and dismiss aria-hidden
    await page.evaluate(
        """
        () => {
          document.querySelectorAll('details').forEach(d => { try { d.open = true; } catch {} });
          document.querySelectorAll('[role="tabpanel"][hidden]').forEach(el => { try { el.hidden = false; } catch {} });
          document.querySelectorAll('[aria-hidden="true"]').forEach(el => { try { el.setAttribute('aria-hidden','false'); } catch {} });
          document.querySelectorAll('[hidden]').forEach(el => { try { el.removeAttribute('hidden'); } catch {} });
        }
        """
    )


def get_indentation(root_tag: Tag) -> str:
    indent = ""
    for parent in reversed(root_tag.find_parents(["blockquote", "ol", "ul"])):
        match parent.name:
            case "blockquote":
                indent += "> "
            case "ol":
                indent += (" " * 3)
            case "ul":
                indent += (" " * 2)
    return indent


def html_to_md(html: str):
    soup = BeautifulSoup(html, "lxml")

    root_tag = soup.select_one("main")  # Prefer main if exists
    if not root_tag:
        root_tag = soup.select_one("body")

    decompose_non_content_elements(root_tag)
    md_convert_heading_tags(root_tag)
    md_convert_simple_tags(root_tag)
    md_convert_list_tags(root_tag)
    md_convert_pre_tags(root_tag)

    #text = str(root_tag)
    for element in reversed(root_tag.find_all()):
        if KEEP_HTML_MARKER in element.get("class", AttributeValueList()):
            # Leave tags that have been marked for retention
            element.attrs.pop("class")
            attr_blob = ' '.join([f'{attr}="{val}"' for attr, val in element.attrs.items()])
            element.insert_before(NavigableString(f"<{element.name} {attr_blob}>"))
            element.insert_after(f"</{element.name}>")
            element.unwrap()
        elif element.name in {
            "aside",
            "audio",
            "details",
            "figcaption",
            "figure",
            "mark",
            "picture",
            "small",
            "sub",
            "summary",
            "sup",
            "u",
            "video",
        }:
            # Leave tags that don't have equivalents in markdown
            element.insert_before(NavigableString(f"<{element.name}>"))
            element.insert_after(f"</{element.name}>")
            element.unwrap()
        elif not element.text.strip():
            element.unwrap()
    text = root_tag.get_text()
    text = normalize_newlines(text)
    text = text.translate(UNICODE_TRANSLATIONS)
    return text.strip()


def md_convert_simple_tag(tag: Tag):
    if not isinstance(tag, Tag) or tag.name is None:
        return  # If already decomposed

    class_attrs = tag.get("class", AttributeValueList())
    if MD_PROCESSED_MARKER in class_attrs:
        return

    match tag.name:
        case "hr":
            tag.replace_with(NavigableString("---"))
        case "audio"|"picture"|"video":
            children_to_keep = tag.find_all(["img", "source", "track"], recursive=False)
            tag.clear()
            tag.extend(children_to_keep)
        case "img"|"source"|"track":
            if (tag.name == "img"
                    and (tag.has_attr("hidden") or (not tag.has_attr("src") and not tag.has_attr("title")))):
                tag.decompose()
            else:
                for attr in set(tag.attrs.keys()):
                    if attr not in {"src", "srcset", "title"}:
                        tag.attrs.pop(attr)
                class_attrs.append(KEEP_HTML_MARKER)
        case "figure":
            for element in tag.contents:
                if not isinstance(element, Tag):
                    element.decompose()
        case "svg":
            for attr in set(tag.attrs.keys()):
                if attr not in {"role", "viewbox", "xmlns"}:
                    tag.attrs.pop(attr)
            children_to_keep = []
            for element in tag.find_all(recursive=False):
                if isinstance(element, Tag):
                    element_class_attrs = element.get("class", AttributeValueList())
                    element_class_attrs.extend([KEEP_HTML_MARKER, MD_PROCESSED_MARKER])
                    element["class"] = element_class_attrs
                    children_to_keep.append(element)
            tag.clear()
            tag.extend(children_to_keep)
            class_attrs.append(KEEP_HTML_MARKER)
        case _:
            tag_text = tag.get_text()
            if tag_text:
                if tag.name == "a":
                    href = tag.get("href")
                    if not href:
                        tag.decompose()
                        return
                    normalize_navigable_strings(tag)
                    # TODO: implement link conversion
                elif tag.name == "aside":
                    normalize_navigable_strings(tag)
                elif tag.name in {"b", "strong"}:
                    tag.insert_before(NavigableString("**"))
                    tag.insert_after(NavigableString("**"))
                elif tag.name == "code" and tag.parent.name != "pre":  # block code handled separately
                    tag.replace_with(NavigableString(f"`{tag_text}`"))
                elif tag.name == "div":
                    normalize_navigable_strings(tag)
                elif tag.name in {"i", "em"}:
                    tag.insert_before(NavigableString("*"))
                    tag.insert_after(NavigableString("*"))
                elif tag.name == "p":
                    normalize_navigable_strings(tag)
                elif tag.name == "s":
                    tag.insert_before(NavigableString("~~"))
                    tag.insert_after(NavigableString("~~"))
                elif tag.name == "span":
                    if tag.has_attr("data-literal"):
                        tag.replace_with(NavigableString(escape_markdown(tag_text)))
                    elif not tag.find_parent("code"):
                        normalize_navigable_strings(tag)
                        tag.unwrap()
            else:
                tag.decompose()
                return

    if isinstance(tag, Tag):
        class_attrs.append(MD_PROCESSED_MARKER)
        tag["class"] = class_attrs


def md_convert_simple_tags(root_tag: Tag):
    """Finds tags and converts them in phases."""
    # Media tags
    for tag in reversed(root_tag.find_all([
        "audio",
        "picture",
        "svg",
        "video",
    ])):
        md_convert_simple_tag(tag)
    # Phrasing content + some media related tags
    for tag in reversed(root_tag.find_all([
        "a",
        "b",
        "code",
        "em",
        "i",
        "img",
        "s",
        "source",
        "span",
        "strong",
        "track",
    ])):
        md_convert_simple_tag(tag)
    # Flow content tags
    for tag in reversed(root_tag.find_all([
        "aside",
        "div",
        "figure",
        "hr",
        "p",
    ])):
        md_convert_simple_tag(tag)
    # Line break
    for tag in root_tag.select("br"):
        tag.replace_with(NavigableString("\n\n"))


def md_convert_heading_tags(root_tag: Tag):
    for tag in root_tag.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        class_attrs = tag.get("class", AttributeValueList())
        if MD_PROCESSED_MARKER in class_attrs:
            continue

        for tag_component in tag.contents:
            if isinstance(tag_component, NavigableString):
                tag_component.replace_with(NavigableString(
                    tag_component.get_text().replace("\n", " ").strip()
                ))
        tag.insert_before(NavigableString(f"\n{int(tag.name.lstrip('h')) * '#'} "))
        tag.insert_after(NavigableString("\n"))

        class_attrs.append(MD_PROCESSED_MARKER)
        tag["class"] = class_attrs


def md_convert_list_tags(root_tag: Tag):
    for list_tag in root_tag.find_all(["ol", "ul"]):
        class_attrs = list_tag.get("class", AttributeValueList())
        if MD_PROCESSED_MARKER in class_attrs:
            continue

        for element in list_tag.contents:
            # Remove anything that isn't a <li>
            if isinstance(element, NavigableString) or (isinstance(element, Tag) and element.name != "li"):
                element.decompose()

        # Determine starting index for ordered lists (default 1)
        is_ordered = (list_tag.name == 'ol')
        index = 1
        if is_ordered and list_tag.has_attr('start'):
            try:
                index = int(list_tag['start'])
            except ValueError:
                index = 1

        indent = get_indentation(list_tag)
        for item_idx, item_tag in enumerate(list_tag.find_all('li', recursive=False)):
            normalize_navigable_strings(item_tag)
            if is_ordered:
                prefix = f"{index}. "
                index += 1
            else:
                prefix = "- "
            indented_prefix = f"{indent}{prefix}"
            if item_idx > 0 or (list_tag.previous_sibling and not list_tag.previous_sibling.get_text().endswith("\n")):
                indented_prefix = "\n" + indented_prefix
            item_tag.insert_before(NavigableString(indented_prefix))

        if list_tag.next_sibling and not list_tag.next_sibling.get_text().startswith("\n"):
            list_tag.append(NavigableString(f"\n{indent}"))

        class_attrs.append(MD_PROCESSED_MARKER)
        list_tag["class"] = class_attrs


def md_convert_pre_tags(root_tag: Tag):
    for tag in root_tag.find_all("pre"):
        while len(tag.parent.contents) == 1:
            # With some exceptions, unwrap all parents where the pre tag is the only child
            if tag.parent.name in {"aside", "blockquote", "details", "figure", "li", "td", "th"}:
                break
            else:
                tag.parent.unwrap()

        fence = "```"
        indent = get_indentation(tag)
        inner_text = tag.get_text().strip().replace("\n", f"\n{indent}")
        if fence in inner_text:
            fence = "~~~"
        if tag.parent.name == "li" and tag.parent.index(tag) == 0:
            # If 1st element, indention handled by md_convert_list_tags
            tag.insert_before(NavigableString(f"{fence}\n{indent}"))
        else:
            if tag.previous_sibling and not tag.previous_sibling.get_text().endswith("\n"):
                tag.insert_before(NavigableString(f"\n"))
            tag.insert_before(NavigableString(f"{indent}{fence}\n{indent}"))
        if tag.next_sibling and not tag.next_sibling.get_text().startswith("\n"):
            tag.insert_after(NavigableString(f"\n{indent}"))
        tag.insert_after(NavigableString(f"\n{indent}{fence}"))
        tag.replace_with(NavigableString(inner_text))


def normalize_navigable_strings(tag: Tag):
    for element_idx, element in enumerate(tag.contents):
        if isinstance(element, NavigableString):
            element_text = element.get_text()
            if element_idx == 0:
                element_text = element_text.lstrip()
            if element_text in {"", "\n"}:
                element.decompose()
                continue
            element_text = re.sub(r"[\n\s\t]+", " ", element_text)
            element.replace_with(NavigableString(element_text))


def normalize_newlines(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text
