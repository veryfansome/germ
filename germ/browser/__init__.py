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

MD_PROCESSED_MARKER = "__md__"


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
            return await self.page_to_markdown(page)

        try:
            remaining_ms = time_budget.remaining_ms()
            if remaining_ms >= 1000:
                # Wait for page to settle if time allows
                await self.wait_for_page_to_settle(page, max_ms=(remaining_ms - 200))  # Leave room

            await _expand_interactive_docs(page)

            return await self.page_to_markdown(page)
        except Exception:
            await page.close()
            raise

    async def page_to_markdown(self, page: Page) -> str:
        html = await page.content()
        async with self._thread_pool_semaphore:
            return await asyncio.get_running_loop().run_in_executor(
                self._thread_pool, html_to_markdown, html
            )

    async def stop(self):
        await super().stop()
        self._thread_pool.shutdown(wait=False, cancel_futures=True)


def decompose_non_content_elements(root_tag: Tag):
    """Strip out things that are not content related."""
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
        "svg",
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
    parent_ol_cnt = len(root_tag.find_parents("ol"))
    parent_ul_cnt = len(root_tag.find_parents("ul"))
    indent = (" " * 3 * parent_ol_cnt) + (" " * 2 * parent_ul_cnt)
    return indent


def html_to_markdown(html: str):
    html = html.replace("\xa0", " ")
    soup = BeautifulSoup(html, "lxml")

    root_tag = soup.select_one("main")  # Prefer main if exists
    if not root_tag:
        root_tag = soup.select_one("body")

    decompose_non_content_elements(root_tag)
    markdownify_heading_tags(root_tag)
    markdownify_simple_tags(root_tag)
    markdownify_list_tags(root_tag)
    markdownify_pre_tags(root_tag)

    for element in reversed(root_tag.select("*")):
        element.attrs = {}  # Strip original attributes
        if not element.text.strip() or element.name not in {
            # Leave useful tags that don't have equivalents in markdown
            "aside",
            "details",
            "summary",
        }:
            element.unwrap()
    text = str(root_tag)  # TODO: This still leaves the root tag
    text = normalize_newlines(text)
    return text.strip()


def markdownify_simple_tags(root_tag: Tag):
    # TODO: Maybe split flow and phrase into two passes
    for tag in reversed(root_tag.select(",".join([
        "a",
        "aside",
        "b",
        "br",
        "code",
        "div",
        "em",
        "hr",
        "i",
        "p",
        "s",
        "span",
        "strong",
    ]))):
        if not isinstance(tag, Tag) or tag.name is None:
            continue  # Continue if decomposed earlier in loop

        class_attrs = tag.get("class", AttributeValueList())  # TODO: Maybe move this lower down
        if MD_PROCESSED_MARKER in class_attrs:
            continue

        if tag.name in {"br"}:
            tag.replace_with(NavigableString("\n\n"))
        elif tag.name in {"hr"}:
            tag.replace_with(NavigableString("---"))
        else:
            tag_text = tag.get_text()
            if tag_text:#.strip():
                if tag.name == "a":
                    href = tag.get("href")
                    if not href:
                        tag.decompose()
                        continue
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
                    else:
                        tag.unwrap()
            else:
                tag.decompose()
                continue

        if isinstance(tag, Tag):
            class_attrs.append(MD_PROCESSED_MARKER)
            tag["class"] = class_attrs


def markdownify_heading_tags(root_tag: Tag):
    for tag in root_tag.select("h1,h2,h3,h4,h5,h6"):
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


def markdownify_list_tags(root_tag: Tag):
    for list_tag in root_tag.find_all(["ol", "ul"], recursive=True):
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
        for item_tag in list_tag.find_all('li', recursive=False):
            normalize_navigable_strings(item_tag)
            if is_ordered:
                prefix = f"{index}. "
                index += 1
            else:
                prefix = "- "
            item_tag.insert(0, NavigableString(f"\n{indent}{prefix}"))

        if list_tag.next_sibling:
            list_tag.append(NavigableString(f"\n{indent}"))

        class_attrs.append(MD_PROCESSED_MARKER)
        list_tag["class"] = class_attrs


def markdownify_pre_tags(root_tag: Tag):
    for tag in root_tag.find_all("pre", recursive=True):
        class_attrs = tag.get("class", AttributeValueList())
        if MD_PROCESSED_MARKER in class_attrs:
            continue

        ticks = ""
        while len(tag.parent.contents) == 1:
            if "code" in tag.parent.name:  # Some frameworks use custom tags that wrap around <pre>
                ticks = "```"
            tag.parent.unwrap()  # Unwrap all parents where the pre tag is the only child
        if len(tag.contents) == 1:
            child = tag.contents[0]
            if isinstance(child, Tag) and child.name == "code":
                ticks = "```"

        indent = get_indentation(tag)

        if tag.previous_sibling:
            tag.insert_before(NavigableString(f"\n{indent}"))
        tag.insert_before(NavigableString(f"{ticks}\n{indent}"))
        if tag.next_sibling:
            tag.insert_after(NavigableString(f"\n{indent}"))
        tag.insert_after(NavigableString(f"\n{indent}{ticks}"))

        if not ticks:
            for element in tag.contents:
                if isinstance(element, NavigableString):
                    element_text = element.get_text().strip().replace("\n", f"\n{indent}")
                    element.replace_with(NavigableString(element_text))
        else:
            inner_text = tag.get_text().strip().replace("\n", f"\n{indent}")
            tag.replace_with(NavigableString(inner_text))

        class_attrs.append(MD_PROCESSED_MARKER)
        tag["class"] = class_attrs


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
