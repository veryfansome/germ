import asyncio
import logging
import os
import re

import bs4.element
from bs4 import BeautifulSoup
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from enum import Enum
from pathlib import Path
from playwright.async_api import Browser, BrowserContext, Page, Playwright, Request, Response, Route, async_playwright
from prometheus_client import Gauge
from pydantic import BaseModel
from urllib.parse import SplitResult, parse_qsl, urlencode, urljoin, urlparse, urlsplit

from germ.utils import TimeBudget

logger = logging.getLogger(__name__)

blocked_analytics_hosts = (
    "adservice.google.com",
    "doubleclick.net",
    "facebook.net",
    "google-analytics.com",
    "googlesyndication.com",
    "googletagmanager.com",
    "hotjar.com",
    "mixpanel.com",
    "segment.io",
    "sentry.io",
)

concurrent_browser_contexts_gauge = Gauge(
    "concurrent_browser_contexts", "Number of concurrent browser contexts in use.")


class FetchResult(BaseModel):
    content_type: str
    extraction_status: "TextExtractionStatus"
    status_code: int | None
    text: str
    url: str


class TextExtractionStatus(Enum):
    EXTRACTED = "extracted"
    PASSED_THROUGH_NOT_TEXT = "passed_through_not_text"
    PASSED_THROUGH_RAW_HTML = "passed_through_raw_html"
    SKIPPED_ERROR = "skipped_error"
    SKIPPED_NOT_TEXT = "skipped_not_text"


class WebBrowser:
    def __init__(self, max_contexts: int = 24):
        workers = min(os.cpu_count() or 4, 8)

        self._browser: Browser | None = None
        self._cache = {}
        self._contexts: OrderedDict[tuple[int, str, str | None, int | None], BrowserContext] = OrderedDict()
        self._contexts_lock: asyncio.Lock = asyncio.Lock()
        self._max_contexts = max_contexts
        self._playwright: Playwright | None = None
        self._thread_pool = ThreadPoolExecutor(max_workers=workers)
        self._thread_pool_semaphore = asyncio.Semaphore(workers)

    async def _get_or_create_context(
            self, url_split: SplitResult, user_id: int, user_agent: str, extra_headers: dict[str, str]
    ) -> BrowserContext:
        # Normalize URL parts
        url_hostname = (url_split.hostname or "").lower()
        url_port = url_split.port or (443 if url_split.scheme=="https" else 80)

        # UA and extra headers are passed-through per user for realism to minimize anti-bot friction. They're not used
        # in the context key because they don't need to perfectly mirror the user's behavior if the user changes
        # browsers. This avoids hardcoding global values that need to be updated over time. Keying on user + host
        # keeps isolation for user credentials if that's supported in the future.
        context_key = (user_id, url_split.scheme, url_hostname, url_port)
        async with self._contexts_lock:
            if context_key in self._contexts:
                # LRU: mark as recently used
                context = self._contexts.pop(context_key)
                self._contexts[context_key] = context
                return context

            locale = "en-US"
            if "accept-language" in extra_headers:
                locale = extra_headers["accept-language"].split(",", 1)[0].split(";", 1)[0].strip()
            context = await self._browser.new_context(
                extra_http_headers={
                    "upgrade-insecure-requests": "1",
                    **extra_headers,   # Filtered list from user's browser: accept*, sec-fetch-*, and sec-ch-ua*
                },
                locale=locale,
                service_workers="block",
                user_agent=user_agent,
            )
            # Performance optimization using route blocker to abort non-essential requests
            await context.route("**/*", _route_blocker)
            self._contexts[context_key] = context

            # Evict if over capacity threshold (best-effort to avoid blocking)
            contexts_to_close = []
            while len(self._contexts) > self._max_contexts:
                key_to_expire, context_to_expire = self._contexts.popitem(last=False)
                if len(context_to_expire.pages) == 0:
                    contexts_to_close.append(context_to_expire.close())
                else:
                    # Put it back since we can’t evict now
                    self._contexts[key_to_expire] = context_to_expire
                    # Break if nothing can be evicted
                    if all(len(c.pages) > 0 for c in self._contexts.values()):
                        break

            concurrent_browser_contexts_gauge.set(len(self._contexts))

        await asyncio.gather(*contexts_to_close)
        return context

    async def fetch(
            self, url: str, user_id: int, user_agent: str, extra_headers: dict[str, str],
            timeout_ms: int = 15000, wait_selector: str | None = None,
    ) -> FetchResult:
        time_budget = TimeBudget.from_ms(timeout_ms)

        url_parts = urlsplit(url)
        if url_parts.hostname is None:
            raise RuntimeError(f"Invalid URL: {url}")
        if not url_parts.scheme:
            url_parts = urlsplit(f"https://{url}")

        url = url_parts.geturl()
        context = await self._get_or_create_context(url_parts, user_id, user_agent, extra_headers)
        page: Page = await context.new_page()
        try:
            resp: Response | None = await page.goto(
                url, wait_until="domcontentloaded", timeout=time_budget.remaining_ms()
            )
        except Exception:
            await page.close()
            raise

        resp_status = None
        resp_content_type = ""
        if resp is not None:
            resp_status = resp.status
            resp_content_type = resp.headers.get("content-type", "").lower()

        if resp_status is None or resp_status >= 400:
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.SKIPPED_ERROR,
                status_code=resp_status,
                text="",
                url=page.url
            )
        if "application/pdf" in resp_content_type or "octet-stream" in resp_content_type:
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.SKIPPED_NOT_TEXT,
                status_code=resp_status,
                text="",
                url=page.url
            )
        if "application/json" in resp_content_type:
            resp_text = await resp.text()
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.PASSED_THROUGH_NOT_TEXT,
                status_code=resp_status,
                text=resp_text,
                url=page.url
            )
        if ("text/plain" in resp_content_type
                or "text/markdown" in resp_content_type
                or "text/x-markdown" in resp_content_type):
            resp_text = await resp.text()
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.PASSED_THROUGH_NOT_TEXT,
                status_code=resp_status,
                text=_normalize_newlines(resp_text),
                url=page.url
            )

        remaining_ms = time_budget.remaining_ms()
        if remaining_ms <= 0:
            html = await page.content()
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.PASSED_THROUGH_RAW_HTML,
                status_code=resp_status,
                text=html,
                url=page.url
            )

        try:
            if wait_selector:
                await page.wait_for_selector(wait_selector, state="visible", timeout=remaining_ms)

            remaining_ms = time_budget.remaining_ms()
            if remaining_ms >= 1000:
                # Wait for page to settle if time allows
                await _wait_for_page_to_settle(page, max_ms=(remaining_ms - 200))  # Leave room for text extraction

            canonical_url_task = asyncio.create_task(page.evaluate(
                """
                () => document.querySelector('link[rel="canonical"]')?.href || ''
                """
            ))

            await _expand_interactive_docs(page)
            #await _normalize_math_to_tex(page)
            html = await page.content()

            canonical_url = await canonical_url_task
            async with self._thread_pool_semaphore:
                text = await asyncio.get_running_loop().run_in_executor(
                    self._thread_pool, _process_content_html, html, (canonical_url or page.url)
                )

            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.EXTRACTED,
                status_code=resp_status,
                text=text,
                url=page.url
            )
        except asyncio.CancelledError:
            await page.close()
            raise
        except Exception as exc:
            logger.error(f"Error while fetching {url}", exc_info=exc)
            html = await page.content()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.PASSED_THROUGH_RAW_HTML,
                status_code=resp_status,
                text=html,
                url=page.url
            )
        finally:
            await page.close()

    async def start(self):
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self._browser is None:
            self._browser = await self._playwright.chromium.launch(headless=True)

    async def stop(self):
        if self._browser is not None:
            async with self._contexts_lock:
                contexts_to_close = []
                pages_to_close = []
                while self._contexts:
                    _, context_to_expire = self._contexts.popitem(last=False)
                    for page in context_to_expire.pages:
                        pages_to_close.append(page.close())
                    contexts_to_close.append(context_to_expire.close())
                await asyncio.gather(*pages_to_close)
                await asyncio.gather(*contexts_to_close)
                concurrent_browser_contexts_gauge.set(0)
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        self._thread_pool.shutdown(wait=False, cancel_futures=True)


def _aside_to_markdown(aside_tag: bs4.element.Tag, indent: str = ""):
    class_attrs = aside_tag.get("class", bs4.element.AttributeValueList())
    if "__normalized__" in class_attrs:
        return

    for c in aside_tag.contents:
        if isinstance(c, bs4.element.NavigableString):
            c.replace_with(bs4.element.NavigableString(c.get_text().replace("\n", " ")))
        elif isinstance(c, bs4.element.Tag):
            if c.name == "p":
                _normalize_paragraph(c)
            elif c.name == "span":
                _normalize_span(c)
    # TODO: handle nested block quotes
    aside_tag.insert(0, bs4.element.NavigableString(f"\n{indent}> "))

    class_attrs.append("__normalized__")
    aside_tag["class"] = class_attrs


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


def _list_to_markdown(soup: bs4.BeautifulSoup):
    for list_tag in soup.find_all(["ol", "ul"], recursive=True):
        class_attrs = list_tag.get("class", bs4.element.AttributeValueList())
        if "__normalized__" in class_attrs:
            continue

        for list_tag_content in list_tag.contents:
            # Remove newlines and empty texts between <li> tags
            if isinstance(list_tag_content, bs4.element.NavigableString):
                if not list_tag_content.get_text().strip():
                    list_tag_content.decompose()

        is_ordered = (list_tag.name == 'ol')

        # Determine starting index for ordered lists (default 1)
        start = 1
        if is_ordered and list_tag.has_attr('start'):
            try:
                start = int(list_tag['start'])
            except ValueError:
                start = 1

        parent_ol_cnt = len(list_tag.find_parents("ol"))
        parent_ul_cnt = len(list_tag.find_parents("ul"))
        indent = (" " * 3 * parent_ol_cnt) + (" " * 2 * parent_ul_cnt)
        inner_indent = indent + (" " * (3 if is_ordered else 2))
        index = start
        for li_tag in list_tag.find_all('li', recursive=False):
            for li_content_idx, li_tag_content in enumerate(li_tag.contents.copy()):
                if isinstance(li_tag_content, bs4.element.NavigableString):
                    li_content_el_text = li_tag_content.get_text()
                    if li_content_el_text == "\n":
                        li_tag_content.decompose()
                        continue
                    else:
                        li_content_el_text = re.sub(r"[\n\s]+", " ", li_content_el_text)
                    # li_tag_content.replace_with(bs4.element.NavigableString(li_content_el_text))
                    li_tag_content.replace_with(bs4.element.NavigableString(f"({li_content_el_text})"))
                elif isinstance(li_tag_content, bs4.element.Tag):
                    if li_tag_content.name == "p":
                        _normalize_paragraph(li_tag_content, inner_indent)
                    else:
                        for pre_tag in li_tag_content.select("pre", recursive=False):
                            _pre_to_markdown(pre_tag,
                                             has_next_sibling=(li_content_idx <= len(li_tag.contents) - 1),
                                             has_previous_sibling=(li_content_idx != 0),
                                             indent=inner_indent)
            if is_ordered:
                prefix = f"{index}. "
                index += 1
            else:
                prefix = "- "
            li_tag.insert(0, bs4.element.NavigableString(f"\n{indent}{prefix}"))

            for aside_tag in li_tag.find_all("aside", recursive=True):
                _aside_to_markdown(aside_tag, inner_indent)

        if list_tag.next_sibling:
            list_tag.append(bs4.element.NavigableString(f"\n{indent}"))

        class_attrs.append("__normalized__")
        list_tag["class"] = class_attrs


async def _normalize_math_to_tex(page: Page):
    with suppress(Exception):
        # MathJax v3+: wait and replace rendered math with TeX
        await page.evaluate(
            """
            async () => {
              if (window.MathJax?.typesetPromise) {
                await MathJax.typesetPromise();
                try {
                  for (const m of (MathJax.startup?.document?.math || [])) {
                    const tex = m.math;
                    const root = m.typesetRoot;
                    if (root && tex) root.replaceWith(document.createTextNode(tex));
                  }
                } catch {}
              }
            }
            """
        )
        # KaTeX: replace widget with original TeX from <annotation encoding="application/x-tex">
        await page.evaluate(
            """
            () => {
              document.querySelectorAll('.katex .katex-mathml annotation[encoding="application/x-tex"]').forEach(a => {
                const tex = a.textContent || '';
                const container = a.closest('.katex');
                if (tex && container) container.replaceWith(document.createTextNode(tex));
              });
            }
            """
        )


def _normalize_paragraph(p_tag: bs4.element.Tag, indent: str = ""):
    class_attrs = p_tag.get("class", bs4.element.AttributeValueList())
    if "__normalized__" in class_attrs:
        return

    p_tag_text = p_tag.get_text().strip()
    if p_tag_text == "":
        p_tag.decompose()
        return

    for c in p_tag.contents:
        if isinstance(c, bs4.element.NavigableString):
            text = _normalize_whitespace(c.get_text())
            #c.replace_with(bs4.element.NavigableString(text))
            c.replace_with(bs4.element.NavigableString(f"[{text}]"))
        elif isinstance(c, bs4.element.Tag):
            if c.name == "a":
                text = _normalize_whitespace(c.get_text())
                c.replace_with(bs4.element.NavigableString(text))
    # if not p_tag.find_parents(["ol", "ul"]):
    #    p_tag.append(soup.new_string("\n"))

    class_attrs.append("__normalized__")
    p_tag["class"] = class_attrs


def _normalize_span(span_tag: bs4.element.Tag):
    class_attrs = span_tag.get("class", bs4.element.AttributeValueList())
    if "__normalized__" in class_attrs:
        return

    for c in span_tag.contents:
        if isinstance(c, bs4.element.NavigableString):
            text = _normalize_whitespace(c.get_text())
            c.replace_with(bs4.element.NavigableString(text))

    class_attrs.append("__normalized__")
    span_tag["class"] = class_attrs


def _normalize_newlines(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _pre_to_markdown(
        pre_tag: bs4.element.Tag,
        has_next_sibling: bool = False,
        has_previous_sibling: bool = False,
        indent: str = "",
):
    class_attrs = pre_tag.get("class", bs4.element.AttributeValueList())
    if "__normalized__" in class_attrs:
        return

    partially_indented_text = pre_tag.get_text().strip().replace("\n", f"\n{indent}")
    markdown_text = ""
    if has_previous_sibling or pre_tag.previous_sibling:
        markdown_text += f"\n{indent}"
    markdown_text += (
        f"```"
        f"\n{indent}{partially_indented_text}"
        f"\n{indent}```"
    )
    if has_next_sibling or pre_tag.next_sibling:
        markdown_text += f"\n{indent}"
    pre_tag.replace_with(bs4.element.NavigableString(markdown_text))

    class_attrs.append("__normalized__")
    pre_tag["class"] = class_attrs


def _process_content_html(html: str, base_url: str) -> str:
    #return html
    soup = BeautifulSoup(html, "lxml")

    for sel in [
        ".breadcrumb, [class*=breadcrumb], [id*=breadcrumb]",
        "[class*=nocontent]",
        #"[class*=search]",
        "[role*=navigation]",
        "[role*=presentation]",
        "devsite-feature-tooltip, devsite-thumb-rating",
        "head",
        "nav, [class*=-nav], [class*=nav-]",
        "noscript, script, style, template",
        'button, .button, [class*="button"], [id*="button"], [role*="button"]',
        ## common site chrome that sometimes slips into Readability content
        #".feedback, [class*=feedback], [id*=feedback]",
        #".sr-only, .visually-hidden",
        #".toc, [class*=toc], [id*=toc]",
        #".toolbar, [class*=toolbar], [id*=toolbar]",
        ## Strip code line-number scaffolding
        #'.hljs-ln-numbers',
        #'.line-numbers',
        #'.linenodiv',
        #'table.linenos',
        #'td.gutter',
        #'td.linenos',
    ]:
        for el in soup.select(sel):
            el.decompose()

    #return _normalize_newlines(str(soup))

    for a_tag in soup.find_all("a", href=False):
        a_tag_text = a_tag.get_text().replace("\n", " ")
        if not a_tag_text.strip():
            a_tag.decompose()
        else:
            a_tag.replace_with(bs4.element.NavigableString(a_tag_text))

    # Normalize <br> into explicit newlines
    for br in soup.select("br"):
        br.replace_with(soup.new_string("\n"))

    # Convert headings to Markdown
    for tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        for el in soup.find_all(tag):
            el_text = el.get_text(separator=" ", strip=True)
            el.replace_with(soup.new_string(f"\n{int(tag.lstrip('h')) * '#'} {el_text}\n"))

    for el in soup.find_all("strong"):
        el_text = el.get_text()
        el.replace_with(soup.new_string(f"**{el_text}**"))

    for el in soup.find_all("code"):
        el_text = el.get_text()
        if el.parent.name != "pre":
            el.replace_with(soup.new_string(f"`{el_text}`"))

    # Convert data tables to Markdown
    _list_to_markdown(soup)

    # Convert data tables to Markdown
    #for tbl in soup.find_all("table"):
    #    # headers
    #    thead = tbl.find("thead")
    #    headers = []
    #    if thead:
    #        th = thead.find_all("th")
    #        headers = [h.get_text(separator=" ", strip=True) for h in th]
    #    else:
    #        # some themes use first row as headers
    #        first = tbl.find("tr")
    #        if first and first.find_all("th"):
    #            headers = [c.get_text(separator=" ", strip=True)
    #                       for c in first.find_all(["th","td"], recursive=False)]

    #    rows = []
    #    if headers:
    #        rows.append("| " + " | ".join(headers) + " |")
    #        rows.append("| " + " | ".join("---" for _ in headers) + " |")

    #    # body
    #    for tr in tbl.find_all("tr"):
    #        tds = tr.find_all(["td", "th"], recursive=False)
    #        if not tds:
    #            continue
    #        if headers and tr.find_parent("thead"):
    #            continue
    #        row = [td.get_text(separator=" ", strip=True) for td in tds]
    #        rows.append("| " + " | ".join(row) + " |")

    #    md = "\n".join(rows).strip()
    #    if md:
    #        tbl.replace_with(soup.new_string("\n\n" + md + "\n\n"))

    for aside_tag in soup.find_all("aside", recursive=True):
        _aside_to_markdown(aside_tag)
    for pre_tag in soup.select("pre", recursive=False):
        _pre_to_markdown(pre_tag)

    # Turn <a> into "text (URL)"
    #for a in soup.select('a[href]'):
    #    if not a.get_text(strip=True):
    #        continue

    #    # Skip obvious navigation/chrome or code/pre
    #    if a.find_parent(["nav", "header", "footer", "aside", "pre", "code"]):
    #        continue

    #    # Skip anchors inside obvious navigation containers
    #    skip = False
    #    for parent in a.parents:
    #        if parent is None or not getattr(parent, "name", None):
    #            break
    #        # Semantic nav tags / roles
    #        if parent.name in {"nav"}:
    #            skip = True
    #            break
    #        role = (parent.get("role") or "").lower()
    #        if "navigation" in role:
    #            skip = True
    #            break
    #        # Common chrome classes/ids
    #        parent_id = " ".join(filter(None, [parent.get("id", "")]))
    #        parent_cls = parent.get("class", bs4.element.AttributeValueList())
    #        if isinstance(parent_cls, (list, tuple)):
    #            parent_cls = " ".join(parent_cls)
    #        chrome_blob = f"{parent_id} {parent_cls}".lower()
    #        if any(tok in chrome_blob for tok in ("nav", "menu", "breadcrumb", "tabs", "pager")):
    #            skip = True
    #            break
    #    if skip:
    #        continue

    #    if a["href"].lstrip().startswith("#"):
    #        continue

    #    href = urljoin(base_url or "", a["href"])
    #    parsed_href = urlparse(href)

    #    scheme = (parsed_href.scheme or "https").lower()
    #    if scheme not in {"http", "https", "mailto", "tel"}:
    #        continue

    #    # Remove common trackers for http(s)
    #    if scheme in {"http", "https"}:
    #        netloc = parsed_href.hostname or ""
    #        if parsed_href.port:
    #            netloc = f"{netloc}:{parsed_href.port}"
    #        qry = [(k, v) for (k, v) in parse_qsl(parsed_href.query, keep_blank_values=True)
    #               if k not in {
    #                   # Tracking params
    #                   "fbclid",
    #                   "gclid",
    #                   "mc_cid",
    #                   "mc_eid",
    #                   "utm_campaign",
    #                   "utm_content",
    #                   "utm_medium",
    #                   "utm_source",
    #                   "utm_term",
    #               }]
    #        parsed_href = parsed_href._replace(netloc=netloc, query=urlencode(qry, doseq=True))

    #    a.insert_after(soup.new_string(f" ({parsed_href.geturl()})"))
    #    a.unwrap()

    # Normalized in paragraph newlines
    # TODO: process in list paragraphs separately - maybe first
    for p_tag in soup.select("p"):
        _normalize_paragraph(p_tag)

    text = soup.get_text()
    #text = str(soup)
    text = _normalize_newlines(text)
    text = text.replace("\xa0", " ")
    return text
    #return ""


async def _route_blocker(route: Route, request: Request):
    request_hostname = (urlparse(request.url).hostname or "").lower()
    resource_type = request.resource_type

    if resource_type in {
        "beacon",
        "font",
        "image",
        "media",
    }:
        return await route.abort()

    # Block analytics server-sent events
    if resource_type == "eventsource" and any(request_hostname.endswith(h) for h in blocked_analytics_hosts):
        return await route.abort()

    # Block analytics websockets
    if resource_type == "websocket":
        if any(request_hostname.endswith(h) for h in blocked_analytics_hosts):
            return await route.abort()

    # Block analytics XHR/fetch
    if any(request_hostname.endswith(h) for h in blocked_analytics_hosts) and not request.is_navigation_request():
        return await route.abort()

    await route.continue_()


async def _wait_for_page_to_settle(page: Page, quiet_ms: float = 2000.0, max_ms: float = 10000.0):
    time_budget = TimeBudget.from_ms(max_ms)
    with suppress(Exception):
        # 1. Initial HTML is parsed and most sub-resources (images, stylesheets, iframes) have finished loading.
        await page.wait_for_load_state("load", timeout=time_budget.remaining_ms())

        # 2. Network idleness (best-effort, capped to rely more on DOM quiet)
        network_idle_budget = min(2000.0, time_budget.remaining_ms())
        if network_idle_budget <= 0:
            return
        await page.wait_for_load_state("networkidle", timeout=network_idle_budget)

    # 3. DOM quiet period
    dom_quiet_budget = time_budget.remaining_ms()
    if dom_quiet_budget <= 0:
        return
    await page.evaluate(
        """
        (quiet, maxWait) => new Promise(resolve => {
            let obs = null;
            let timer = null;
            const done = () => {
                try { if (obs) obs.disconnect(); } catch {}
                if (timer) { clearTimeout(timer); timer = null; }
                resolve();
            };
            obs = new MutationObserver(() => {
                if (timer) clearTimeout(timer);
                timer = setTimeout(done, quiet);
            });
            // If already quiet, this resolves after quiet
            timer = setTimeout(done, quiet);
            try { obs.observe(document, {subtree:true, childList:true, attributes:true, characterData:true}); } catch {}
            setTimeout(done, maxWait); // hard cap
        })
        """,
        (min(quiet_ms, dom_quiet_budget), dom_quiet_budget)
    )
