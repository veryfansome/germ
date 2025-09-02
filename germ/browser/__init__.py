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
from readability import Document
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
    title: str = ""
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
        self._readability_js = Path(__file__).with_name("Readability.min.js").read_text(encoding="utf-8")
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
            # Inject Readability.js
            await context.add_init_script(script=self._readability_js)
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
                text=_normalize_whitespace(resp_text),
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
            await _normalize_math_to_tex(page)
            title, html = await _extract_content_html_with_readability_js(page)
            if not html:
                html = await page.content()
                async with self._thread_pool_semaphore:
                    title, html = await asyncio.get_running_loop().run_in_executor(
                        self._thread_pool, _extract_content_html_with_readability_lxml, html
                    )

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
                title=title,
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


async def _extract_content_html_with_readability_js(page: Page) -> tuple[str, str] | tuple[None, None]:
    with suppress(Exception):
        article = await page.evaluate(
            """
            () => {
              // Clone to avoid live mutations during parse
              const doc = document.cloneNode(true);
              if (typeof Readability !== "function" && typeof Readability !== "object") return null;
              const reader = new Readability(doc, { keepClasses: false });
              const res = reader.parse();
              if (!res || !res.content) return null;
              const meta = (sel) => document.querySelector(sel)?.content || null;
              const title =
                res.title ||
                meta('meta[property="og:title"]') ||
                document.querySelector('article h1, main h1')?.innerText?.trim() ||
                document.title || "";
              return { title, content: res.content };
            }
            """
        )
        if article and article.get("content"):
            return article.get("title") or "", article["content"]
    return None, None


def _extract_content_html_with_readability_lxml(html: str) -> tuple[str, str]:
    doc = Document(html)
    title = doc.short_title()
    content_html = doc.summary(html_partial=True)
    return title, content_html


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


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text


def _process_content_html(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Strip code line-number scaffolding
    for sel in [
        '.hljs-ln-numbers'
        '.line-numbers',
        '.linenodiv',
        'table.linenos',
        'td.gutter',
        'td.linenos',
    ]:
        for el in soup.select(sel):
            el.decompose()

    # Convert data tables to Markdown
    for tbl in soup.find_all("table"):
        # headers
        thead = tbl.find("thead")
        headers = []
        if thead:
            th = thead.find_all("th")
            headers = [h.get_text(separator=" ", strip=True) for h in th]
        else:
            # some themes use first row as headers
            first = tbl.find("tr")
            if first and first.find_all("th"):
                headers = [c.get_text(separator=" ", strip=True) for c in first.find_all(["th","td"], recursive=False)]

        rows = []
        if headers:
            rows.append("| " + " | ".join(headers) + " |")
            rows.append("| " + " | ".join("---" for _ in headers) + " |")

        # body
        for tr in tbl.find_all("tr"):
            tds = tr.find_all(["td", "th"], recursive=False)
            if not tds:
                continue
            if headers and tr.find_parent("thead"):
                continue
            row = [td.get_text(separator=" ", strip=True) for td in tds]
            rows.append("| " + " | ".join(row) + " |")

        md = "\n".join(rows).strip()
        if md:
            tbl.replace_with(soup.new_string("\n\n" + md + "\n\n"))

    # Turn <a> into "text (URL)"
    for a in soup.select('a[href]'):
        if not a.get_text(strip=True):
            continue

        # Skip obvious navigation/chrome or code/pre
        if a.find_parent(["nav", "header", "footer", "aside", "pre", "code"]):
            continue

        # Skip anchors inside obvious navigation containers
        skip = False
        for parent in a.parents:
            if parent is None or not getattr(parent, "name", None):
                break
            # Semantic nav tags / roles
            if parent.name in {"nav"}:
                skip = True
                break
            role = (parent.get("role") or "").lower()
            if "navigation" in role:
                skip = True
                break
            # Common chrome classes/ids
            parent_id = " ".join(filter(None, [parent.get("id", "")]))
            parent_cls = parent.get("class", bs4.element.AttributeValueList())
            if isinstance(parent_cls, (list, tuple)):
                parent_cls = " ".join(parent_cls)
            chrome_blob = f"{parent_id} {parent_cls}".lower()
            if any(tok in chrome_blob for tok in ("nav", "menu", "breadcrumb", "tabs", "pager")):
                skip = True
                break
        if skip:
            continue

        if a["href"].lstrip().startswith("#"):
            continue

        href = urljoin(base_url or "", a["href"])
        parsed_href = urlparse(href)

        scheme = (parsed_href.scheme or "https").lower()
        if scheme not in {"http", "https", "mailto", "tel"}:
            continue

        # Remove common trackers for http(s)
        if scheme in {"http", "https"}:
            netloc = parsed_href.hostname or ""
            if parsed_href.port:
                netloc = f"{netloc}:{parsed_href.port}"
            qry = [(k, v) for (k, v) in parse_qsl(parsed_href.query, keep_blank_values=True)
                   if k not in {
                       # Tracking params
                       "fbclid",
                       "gclid",
                       "mc_cid",
                       "mc_eid",
                       "utm_campaign",
                       "utm_content",
                       "utm_medium",
                       "utm_source",
                       "utm_term",
                   }]
            parsed_href = parsed_href._replace(netloc=netloc, query=urlencode(qry, doseq=True))

        a.insert_after(soup.new_string(f" ({parsed_href.geturl()})"))
        a.unwrap()

    # Keep pre/code; flatten tables to text
    text = soup.get_text(separator="\n", strip=True)
    text = _normalize_whitespace(text)
    return text


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
