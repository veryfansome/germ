import asyncio
import logging
import re
from bs4 import BeautifulSoup
from collections import OrderedDict
from contextlib import suppress
from enum import Enum
from playwright.async_api import Browser, BrowserContext, Page, Playwright, Response, async_playwright
from pydantic import BaseModel
from urllib.parse import SplitResult, urlsplit

from germ.utils import TimeBudget

logger = logging.getLogger(__name__)


class FetchResult(BaseModel):
    content_type: str
    extraction_status: "TextExtractionStatus"
    status_code: int | None
    text: str
    url: str


class TextExtractionStatus(Enum):
    SKIPPED = 0
    DONE = 1


class WebBrowser:
    def __init__(self, max_context: int = 4):
        self._browser: Browser | None = None
        self._cache = {}
        self._context_lock: asyncio.Lock = asyncio.Lock()
        self._contexts: OrderedDict[tuple[str, str | None, int | None], BrowserContext] = OrderedDict()
        self._max_context = max_context
        self._playwright: Playwright | None = None

    async def _get_or_create_context(
            self, url_split: SplitResult, accept_language: str, user_agent: str
    ) -> BrowserContext:
        async with self._context_lock:
            if (url_split.scheme, url_split.hostname, url_split.port) in self._contexts:
                # LRU: mark as recently used
                context = self._contexts.pop((url_split.scheme, url_split.hostname, url_split.port))
                self._contexts[(url_split.scheme, url_split.hostname, url_split.port)] = context
                return context

            locale_parts = accept_language.split(",")
            context = await self._browser.new_context(
                locale="en-US" if len(locale_parts) == 1 else locale_parts[0],  # Fallback to en-US
                user_agent=user_agent,
            )
            # Abort non-essential requests
            await context.route("**/*", lambda r: (
                r.abort()
                if r.request.resource_type in {"image", "media", "font"} or
                   any(d in r.request.url for d in ["ads.", "doubleclick", "metrics", "analytics"])
                else r.continue_()
            ))
            self._contexts[(url_split.scheme, url_split.hostname, url_split.port)] = context

            # Evict if over capacity
            contexts_to_close = []
            while len(self._contexts) > self._max_context:
                _, context_to_expire = self._contexts.popitem(last=False)
                contexts_to_close.append(context_to_expire.close())

        await asyncio.gather(*contexts_to_close)
        return context

    async def fetch(
            self, url: str, accept_language: str, user_agent: str,
            enforce_https: bool = False,
            timeout_ms: int = 15000,
            wait_selector: str | None = None,
    ) -> FetchResult:
        time_budget = TimeBudget.from_ms(timeout_ms)

        url_parts = urlsplit(url)
        if url_parts.hostname is None:
            raise RuntimeError(f"Invalid URL: {url}")
        if not url_parts.scheme or (url_parts.scheme == "http" and enforce_https):
            url_parts.scheme = "https"

        url = url_parts.geturl()
        context = await self._get_or_create_context(url_parts, accept_language, user_agent)
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

        if (resp_status is None
                or resp_status >= 400
                or "application/pdf" in resp_content_type
                or "octet-stream" in resp_content_type
        ):
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.SKIPPED,
                status_code=resp_status,
                text="",
                url=page.url
            )
        elif "application/json" in resp_content_type:
            resp_text = await resp.text()
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.SKIPPED,
                status_code=resp_status,
                text=resp_text,
                url=page.url
            )

        text_extraction_budget = time_budget.remaining_ms()
        if text_extraction_budget <= 0:
            html = await page.content()
            await page.close()
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.SKIPPED,
                status_code=resp_status,
                text=html,
                url=page.url
            )

        try:
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=text_extraction_budget)
            else:
                await wait_for_page_to_settle(page, max_ms=text_extraction_budget)

            dom_text = await page.inner_text("body", timeout=time_budget.remaining_ms())

            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.DONE,
                status_code=resp_status,
                text=dom_text,
                url=page.url
            )
        except asyncio.CancelledError:
            await page.close()
            raise
        except Exception:
            html = await page.content()
            if time_budget.remaining_ms() <= 0:
                return FetchResult(
                    content_type=resp_content_type,
                    extraction_status=TextExtractionStatus.DONE,
                    status_code=resp_status,
                    text=html,
                    url=page.url
                )
            elif len(html) > 100_000:  # Threshold for offloading larger docs to unblock event loop
                dom_text = extract_visible_text_from_html(html)
            else:
                dom_text = await asyncio.to_thread(extract_visible_text_from_html, html)
            return FetchResult(
                content_type=resp_content_type,
                extraction_status=TextExtractionStatus.DONE,
                status_code=resp_status,
                text=dom_text,
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
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()


def extract_visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    return re.sub(r"\n{2,}", "\n\n", text)


async def wait_for_page_to_settle(page: Page, quiet_ms: int = 800, max_ms: int = 10000):
    # 1. Initial HTML is parsed and most sub-resources (images, stylesheets, iframes) have finished loading.
    # 2. Network idleness and DOM quiet period as a best-effort proxy for completion of client-side rendering.
    time_budget = TimeBudget.from_ms(max_ms)
    with suppress(Exception):
        await page.wait_for_load_state("load", timeout=time_budget.remaining_ms())

        network_idle_budget = time_budget.remaining_ms()
        if network_idle_budget <= 0:
            return
        await page.wait_for_load_state("networkidle", timeout=network_idle_budget)

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
