import asyncio
import logging
import re
from bs4 import BeautifulSoup
from collections import OrderedDict
from contextlib import suppress
from enum import Enum
from playwright.async_api import Browser, BrowserContext, Page, Playwright, Request, Response, Route, async_playwright
from pydantic import BaseModel
from urllib.parse import SplitResult, urlparse, urlsplit

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
    def __init__(self, max_context: int = 24):
        self._browser: Browser | None = None
        self._cache = {}
        self._context_lock: asyncio.Lock = asyncio.Lock()
        self._contexts: OrderedDict[tuple[str, str | None, int | None], BrowserContext] = OrderedDict()
        self._max_context = max_context
        self._playwright: Playwright | None = None

    async def _get_or_create_context(
            self, url_split: SplitResult, user_agent: str, extra_headers: dict[str, str]
    ) -> BrowserContext:
        context_key = (url_split.scheme, url_split.hostname, url_split.port)
        async with self._context_lock:
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
                    **extra_headers,   # accept*, sec-fetch-*, and sec-ch-ua*
                },
                locale=locale,
                service_workers="block",
                user_agent=user_agent,
            )
            # Performance optimization using route blocker to abort non-essential requests
            await context.route("**/*", _route_blocker)
            self._contexts[context_key] = context

            # Evict if over capacity
            contexts_to_close = []
            while len(self._contexts) > self._max_context:
                key_to_expire, context_to_expire = self._contexts.popitem(last=False)
                if len(context_to_expire.pages) == 0:
                    contexts_to_close.append(context_to_expire.close())
                else:
                    # Put it back since we can’t evict now
                    self._contexts[key_to_expire] = context_to_expire
                    break

        await asyncio.gather(*contexts_to_close)
        return context

    async def fetch(
            self, url: str, user_agent: str, extra_headers: dict[str, str],
            timeout_ms: int = 15000, wait_selector: str | None = None,
    ) -> FetchResult:
        time_budget = TimeBudget.from_ms(timeout_ms)

        url_parts = urlsplit(url)
        if url_parts.hostname is None:
            raise RuntimeError(f"Invalid URL: {url}")
        if not url_parts.scheme:
            url_parts = urlsplit(f"https://{url}")

        url = url_parts.geturl()
        context = await self._get_or_create_context(url_parts, user_agent, extra_headers)
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
                await page.wait_for_selector(wait_selector, state="visible", timeout=text_extraction_budget)
            else:
                await _wait_for_page_to_settle(page, max_ms=text_extraction_budget)

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
                dom_text = await asyncio.to_thread(_extract_visible_text_from_html, html)
            else:
                dom_text = _extract_visible_text_from_html(html)
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


def _extract_visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    return re.sub(r"\n{2,}", "\n\n", text)


async def _route_blocker(route: Route, request: Request):
    rtype = request.resource_type
    if rtype in {
        "beacon",
        "font",
        "image",
        "media",
    }:
        await route.abort()
        return

    request_hostname = (urlparse(request.url).hostname or "").lower()
    blocked_hosts = (
        "doubleclick.net", "googletagmanager.com", "google-analytics.com",
        "adservice.google.com", "facebook.net", "mixpanel.com",
    )
    # Avoid breaking primary navigations; only block non-navigation subresources.
    if any(request_hostname.endswith(h) for h in blocked_hosts) and not request.is_navigation_request():
        await route.abort()
        return

    await route.continue_()


async def _wait_for_page_to_settle(page: Page, quiet_ms: int = 2000, max_ms: int = 10000):
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
