import aiofiles
import aiofiles.os
import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from playwright.async_api import Browser, BrowserContext, Page, Playwright, Request, Response, Route, async_playwright
from prometheus_client import Gauge
from pydantic import BaseModel
from urllib.parse import SplitResult, urlparse, urlsplit

from germ.settings import germ_settings
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
    location: str | None = None
    url: str
    url_sig: str | None = None

    # TODO:
    #   - Extract links and click-ables
    #   - Extract summary and features

    @classmethod
    async def save_in_page_pdf(cls, page: Page, resp: Response,
                               data_dir: str = f"{germ_settings.GERM_DATA_DIR}/webpage") -> "FetchResult":
        sig = _url_signature(resp.url)
        location = f"{data_dir}/{sig}.pdf"
        resp_body = await resp.body()
        await aiofiles.os.makedirs(data_dir, exist_ok=True)
        async with aiofiles.open(location, "wb") as f:
            await f.write(resp_body)
        await page.close()
        return cls(
            location=location,
            url=resp.url,
            url_sig=sig,
        )

    @classmethod
    async def save_page_as_pdf(cls, page: Page,
                               data_dir: str = f"{germ_settings.GERM_DATA_DIR}/webpage") -> "FetchResult":
        sig = _url_signature(page.url)
        location = f"{data_dir}/{sig}.pdf"
        await aiofiles.os.makedirs(data_dir, exist_ok=True)
        await page.emulate_media(media="screen")
        await page.pdf(path=location, format="A2", print_background=True)
        await page.close()
        return cls(
            location=location,
            url=page.url,
            url_sig=sig,
        )


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
                reduced_motion="reduce",
                service_workers="block",
                user_agent=user_agent,
            )
            # Performance optimization using route blocker to abort non-essential requests
            await context.route("**/*", _route_blocker)
            await context.add_init_script(
                """
                (() => {
                    try {
                        // Quiet analytics pings that _route_blocker can't see as a resource_type
                        Object.defineProperty(navigator, 'sendBeacon', {
                            value: function() { return true; },
                            configurable: true
                        });
                    } catch (_) {}
                })();
                """
            )
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

    async def fetch_url(
            self, url: str, user_id: int, user_agent: str, extra_headers: dict[str, str],
            timeout_ms: int = 15000, wait_selector: str | None = None,
    ) -> FetchResult:
        time_budget = TimeBudget.from_ms(timeout_ms)

        url_parts = _split_and_normalize_url(url)
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
            raise RuntimeError(f"Failed to load page: {url}")
        elif "application/pdf" in resp_content_type:
            return await FetchResult.save_in_page_pdf(page, resp)
        elif "application/xhtml+xml" not in resp_content_type and "text/html" not in resp_content_type:
            await page.close()
            raise RuntimeError(f"Unsupported content_type: {resp_content_type}")

        remaining_ms = time_budget.remaining_ms()
        if remaining_ms <= 0:
            return await FetchResult.save_page_as_pdf(page)

        try:
            if wait_selector:
                await page.wait_for_selector(wait_selector, state="visible", timeout=remaining_ms)

            remaining_ms = time_budget.remaining_ms()
            if remaining_ms >= 1000:
                # Wait for page to settle if time allows
                await _wait_for_page_to_settle(page, max_ms=(remaining_ms - 200))  # Leave room

            await _expand_interactive_docs(page)

            return await FetchResult.save_page_as_pdf(page)
        except asyncio.CancelledError:
            await page.close()
            raise
        except Exception as exc:
            logger.error(f"Error while waiting for page to settle: {url}", exc_info=exc)
            return await FetchResult.save_page_as_pdf(page)

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


def _split_and_normalize_url(url: str) -> SplitResult:
    parts = urlsplit(url)
    if not parts.scheme:
        parts = urlsplit(f"https://{url}")
    # Reject obviously invalid after normalization
    if not parts.hostname:
        raise RuntimeError(f"Invalid URL: {url}")
    return parts


async def _route_blocker(route: Route, request: Request):
    request_hostname = (urlparse(request.url).hostname or "").lower()
    resource_type = request.resource_type

    if resource_type in {
        "media",  # Keep images; drop audio/video streams
    }:
        return await route.abort()

    # Block analytics server-sent events
    if resource_type == "eventsource" and any(request_hostname.endswith(h) for h in blocked_analytics_hosts):
        return await route.abort()

    # Block analytics websockets
    if resource_type == "websocket" and any(request_hostname.endswith(h) for h in blocked_analytics_hosts):
        return await route.abort()

    # Block analytics XHR/fetch
    if any(request_hostname.endswith(h) for h in blocked_analytics_hosts) and not request.is_navigation_request():
        return await route.abort()

    await route.continue_()


def _url_signature(url: str, size: int = 16) -> str:  # 16 bytes = 128-bit
    h = hashlib.blake2s(url.encode('utf-8'), digest_size=size)
    return h.hexdigest()


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
