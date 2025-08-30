import asyncio
import logging
import re
from bs4 import BeautifulSoup
from collections import OrderedDict
from playwright.async_api import Browser, BrowserContext, Page, Playwright, Response, async_playwright
from traceback import format_exc
from urllib.parse import urlsplit

from germ.settings import germ_settings
from germ.utils import TimeBudget

logger = logging.getLogger(__name__)


class WebBrowser:
    def __init__(self, max_context: int = 32):
        self._browser: Browser | None = None
        self._cache = {}
        self._context_lock: asyncio.Lock = asyncio.Lock()
        self._contexts: OrderedDict[str, BrowserContext] = OrderedDict()
        self._max_context = max_context
        self._p: Playwright | None = None

    async def _get_or_create_context(self, hostname: str, user_agent: str) -> BrowserContext:
        async with self._context_lock:
            if hostname in self._contexts:
                # LRU: mark as recently used
                context = self._contexts.pop(hostname)
                self._contexts[hostname] = context
                return context

            context = await self._browser.new_context(
                locale=germ_settings.GERM_BROWSER_LOCALE,
                user_agent=user_agent,
            )
            # Abort non-essential requests
            await context.route("**/*", lambda r: (
                r.abort()
                if r.request.resource_type in {"image", "media", "font"} or
                   any(d in r.request.url for d in ["ads.", "doubleclick", "metrics", "analytics"])
                else r.continue_()
            ))
            self._contexts[hostname] = context

            # Evict if over capacity
            contexts_to_close = []
            while len(self._contexts) > self._max_context:
                _, context_to_expire = self._contexts.popitem(last=False)
                contexts_to_close.append(context_to_expire.close())

        await asyncio.gather(*contexts_to_close)
        return context

    async def fetch(self, url: str, user_agent: str,
                    wait_selector: str | None = None, timeout_ms: int = 15000):
        time_budget = TimeBudget.from_ms(timeout_ms)

        url_hostname = urlsplit(f"https://{url}" if "://" not in url else url).hostname
        context = await self._get_or_create_context(url_hostname, user_agent)
        page: Page = await context.new_page()
        try:
            resp: Response | None = await page.goto(
                url, wait_until="domcontentloaded", timeout=time_budget.remaining_ms()
            )

            resp_status = resp.status if resp else None
            if resp_status is None or resp_status >= 400:
                await page.close()
                return page.url, "", resp_status

            resp_content_type = resp.headers.get("content-type").lower() if resp else ""
            if "application/json" in resp_content_type:
                body = await resp.text()
                await page.close()
                return page.url, body, resp_status
            if "application/pdf" in resp_content_type or "octet-stream" in resp_content_type:
                await page.close()
                return page.url, "", resp_status

        except asyncio.CancelledError:
            await page.close()
            raise
        except Exception:
            logger.error(f"Error while fetching {url}\n{format_exc()}")
            await page.close()
            return page.url, "", None

        try:
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=time_budget.remaining_ms())
            else:
                await wait_for_page_to_settle(page, max_ms=time_budget.remaining_ms())
            dom_text = await page.inner_text("body", timeout=time_budget.remaining_ms())
            return page.url, dom_text, resp_status
        except Exception:
            if time_budget.remaining_ms() <= 0:
                return page.url, "", resp_status
            html = await page.content()
            dom_text = extract_visible_text_from_html(html)
            return page.url, dom_text, resp_status
        finally:
            await page.close()

    async def start(self):
        if self._p is None:
            self._p = await async_playwright().start()
        if self._browser is None:
            self._browser = await self._p.chromium.launch(headless=True)

    async def stop(self):
        if self._browser is not None:
            await self._browser.close()
        if self._p is not None:
            await self._p.stop()


def extract_visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{2,}", "\n\n", text)


async def wait_for_page_to_settle(page: Page, quiet_ms: int = 800, max_ms: int = 10000):
    # 1. Initial HTML is parsed and most sub-resources (images, stylesheets, iframes) have finished loading.
    # 2. Network idleness and DOM quiet period as a best-effort proxy for completion of client-side rendering.
    time_budget = TimeBudget.from_ms(max_ms)
    try:
        await page.wait_for_load_state("load", timeout=time_budget.remaining_ms())
        await page.wait_for_load_state("networkidle", timeout=time_budget.remaining_ms() // 2)
    except Exception:
        pass
    finally:
        if time_budget.remaining_ms() <= 0:
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
        (min(quiet_ms, time_budget.remaining_ms()), time_budget.remaining_ms())
    )
