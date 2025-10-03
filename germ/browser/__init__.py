import aiofiles
import aiofiles.os
import asyncio
import hashlib
import logging
from playwright.async_api import Page, Response
from pydantic import BaseModel

from germ.browser.base import BaseBrowser
from germ.settings import germ_settings

logger = logging.getLogger(__name__)


class FetchResult(BaseModel):
    location: str | None = None
    url: str
    url_sig: str | None = None

    # TODO:
    #   - Extract links and click-ables
    #   - Extract summary and features

    @classmethod
    async def save_in_page_pdf(cls, page: Page, resp: Response, data_dir: str) -> "FetchResult":
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
    async def save_page_as_pdf(cls, page: Page, data_dir: str) -> "FetchResult":
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


class PageFetchingWebBrowser(BaseBrowser):
    def __init__(self, max_contexts: int = 24):
        super().__init__(max_contexts)

    async def fetch_url(
            self, url: str, user_id: int, user_agent: str, extra_headers: dict[str, str],
            data_dir: str = f"{germ_settings.GERM_DATA_DIR}/webpage",
            timeout_ms: int = 15000, wait_selector: str | None = None,
    ) -> FetchResult:
        time_budget, resp, page = await self.goto_url(url, user_id, user_agent, extra_headers, timeout_ms=timeout_ms)

        resp_status = None
        resp_content_type = ""
        if resp is not None:
            resp_status = resp.status
            resp_content_type = resp.headers.get("content-type", "").lower()

        if resp_status is None or resp_status >= 400:
            await page.close()
            raise RuntimeError(f"Failed to load page: {url}")
        elif "application/pdf" in resp_content_type:
            return await FetchResult.save_in_page_pdf(page, resp, data_dir)
        elif "application/xhtml+xml" not in resp_content_type and "text/html" not in resp_content_type:
            await page.close()
            raise RuntimeError(f"Unsupported content_type: {resp_content_type}")

        remaining_ms = time_budget.remaining_ms()
        if remaining_ms <= 0:
            return await FetchResult.save_page_as_pdf(page, data_dir)

        try:
            if wait_selector:
                await page.wait_for_selector(wait_selector, state="visible", timeout=remaining_ms)

            remaining_ms = time_budget.remaining_ms()
            if remaining_ms >= 1000:
                # Wait for page to settle if time allows
                await self.wait_for_page_to_settle(page, max_ms=(remaining_ms - 200))  # Leave room

            await _expand_interactive_docs(page)

            return await FetchResult.save_page_as_pdf(page, data_dir)
        except asyncio.CancelledError:
            await page.close()
            raise
        except Exception as exc:
            logger.error(f"Error while waiting for page to settle: {url}", exc_info=exc)
            return await FetchResult.save_page_as_pdf(page, data_dir)


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


def _url_signature(url: str, size: int = 16) -> str:  # 16 bytes = 128-bit
    h = hashlib.blake2s(url.encode('utf-8'), digest_size=size)
    return h.hexdigest()
