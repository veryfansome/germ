import re
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


def visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return re.sub(r"\n{2,}", "\n\n", text)


async def fetch_page_text(url: str, wait_selector: str|None=None, timeout_ms=15000):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        resp_status = resp.status if resp else None

        if resp_status >= 400:
            final_url = page.url
            await browser.close()
            return final_url, "", resp_status

        if wait_selector:
            await page.wait_for_selector(wait_selector, timeout=timeout_ms)

        try:
            dom_text = await page.inner_text("body", timeout=5000)
        except Exception:
            html = await page.content()
            dom_text = visible_text_from_html(html)

        final_url = page.url
        await browser.close()
        return final_url, dom_text, resp_status
