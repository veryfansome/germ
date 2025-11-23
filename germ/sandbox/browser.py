import asyncio

from germ.browser.agentic import AgenticBrowser


if __name__ == "__main__":
    from germ.observability.logging import setup_logging
    setup_logging()

    wb = AgenticBrowser()

    async def _main():
        await wb.start()
        url = "https://cloud.google.com/storage/docs/uploading-objects#rest-upload-objects"
        message_received_summary = "The user want to know how to access stored objects using python."

        #url = "https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/blockquote"
        #message_received_summary = "The user want to know when to use a blockquote."

        #url = "https://en.wikipedia.org/wiki/Drosophilidae"
        #message_received_summary = "The user want to know about fruitfly life cycle."

        #url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"

        #url = "https://en.wikipedia.org/wiki/PageRank"

        #url = "https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/storage_bucket"

        #url = "https://requests.readthedocs.io/en/latest/user/quickstart/#make-a-request"

        #url = "https://www.w3schools.com/tags/default.asp"

        try:
            result = await wb.scan_page_content_for_related_info(
                message_received_summary,
                url, 0,
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
                {
                    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'accept-encoding': 'gzip, deflate, br, zstd', 'accept-language': 'en-US,en;q=0.9',
                    'sec-ch-ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
                    'sec-ch-ua-mobile': '?0', 'sec-ch-ua-platform': '"macOS"', 'upgrade-insecure-requests': '1',
                    'sec-fetch-dest': 'document', 'referer': 'http://localhost:8001/register',
                    'sec-fetch-site': 'same-origin', 'sec-fetch-mode': 'navigate', 'sec-fetch-user': '?1',
                },
            )
            print(result)
        except Exception as e:
            print(e)
        await wb.stop()
    asyncio.run(_main())