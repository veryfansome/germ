import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)

default_params = {
    "action": "query",
    "format": "json",
    "prop": "extracts|info|categories|links",
    'exintro': 'true',  # Get only the introduction part of the article
    'explaintext': 'true',  # Return plain text instead of HTML
    'inprop': 'url',  # Include the page URL in the info
    'pllimit': '500',  # Limit to 500 links (maximum)
}
wikipedia_url = "https://en.wikipedia.org/w/api.php"


async def get_wikipedia_article(title):

    params = {
        'titles': title,
        **default_params,
    }

    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(wikipedia_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data['query']['pages']
                    #logger.info(pages)

                    # Get the page ID (could be negative if the page doesn't exist)
                    page_id = next(iter(pages))
                    page_data = pages[page_id]

                    # Extract the page content (the introduction)
                    page_content = page_data.get('extract', 'No extract available')
                    logger.info(f"Page content: {page_content}")

                    page_links = page_data.get('links', [])
                    logger.info(f"Page links: {page_links}")

                    if 'continue' in data:
                        params['plcontinue'] = data['continue']['plcontinue']
                    else:
                        break
                else:
                    logger.error(f"Error: Unable to retrieve data, status={response.status} text={response.text}")
                    break


async def main():
    article_title = "List_of_lists_of_lists"
    await get_wikipedia_article(article_title)


if __name__ == '__main__':
    from observability.logging import setup_logging
    setup_logging()
    asyncio.run(main())
