from bs4 import BeautifulSoup
from typing import Any
import logging
import mistune
import re

logger = logging.getLogger(__name__)

href_domain_pattern = r"^(?P<domain>[a-zA-Z0-9](?:(?![-.]{2})[a-zA-Z0-9.-]){0,61}[a-zA-Z0-9])"
# ^: Asserts the start of the string.
# (?P<domain>...): Named capturing group for the domain.
# [a-zA-Z0-9]: The first character must be alphanumeric.
# (?:...): A non-capturing group for the rest of the domain.
# (?![-.]{2}): A negative lookahead that asserts what follows is not two consecutive hyphens or dots.
# [a-zA-Z0-9.-]: Allows alphanumeric characters, hyphens, and dots.
# {0,61}: Allows up to 61 additional characters (to ensure the total length of the label does not exceed 63 characters).
# [a-zA-Z0-9]: Ensures the last character of the domain is alphanumeric.

href_scheme_pattern = r"^(?P<scheme>[a-z0-9+-]+):(?![0-9]+)"
# ^: Asserts the start of the string.
# (?P<scheme>...): Named capturing group for the scheme.
# [a-zA-Z0-9.-]+:: Allows alphanumeric characters, plus, and minus up to a colon.
# (?![0-9]+): A negative lookahead that asserts what follows is not consecutive numerals, which might be `name:port`


class MarkdownPageElementExtractor(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__()
        self.elements = []

    def block_code(self, code, info=None):
        self.elements.append(('block_code', info, code))
        return super().block_code(code, info)

    def block_error(self, text):
        self.elements.append(('block_error', text))
        return super().block_error(text)

    def block_html(self, html):
        self.elements.append(('block_html', html))
        return super().block_html(html)

    def block_quote(self, text):
        self.elements.append(('block_quote', text))
        return super().block_quote(text)

    def block_text(self, text):
        self.elements.append(('block_text', text))
        return super().block_text(text)

    def heading(self, text, level, **attrs):
        self.elements.append(('heading', level, text))
        return super().heading(text, level, **attrs)

    def list(self, text: str, ordered: bool, **attrs: Any) -> str:
        self.elements.append(('list', ordered))
        return super().list(text, ordered, **attrs)

    def list_item(self, text):
        self.elements.append(('list_item', text))
        return super().list_item(text)

    def paragraph(self, text):
        self.elements.append(('paragraph', text))
        return super().paragraph(text)


def extract_href_features(href: str):
    """
    Extract useful info for supported schemes from variable `href` values.

    :param href:
    :return:
    """
    scheme_match = re.search(href_scheme_pattern, href)
    artifacts = {}
    if scheme_match:
        artifacts["scheme"] = scheme_match.group("scheme")
        if artifacts["scheme"] in ["file", "http"]:
            href = href[5:]
        elif artifacts["scheme"] == "https":
            href = href[6:]
        else:
            # Excludes: data, javascript, mailto, tel, and custom schemes
            return {"skipped": True, "reason": f"`{artifacts["scheme"]}` is not a supported href scheme"}
    if href.startswith("./") or href.startswith("../"):
        return {"skipped": True, "reason": "relative URL"}
    elif href.startswith("//"):
        if not href.startswith("///") and "scheme" not in artifacts:
            artifacts["scheme"] = "https"  # If two, not three, possibly relative, assume https
        href = href[2:]

    # No strip every thing in the back
    #print(href, artifacts)
    return artifacts


def extract_markdown_page_elements(text: str):
    extractor = MarkdownPageElementExtractor()
    mistune.create_markdown(renderer=extractor)(text)
    return extractor.elements


def get_html_soup(text) -> BeautifulSoup:
    return BeautifulSoup(text, 'html.parser')


async def strip_html_elements(soup: BeautifulSoup, tag: str = None):
    text_elements = []
    html_elements = {}
    for idx, element in enumerate(soup.find_all() if tag is None else soup.find(tag)):
        if element.name is None:  # This is just text
            text_elements.append(element.string)
        else:
            text_elements.append("[oops]")  # Placeholder
            html_elements[idx] = element
    # Iterate through elements to write over the placeholders
    element_artifacts = list(html_elements.values())
    for idx, element in html_elements.items():
        if element.find():  # Has inner elements
            logger.info(f"found <{element.name}>, with inner elements: {element}")
            inner_string, inner_artifacts = await strip_html_elements(element)
            if inner_artifacts:
                element_artifacts += inner_artifacts

            if element.name == "a":
                href_features = extract_href_features(element['href'])
                logger.info(f"href_features: {href_features}")
                # TODO: - Replace inner_string
                #       - Use some alphabetical hash based on the domain, with the first letter upper cased so that the
                #         word would be seen as a proper noun by the POS tagger.
                #       - Use regex to capture [\s]*[\w]+\.[\w]+[^\s]*
                #       - Capture protocol, path, and params if any
                #       - Do a domain resolution against captured domain string to see
                #       - But it could also be a local link so we'll have to be careful.
                pass

            text_elements[idx] = inner_string
        else:  # Doesn't have inner elements
            logger.info(f"stripped <{element.name}>, kept inner string: {element.string}")
            if element.name == "a":
                href_features = extract_href_features(element['href'])
                logger.info(f"href_features: {href_features}")
                # TODO: Same as above
                pass
            text_elements[idx] = element.string if element.string else ""
    return ''.join(text_elements), element_artifacts


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    def test(url: str):
        print(url, extract_href_features(url))
        #extract_href_features(url)

    test("https://www.google.com")
    test("http://localhost:8080")
    test("localhost:8080")
    test("localhost:8080/index.html")
    test("http://localhost:8080?foo=foo#bar")
    test("http://localhost:8080/?foo=foo#bar")
    test("http://localhost:8080/path/to/some/object?foo=foo")
    test("http://127.0.0.1:8080")
    test("page.html")
    test("page.html#section1")
    test("folder/page.html")
    test("../page.html")
    test("//www.google.com")
    test("mailto:someone@example.com?subject=Hello&body=Message")
    test("tel:+1234567890")
    test("file:///C:/path/to/file.txt")
    test("file:///src/README.md")
    test("javascript:alert('Hello World!')")
    test("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==")
