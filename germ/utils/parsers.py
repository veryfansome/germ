import dns.exception
import dns.resolver
import logging
import mimetypes
import mistune
import os
import re
from bs4 import BeautifulSoup
from bs4.element import PageElement as BeautifulSoupPageElement
from copy import copy
from enum import Enum
from pydantic import BaseModel
from typing import Any

from germ.data.iana import IanaTLDCacher
from germ.utils.patterns import ipv4_addr_pattern, ipv6_addr_pattern, naive_sentence_end_pattern

logger = logging.getLogger(__name__)

iana_data = IanaTLDCacher()


class TextElement(BaseModel):
    text: list[int | str]
    elements: list[Any] | None


class PageElement(BaseModel):
    type: int
    headings: dict[int, TextElement] | None


class PageElementType(Enum):
    CODE_BLOCK = 1
    LIST = 2
    PARAGRAPH = 3


class CodeElement(PageElement):
    type: int = PageElementType.CODE_BLOCK
    text: list[int | str]
    language: str | None


class ListElement(PageElement):
    type: int = PageElementType.LIST
    ordered: bool
    items: list[TextElement]


class ParagraphElement(PageElement, TextElement):
    type: int = PageElementType.PARAGRAPH


class ParsedMarkdownPage(BaseModel):
    scaffold: list[CodeElement | ListElement | ParagraphElement] = []
    code: list[str] = []
    text: list[str] = []


class MarkdownPageElementExtractor(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__()
        self._headings_context: dict[int, TextElement] = {}
        self._list_items: list[TextElement] = []
        self.elements: list[CodeElement | ListElement | ParagraphElement] = []

    def block_code(self, code, info=None):
        self.elements.append(CodeElement(
            language=info,
            text=[code],
            headings=copy(self._headings_context),
        ))
        return super().block_code(code, info)

    def heading(self, text, level, **attrs):
        p_soup = get_html_soup(f"<p>{text}</p>")
        p_text, p_elements = strip_html_elements(p_soup, "p")
        self._headings_context[level] = TextElement(
            text=split_sentences(p_text),
            elements=p_elements,
        )
        return super().heading(text, level, **attrs)

    def list(self, text: str, ordered: bool, **attrs: Any) -> str:
        self.elements.append(ListElement(
            ordered=ordered,
            items=copy(self._list_items),
            headings=copy(self._headings_context),
        ))
        self._list_items = []
        return super().list(text, ordered, **attrs)

    def list_item(self, text):
        p_soup = get_html_soup(f"<p>{text}</p>")
        p_text, p_elements = strip_html_elements(p_soup, "p")
        self._list_items.append(TextElement(
            text=split_sentences(p_text),
            elements=p_elements,
        ))
        return super().list_item(text)

    def paragraph(self, text):
        p_soup = get_html_soup(f"<p>{text}</p>")
        p_text, p_elements = strip_html_elements(p_soup, "p")
        self.elements.append(ParagraphElement(
            text=split_sentences(p_text),
            elements=p_elements,
            headings=copy(self._headings_context),
        ))
        return super().paragraph(text)


def extract_href_features(href: str):
    """
    Extract useful info for supported schemes from variable `href` values.

    :param href:
    :return:
    """
    scheme_match = re.search(r"^(?P<scheme>[a-z0-9+-]+):(?![0-9]+)", href)
    # ^: Asserts the start of the string.
    # (?P<scheme>...): Named capturing group for the scheme.
    # [a-zA-Z0-9.-]+: Allows alphanumeric characters, plus, and minus up to a colon.
    # (?![0-9]+): A negative lookahead that asserts what follows is not consecutive numerals, which might be `name:port`
    artifacts = {}
    if scheme_match:
        artifacts["scheme"] = scheme_match.group("scheme")
        if artifacts["scheme"] in ["file", "http"]:
            href = href[5:]
        elif artifacts["scheme"] == "https":
            href = href[6:]
        else:
            # Excludes: data, javascript, mailto, tel, and custom schemes
            return {"skipped": True, "reason": "unsupported_scheme"}

    # Now stripped till after the :
    if href.startswith("///"):
        if "scheme" not in artifacts:
            artifacts["scheme"] = "file"
        elif artifacts["scheme"] != "file":  # Should be "file"
            return {**artifacts, **{"skipped": True, "reason": "unexpected_pattern"}}
        href = re.sub(r"^/{3,}", "/", href)  # Tolerant of minor typos
    elif href.startswith("//"):
        if "scheme" not in artifacts:
            artifacts["scheme"] = "relative"
        elif artifacts["scheme"] not in ["http", "https"]:
            return {**artifacts, **{"skipped": True, "reason": "unexpected_pattern"}}
        href = href[2:]
    # After this point, everything "should" be fqdn, port, path and/or params

    if "scheme" in artifacts and artifacts["scheme"] == "file":
        file_path_match = re.search(r"^(?P<path>(/C:)?[a-zA-Z0-9/._~%-]+)", href)
        if file_path_match:
            artifacts["path"] = file_path_match.group("path")
            artifacts["path_exists"] = os.path.exists(artifacts["path"])
            if artifacts["path_exists"]:
                artifacts["path_is_dir"] = os.path.isdir(artifacts["path"])
                if not artifacts["path_is_dir"]:
                    artifacts["path_is_file"] = os.path.isfile(artifacts["path"])
                    if not artifacts["path_is_file"]:
                        artifacts["path_is_link"] = os.path.islink(artifacts["path"])
            href = href[len(artifacts["path"]):]
        else:
            return {**artifacts, **{"skipped": True, "reason": "unexpected_pattern"}}
    else:
        # Unknown scheme or http(s) and relative http scheme
        if "scheme" not in artifacts:
            artifacts["scheme"] = "relative"

        fqdn_match = re.search(r"^(?P<fqdn>[a-zA-Z0-9](?:(?![-.]{2})[a-zA-Z0-9.-]){0,61}[a-zA-Z0-9])", href)
        # ^: Asserts the start of the string.
        # (?P<fqdn>...): Named capturing group for the fqdn.
        # [a-zA-Z0-9]: The first character must be alphanumeric.
        # (?:...): A non-capturing group for the rest of the fqdn.
        # (?![-.]{2}): A negative lookahead that asserts what follows is not two consecutive hyphens or dots.
        # [a-zA-Z0-9.-]: Allows alphanumeric characters, hyphens, and dots.
        # {0,61}: Allows up to 61 additional characters (to ensure total length does not exceed 63 characters).
        # [a-zA-Z0-9]: Ensures the last character of the fqdn is alphanumeric.
        if fqdn_match:
            matched_blob = fqdn_match.group("fqdn")
            if matched_blob in ["localhost"]:
                artifacts["fqdn"] = matched_blob
                href = href[len(matched_blob):]
            elif "." not in matched_blob:
                # matched_blob could actually be part of a path
                artifacts["fqdn"] = "relative"  # Probably
            else:
                if ipv4_addr_pattern.search(matched_blob):
                    artifacts["ipv4_address"] = matched_blob
                elif iana_data.is_possible_public_fqdn(matched_blob):  # Things that look like fqdns can be files
                    #    ^ Empty if load_tld_cache() not called
                    artifacts["fqdn"] = matched_blob
                else:
                    guessed_mimetype, _ = mimetypes.guess_type(matched_blob)
                    if guessed_mimetype:
                        artifacts["fqdn"] = "relative"
                        artifacts["path"] = matched_blob
                    else:
                        return {**artifacts, **{"skipped": True, "reason": "unexpected_pattern"}}
                href = href[len(matched_blob):]
        else:
            ipv6_addr_match = ipv6_addr_pattern.search(href)
            if ipv6_addr_match:
                ipv6_address = ipv6_addr_match.group("ipv6_addr")
                href = href[len(ipv6_address):]
                artifacts["ipv6_address"] = ipv6_address.strip("[]")
            else:
                artifacts["fqdn"] = "relative"

        if not href:
            return artifacts

        port_match = re.search(r"^(?P<port>:\d{,5}(?=[/?#]?))", href)
        # (?P<port>...): Named capturing group for the port.
        # :\d{,5}: Colon and up to five digits
        # (?=[/?#]?): A positive lookahead for possible end of port characters
        if port_match:
            port_match = port_match.group("port")
            href = href[len(port_match):]
            artifacts["port"] = port_match[1:]  # Strip the leading colon

        if not href:
            return artifacts

        url_path_match = re.search(r"^(?P<path>[a-zA-Z0-9/._~%-]+)(?=[?#]?)", href)
        # (?P<path>...): Named capturing group for the path.
        # [a-zA-Z0-9/._~%-]+): Capture consecutive valid path characters
        # (?=[?#]?): A positive lookahead for possible end of path characters
        if url_path_match:
            artifacts["path"] = url_path_match.group("path")
            href = href[len(artifacts["path"]):]

    # If anything's left, it'll be the query blob
    if href:
        artifacts["query"] = href
    return artifacts


def extract_markdown_page_elements(text: str) -> ParsedMarkdownPage:
    extractor = MarkdownPageElementExtractor()
    mistune.create_markdown(renderer=extractor)(text)

    parsed_page = ParsedMarkdownPage()
    for element_idx, element in enumerate(extractor.elements):
        element_copy = element.model_copy(deep=True)
        for level, heading in element.headings.items():
            for blob_idx, blob in enumerate(heading.text):
                if blob not in parsed_page.text:
                    parsed_page.text.append(blob)
                element_copy.headings[level].text[blob_idx] = parsed_page.text.index(blob)
        if element.type == PageElementType.CODE_BLOCK:
            for chunk_idx, chunk in enumerate(element.text):
                if chunk not in parsed_page.code:
                    parsed_page.code.append(chunk)
                element_copy.text[chunk_idx] = parsed_page.code.index(chunk)
        elif element.type == PageElementType.LIST:
            for item_idx, item in enumerate(element.items):
                for sentence_idx, sentence in enumerate(item.text):
                    if sentence not in parsed_page.text:
                        parsed_page.text.append(sentence)
                    element_copy.items[item_idx].text[sentence_idx] = parsed_page.text.index(sentence)
        elif element.type == PageElementType.PARAGRAPH:
            for sentence_idx, sentence in enumerate(element.text):
                if sentence not in parsed_page.text:
                    parsed_page.text.append(sentence)
                element_copy.text[sentence_idx] = parsed_page.text.index(sentence)
        parsed_page.scaffold.append(element_copy)
    return parsed_page


def fqdn_to_proper_noun(fqdn: str):
    """
    Converts FQDNs to something that looks like a single proper noun. For example, converts www.google.com to
    "GoogleDOTcom"

    :param fqdn:
    :return:
    """
    return "".join([char.upper() if idx == 0 else char for idx, char in enumerate("DOT".join(fqdn.lower().split(".")[-2:]))])


def get_html_soup(text) -> BeautifulSoup:
    return BeautifulSoup(text, 'html.parser')


def resolve_fqdn(fqdn: str, nameservers: list[str] = None, timeout: int = 2):
    """
    Resolve fqdn and return Answer or failure reason.

    :param fqdn:
    :param nameservers:
    :param timeout:
    :return: answer:Answer, error: bool, timed_out: bool
    """
    resolver = dns.resolver.Resolver()
    resolver.nameservers = nameservers if nameservers else ["1.1.1.1"]
    resolver.timeout = timeout
    resolver.lifetime = timeout  # Set the lifetime for the query

    try:
        return resolver.resolve(fqdn), False, False
    except dns.exception.Timeout:
        return None, False, True
    except dns.exception.DNSException as e:
        return None, True, False


def split_sentences(p_text: str) -> list[str]:
    sentences: list[str] = []
    while p_text:
       # Not always perfect but good enough
       sentence_end_match = naive_sentence_end_pattern.search(p_text, 0, len(p_text))
       if sentence_end_match:
           sentences.append(p_text[:sentence_end_match.end()])
           p_text = p_text[sentence_end_match.end():].strip()
       else:
           sentences.append(p_text.strip())
           p_text = ""
    return sentences


def strip_html_elements(
        soup: BeautifulSoup | BeautifulSoupPageElement,
        tag: str = None
) -> (str, BeautifulSoupPageElement):
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
            inner_string, inner_artifacts = strip_html_elements(element)
            if inner_artifacts:
                element_artifacts += inner_artifacts
        else:  # Doesn't have inner elements
            logger.info(f"stripped <{element.name}>, kept inner string: {element.string}")
            if element.name == "code":  # Translate back to backticks, which the token classifier knows
                element.string = f"`{element.string}`"
        text_elements[idx] = element.string if element.string else ""
    return ''.join(text_elements), element_artifacts


if __name__ == "__main__":
    from germ.observability.logging import setup_logging
    setup_logging()
