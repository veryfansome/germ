from bs4 import BeautifulSoup
from typing import Any
import logging
import mimetypes
import mistune
import os
import re
import socket

logger = logging.getLogger(__name__)


class DomainNameValidator:
    def __init__(self):
        self.known_tlds = {  # TODO: Persist in DB and add to this list automatically over time
            # Infra
            "arpa",

            # Country TLDs
            "ac", "ad", "ae", "af", "ag", "ai", "al", "am", "ao", "aq", "ar", "as", "at", "au", "aw", "ax", "az",
            "ba", "bb", "bd", "be", "bf", "bg", "bh", "bi", "bj", "bm", "bn", "bo", "bq", "br", "bs", "bt", "bw", "by",
            "bz",
            "ca", "cc", "cd", "cf", "cg", "ch", "ci", "ck", "cl", "cm", "cn", "co", "cr", "cu", "cv", "cw", "cx", "cy",
            "cz",
            "de", "dj", "dk", "dm", "do", "dz",
            "ec", "ee", "eg", "eh", "er", "es", "et", "eu",
            "fi", "fj", "fk", "fm", "fo", "fr",
            "ga", "gd", "ge", "gf", "gg", "gh", "gi", "gl", "gm", "gn", "gp", "gq", "gs", "gt", "gu", "gw", "gy",
            "hk", "hm", "hn", "hr", "ht", "hu",
            "id", "ie", "il", "im", "in", "io", "iq", "ir", "is", "it",
            "je", "jm", "jo", "jp",
            "ke", "kg", "kh", "ki", "km", "kn", "kp", "kr", "kw", "ky", "kz",
            "la", "lb", "lc", "li", "lk", "lr", "ls", "lt", "lu", "lv", "ly",
            "ma", "mc", "md", "me", "mg", "mh", "mk", "ml", "mm", "mn", "mo", "mp", "mq", "mr", "ms", "mt", "mu", "mv",
            "mw", "mx", "my", "mz",
            "na", "nc", "ne", "nf", "ng", "ni", "nl", "no", "np", "nr", "nu", "nz",
            "om", "pa", "pe", "pf", "pg", "ph", "pk", "pl", "pm", "pn", "pr", "ps", "pt", "pw", "py",
            "qa",
            "re", "ro", "rs", "ru", "rw",
            "sa", "sb", "sc", "sd", "se", "sg", "sh", "si", "sk", "sl", "sm", "sn", "so", "sr", "ss", "st", "su", "sv",
            "sx", "sy", "sz",
            "tc", "td", "tf", "tg", "th", "tj", "tk", "tl", "tm", "tn", "to", "tr", "tt", "tv", "tw", "tz",
            "ua", "ug", "uk", "us", "uy", "uz",
            "va", "vc", "ve", "vg", "vi", "vn", "vu",
            "wf", "ws",
            "ye", "yt",
            "za", "zm", "zw",

            # Generic TLDs
            "academy", "accountant", "accountants", "active", "actor", "ads", "adult", "aero", "africa", "agency",
            "airforce", "amazon", "amex", "analytics", "apartments", "app", "apple", "archi", "army", "art", "arte",
            "associates", "attorney", "auction", "audible", "audio", "author", "auto", "autos", "aws",

            "baby",
            "band",
            "bank",
            "bar",
            "barefoot",
            "bargains",
            "baseball",
            "basketball",
            "beauty",
            "biz",
            "blog",
            "co",
            "com",
            "design",
            "dev",
            "info",
            "me",
            "net",
            "online",
            "org",
            "site",
            "store",
            "tech",
            "xyz",

            "coop",
            "edu",
            "gov",
            "int",
            "mil",
            "museum",
        }


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
        if "scheme" not in artifacts or artifacts["scheme"] != "file":  # Should be "file"
            return {**artifacts, **{"skipped": True, "reason": "unexpected_pattern"}}
        href = re.sub(r"^/{3,}", "/", href)  # Tolerant of minor typos
    elif href.startswith("//"):
        if "scheme" not in artifacts:
            artifacts["scheme"] = "relative"
        elif artifacts["scheme"] not in ["http", "https"]:
            return {**artifacts, **{"skipped": True, "reason": "unexpected_pattern"}}
        href = href[2:]
    # After this point, everything "should" be domain, port, path and/or params

    if "scheme" in artifacts and artifacts["scheme"] == "file":
        filename_match = re.search(r"^(?P<path>(/C:)?[a-zA-Z0-9/._~%-]+)", href)
        if filename_match:
            artifacts["path"] = filename_match.group("path")
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

        domain_match = re.search(r"^(?P<domain>[a-zA-Z0-9](?:(?![-.]{2})[a-zA-Z0-9.-]){0,61}[a-zA-Z0-9])", href)
        # ^: Asserts the start of the string.
        # (?P<domain>...): Named capturing group for the domain.
        # [a-zA-Z0-9]: The first character must be alphanumeric.
        # (?:...): A non-capturing group for the rest of the domain.
        # (?![-.]{2}): A negative lookahead that asserts what follows is not two consecutive hyphens or dots.
        # [a-zA-Z0-9.-]: Allows alphanumeric characters, hyphens, and dots.
        # {0,61}: Allows up to 61 additional characters (to ensure total length does not exceed 63 characters).
        # [a-zA-Z0-9]: Ensures the last character of the domain is alphanumeric.
        if domain_match:
            matched_blob = domain_match.group("domain")
            if matched_blob in ["127.0.0.1", "localhost"]:
                artifacts["domain"] = matched_blob
                href = href[len(matched_blob):]
            elif "." not in matched_blob:
                # matched_blob could actually be part of a path
                artifacts["domain"] = "relative"  # Probably
            else:
                # Things that look like domains can actually be files
                guessed_mimetype, _ = mimetypes.guess_type(matched_blob)
                if guessed_mimetype:
                    artifacts["domain"] = "relative"
                    artifacts["path"] = matched_blob
                else:
                    artifacts["domain"] = matched_blob
                href = href[len(matched_blob):]

    #print(href)
    return artifacts


def extract_markdown_page_elements(text: str):
    extractor = MarkdownPageElementExtractor()
    mistune.create_markdown(renderer=extractor)(text)
    return extractor.elements


def get_html_soup(text) -> BeautifulSoup:
    return BeautifulSoup(text, 'html.parser')


def is_resolvable_domain(domain: str):
    try:
        resolved_address = socket.gethostbyname(domain)
        return True, resolved_address
    except socket.gaierror:
        return False, None


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

    test("/")
    test("/ui.css")
    test("https://www.google.com")
    test("http://localhost:8080")
    test("localhost:8080")
    test("localhost:8080/index.html")
    test("https://example.com:8080?foo=foo#bar")
    test("https://example.me:8080/?foo=foo#bar")
    test("https://example.me:8080/index.php?foo=foo")
    test("https://example.me:8080/us.js?foo=foo")
    test("https://example.photography:8080/path/to/some/object?foo=foo")
    test("http://127.0.0.1:8080")
    test("page.html")
    test("page.html#section1")
    test("folder/page.html")
    test("../page.html")
    test("//www.google.com")
    test("www.google.com")
    test("mailto:someone@example.com?subject=Hello&body=Message")
    test("tel:+1234567890")
    test("file:///C:/path/to/file.txt")
    test("file:///src/README.md")
    test("javascript:alert('Hello World!')")
    test("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==")
