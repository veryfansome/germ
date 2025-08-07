import dns.exception
import dns.resolver
import logging
import mimetypes
import os
import re
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pydantic import BaseModel

from germ.data.iana import IanaTLDCacher
from germ.utils.patterns import ipv4_addr_pattern, ipv6_addr_pattern, naive_sentence_end_pattern

logger = logging.getLogger(__name__)

iana_data = IanaTLDCacher()

# # Markdown parsing
end_of_line_pattern = re.compile(r'(\r\n|\r|\n)$')
leading_whitespace_pattern = re.compile(r'^[ \t]*')
md_parser = MarkdownIt()


class ParsedDoc(BaseModel):
    code_blocks: list[str]
    text: str

    @classmethod
    def from_text(cls, doc_text: str) -> ("ParsedDoc", str):
        tokens = md_parser.parse(doc_text)
        ranges: list[tuple[int, int]] = []
        blocks: list[str] = []
        for tok in tokens:
            if tok.type in {"fence", "code_block"}:
                start, end = tok.map  # end is *exclusive*
                ranges.append((start, end))
                blocks.append(tok.content)  # body only; fences stripped

        if not ranges:  # fast-path: no blocks at all
            return cls(code_blocks=blocks, text=doc_text)

        # Build sanitized text by stitching original segments + placeholders.
        lines = doc_text.splitlines(keepends=True)
        sanitized_parts: list[str] = []
        last = 0
        for idx, (start, end) in enumerate(ranges, 1):
            # Unchanged part before this block
            sanitized_parts.append("".join(lines[last:start]))

            # Derive indentation + EOL from original fences
            open_line = lines[start]  # first line with the fence
            indent = leading_whitespace_pattern.match(open_line).group(0)
            eol_match = end_of_line_pattern.search(lines[end - 1])  # closing fence line
            eol_chars = eol_match.group(0) if eol_match else ""

            sanitized_parts.append(f"{indent}[CODE_BLOCK:{idx}]{eol_chars}")
            last = end  # skip over the whole block
        sanitized_parts.append("".join(lines[last:]))  # remainder of the doc
        return cls(code_blocks=blocks, text="".join(sanitized_parts))


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


if __name__ == "__main__":
    from germ.observability.logging import setup_logging
    setup_logging()
