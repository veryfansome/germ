import logging
import re
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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


def get_html_soup(text) -> BeautifulSoup:
    return BeautifulSoup(text, 'html.parser')
