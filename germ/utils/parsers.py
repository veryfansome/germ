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
    code_blocks: list[tuple[str | None, str]]
    text_parts: list[str]

    @classmethod
    def from_text(cls, doc_text: str) -> ("ParsedDoc", str):
        tokens = md_parser.parse(doc_text)
        ranges: list[tuple[int, int]] = []
        blocks: list[tuple[str | None, str]] = []
        for tok in tokens:
            if tok.type in {"fence", "code_block"}:
                start, end = tok.map  # end is *exclusive*
                ranges.append((start, end))
                tok_info = (tok.info or "").strip()
                lang = tok_info.split()[0].strip() if tok_info else None
                blocks.append((lang, tok.content))  # body only; fences stripped

        if not ranges:  # fast-path: no blocks at all
            return cls(code_blocks=blocks, text_parts=[doc_text])

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
        return cls(code_blocks=blocks, text_parts=sanitized_parts)

    def restore(self) -> str:
        text_parts = []
        code_block_pointer = 0
        for part in self.text_parts:
            if not "[CODE_BLOCK:" in part:
                text_parts.append(part)
            else:
                indent = leading_whitespace_pattern.match(part).group(0)
                text_parts.append(f"{indent}```{self.code_blocks[code_block_pointer][0] or ''}\n")
                text_parts.append("\n".join([
                    (f"{indent}{l}" if l else l) for l in self.code_blocks[code_block_pointer][1].split("\n")
                ]))
                text_parts.append(f"{indent}```\n")
                code_block_pointer += 1
        return "".join(text_parts)


def get_html_soup(text) -> BeautifulSoup:
    return BeautifulSoup(text, 'html.parser')
