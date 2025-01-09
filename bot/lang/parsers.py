from bs4 import BeautifulSoup
from typing import Any
import mistune


class LargePageElementExtractor(mistune.HTMLRenderer):
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


def extract_large_html_elements(text: str):
    extractor = LargePageElementExtractor()
    mistune.create_markdown(renderer=extractor)(text)
    return extractor.elements


def get_html_parser(text):
    return BeautifulSoup(text, 'html.parser')
