import mistune


class MarkdownPageExtractor(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__()
        self.elements = []

    def heading(self, text, level, **attrs):
        self.elements.append(('heading', level, text))
        return super().heading(text, level, **attrs)

    def paragraph(self, text):
        self.elements.append(('paragraph', text))
        return super().paragraph(text)

    def list_item(self, text):
        self.elements.append(('list_item', text))
        return super().list_item(text)

    def block_code(self, code, info=None):
        self.elements.append(('code_block', info, code))
        return super().block_code(code, info)

    def block_quote(self, text):
        self.elements.append(('block_quote', text))
        return super().block_quote(text)
