import mistune


class HtmlSentenceExtractor(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__()
        self.elements = []

    def strong(self, text):
        self.elements.append(('strong', text))
        return super().strong(text)

    def em(self, text):
        self.elements.append(('em', text))
        return super().em(text)

    def link(self, link, title, text):
        self.elements.append(('link', link, title, text))
        return super().link(link, title, text)

    def image(self, src, title, alt):
        self.elements.append(('image', src, title, alt))
        return super().image(src, title, alt)

    def code(self, text, lang=None):
        self.elements.append(('inline_code', lang, text))
        return super().code(text, lang)
