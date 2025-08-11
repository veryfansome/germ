from germ.utils.parsers import ParsedDoc


def test_markdown_document_parsing():
    with open("tests/data/markdown_docs/cloud_storage_object_access.md") as fd:
        markdown_text = fd.read()

        doc = ParsedDoc.from_text(markdown_text)
        code_block_cnt = len(doc.code_blocks)
        assert code_block_cnt == 5, code_block_cnt

        assert doc.restore() == markdown_text
