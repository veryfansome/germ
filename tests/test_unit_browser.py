from bs4 import BeautifulSoup
from germ.browser import get_indentation, html_to_markdown

def test_get_indentation():
    soup = BeautifulSoup(
        """
        <html><body><main>
            <p>Something something.</p>
            <ul>
                <li>Item 1
                    <ol>
                        <li>Item a</li>
                        <li>Item b</li>
                        <li>Item c</li>
                    </ol>
                </li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
        </main></body></html>
        """, 'lxml')
    p_tag = soup.select_one("p")
    assert get_indentation(p_tag) == ""
    ul_tag = soup.select_one("ul")
    assert get_indentation(ul_tag) == ""
    ol_tag = soup.select_one("ol")
    assert get_indentation(ol_tag) == "  "


def test_html_to_markdown():
    with open("tests/data/html_docs/markdown_torture_test.html") as fd:
        html = fd.read()

        md = html_to_markdown(html)
        print(md)


if __name__ == '__main__':
    test_html_to_markdown()