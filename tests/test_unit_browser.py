from bs4 import BeautifulSoup
from germ.browser import get_indentation, html_to_md, md_convert_list_tags, md_convert_pre_tags


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


def test_md_convert_pre_tags():
    soup = BeautifulSoup(
        """
        <html><body><main>
            <pre><code>// Code in blockquote
console.log('> not a quote');
</code></pre>
            <ul><li><pre><code>// Code in blockquote
console.log('> not a quote');
</code></pre></li></ul>
            <ul><li>Some text before code<pre><code>// Code in blockquote
console.log('> not a quote');
</code></pre></li></ul>
            <ol><li><pre><code>// Code in blockquote
console.log('> not a quote');
</code></pre></li></ol>
            <ol><li>Some text before code<pre><code>// Code in blockquote
console.log('> not a quote');
</code></pre></li></ol>
            <ol><li>Some text with a newline before code
                <pre><code>// Code in blockquote
console.log('> not a quote');
</code></pre></li></ol>
            <blockquote><ol><li><pre><code>// Code in blockquote
console.log('> not a quote');
</code></pre></li></ol></blockquote>
        </main></body></html>
        """, 'lxml')
    main_tag = soup.select_one("main")
    md_convert_list_tags(main_tag)
    md_convert_pre_tags(main_tag)
    assert main_tag.get_text() == """
```
// Code in blockquote
console.log('> not a quote');
```
- ```
  // Code in blockquote
  console.log('> not a quote');
  ```
- Some text before code
  ```
  // Code in blockquote
  console.log('> not a quote');
  ```
1. ```
   // Code in blockquote
   console.log('> not a quote');
   ```
1. Some text before code
   ```
   // Code in blockquote
   console.log('> not a quote');
   ```
1. Some text with a newline before code 
   ```
   // Code in blockquote
   console.log('> not a quote');
   ```
> 1. ```
>    // Code in blockquote
>    console.log('> not a quote');
>    ```
"""


if __name__ == '__main__':
    with open("tests/data/html_docs/markdown_torture_test.html") as fd:
        html = fd.read()

        md = html_to_md(html)
        print(md)