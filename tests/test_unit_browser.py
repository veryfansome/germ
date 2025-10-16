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


def test_md_convert_list_tags():
    soup = BeautifulSoup("""
        <html><body><main>
        <ul>
          <li>Unordered level 1
            <ul>
              <li>Unordered level 2 with a paragraph.
                <p>This is a paragraph inside a list item.</p>
              </li>
              <li>Unordered level 2 with nested ordered list
                <ol start="3">
                  <li value="3">Item 3</li>
                  <li>Item 4</li>
                </ol>
              </li>
            </ul>
          </li>
          <li>Another bullet</li>
        </ul>

        <ol type="A" start="2" reversed>
          <li>Alpha 2</li>
          <li>Alpha 1</li>
        </ol>
        </main></body></html>
    """, "lxml")
    main_tag = soup.select_one("main")
    md_convert_list_tags(main_tag)
    assert main_tag.get_text() == """
- Unordered level 1 
  - Unordered level 2 with a paragraph. This is a paragraph inside a list item.
  - Unordered level 2 with nested ordered list 
    3. Item 3
    4. Item 4
- Another bullet
2. Alpha 2
3. Alpha 1
"""


def test_md_convert_pre_tags():
    soup = BeautifulSoup(
        """
        <html><body><main>
            <pre>
ASCII Art &lt;pre&gt;:
  +----+
  |    |
  +----+
            </pre>
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
ASCII Art <pre>:
  +----+
  |    |
  +----+
```
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