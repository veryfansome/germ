from bs4 import BeautifulSoup
from germ.browser import (
    get_indentation,
    html_to_md,
)


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
    text = html_to_md("""
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
    """)
    assert text == """
- Unordered level 1 
  - Unordered level 2 with a paragraph. This is a paragraph inside a list item.
  - Unordered level 2 with nested ordered list 
    3. Item 3
    4. Item 4
- Another bullet
2. Alpha 2
3. Alpha 1
""".strip()


def test_md_convert_pre_tags():
    text = html_to_md(
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
        """)
    assert text == """
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
""".strip()


def test_md_convert_simple_tag():
    text = html_to_md("""
            <html><body><main>
            <h2 id="inline-text">Inline semantics, whitespace, and entities</h2>
            <p>
              <em>em</em>, <strong>strong</strong>, <i>i</i>, <b>b</b>, <u>u</u>, <s>strikethrough</s>,
              <mark>mark</mark>, <small>small</small>, H<sub>2</sub>O, E = mc<sup>2</sup>,
              press <kbd>Ctrl</kbd>+<kbd>K</kbd>, <var>x</var>=<samp>42</samp>, inline <code>code()</code>,
              an <abbr title="Internationalization">i18n</abbr> example, a <cite>citation</cite>,
              a short quote <q cite="https://example.com/quote">“hello”</q>,
              and a <time datetime="2025-10-07T12:34:56-07:00">timestamp</time>.
            </p>
            <p>BiDi controls: <bdi>ABC123</bdi> <bdo dir="rtl">!dlroW olleH</bdo> <span dir="rtl">עברית العربية</span>.</p>
            <p>
              Escaped Markdown literals (should remain as text):
              <span data-literal>**bold?** _italic?_ `inline code` [link](https://example.com) ![img](x) > quote</span>
              and triple backticks: <span data-literal>``` not a fenced block ```</span>.
            </p>
            <p>Hard line breaks with <code>&lt;br&gt;</code> and <code>&lt;br /&gt;</code> below:<br>Line 2<br />Line 3</p>
            <aside><p>Soft hyphen &amp;shy;: hy&shy;phen&shy;ation; word break <wbr> opportunity.</p></aside>
            <hr />
            <h1 id="h1-duplicate">H1 (within article)</h1>
            <h2>H2</h2>
            <h3>H3</h3>
            <h4>H4</h4>
            <h5>H5</h5>
            <h6>H6</h6>
            <figure>
              <picture>
                <source srcset="images/sample.webp" type="image/webp" />
                <img src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" alt="1×1 transparent GIF" width="24" height="24" />
              </picture>
              <figcaption>Picture with fallback; transparent data URI image.</figcaption>
            </figure>

            <img src="images/no-alt.png" title="Image with title but no alt" width="16" height="16" />

            <img src="images/no-alt.png" width="16" height="16" /> <!-- Missing alt and title -->

            <figure>
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-label="A circle">
                <circle cx="50" cy="50" r="45" stroke="black" fill="none" />
              </svg>
              <figcaption>Inline SVG.</figcaption>
            </figure>

            <audio controls>
              <source src="media/sample.ogg" type="audio/ogg" />
              <source src="media/sample.mp3" type="audio/mpeg" />
              <p>Your browser does not support the audio element.</p>
            </audio>

            <video controls width="240">
              <source src="media/sample.mp4" type="video/mp4" />
              <track kind="subtitles" src="media/captions.vtt" srclang="en" label="English" default />
              Sorry, your browser doesn't support embedded videos.
            </video>
            </main></body></html>
        """)
    assert text == """
## Inline semantics, whitespace, and entities

*em*, **strong**, *i*, **b**, <u>u</u>, ~~strikethrough~~, <mark>mark</mark>, <small>small</small>, H<sub>2</sub>O, E = mc<sup>2</sup>, press Ctrl+K, x=42, inline `code()`, an i18n example, a citation, a short quote “hello”, and a timestamp. 
BiDi controls: ABC123 !dlroW olleH עברית العربية.
Escaped Markdown literals (should remain as text): \\*\\*bold\\?\\*\\* \\_italic\\?\\_ \\`inline code\\` \\[link\\]\\(https\\:\\/\\/example\\.com\\) \\!\\[img\\]\\(x\\) \\> quote and triple backticks: \\`\\`\\` not a fenced block \\`\\`\\`. 
Hard line breaks with `<br>` and `<br />` below:

Line 2

Line 3
<aside>Soft hyphen &shy;: hyphenation; word break  opportunity.</aside>
---

# H1 (within article)

## H2

### H3

#### H4

##### H5

###### H6

<figure><picture><source srcset="images/sample.webp"></source><img src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="></img></picture><figcaption>Picture with fallback; transparent data URI image.</figcaption></figure>
<img src="images/no-alt.png" title="Image with title but no alt"></img>
<img src="images/no-alt.png"></img> 
<figure><svg xmlns="http://www.w3.org/2000/svg" viewbox="0 0 100 100" role="img"><circle cx="50" cy="50" r="45" stroke="black" fill="none"></circle></svg><figcaption>Inline SVG.</figcaption></figure>
<audio><source src="media/sample.ogg"></source><source src="media/sample.mp3"></source></audio>
<video><source src="media/sample.mp4"></source><track src="media/captions.vtt"></track></video>
""".strip()


if __name__ == '__main__':
    with open("tests/data/html_docs/markdown_torture_test.html") as fd:
        html = fd.read()

        md = html_to_md(html)
        print(md)