"""Hidden acceptance tests for mdlite (spec: llm/pi/specs/mdlite.md)."""
import os
import sys

import pytest

sys.path.insert(0, os.environ.get("APP_DIR", "."))
from mdlite import to_html  # noqa: E402

CASES = [
    ("", ""),
    ("   \n\n  \n", ""),
    ("# Hello", "<h1>Hello</h1>"),
    ("###### Six", "<h6>Six</h6>"),
    ("####### Seven", "<p>####### Seven</p>"),
    ("#NoSpace", "<p>#NoSpace</p>"),
    ("##   padded   ", "<h2>padded</h2>"),
    ("# **Hi** there", "<h1><strong>Hi</strong> there</h1>"),
    ("Hello world", "<p>Hello world</p>"),
    ("line one\nline two", "<p>line one\nline two</p>"),
    ("a\n\nb", "<p>a</p>\n<p>b</p>"),
    ("a\n\n\n\nb", "<p>a</p>\n<p>b</p>"),
    ("a\r\nb\r\n\r\nc", "<p>a\nb</p>\n<p>c</p>"),
    ("a < b & c > d", "<p>a &lt; b &amp; c &gt; d</p>"),
    ("<script>x</script>", "<p>&lt;script&gt;x&lt;/script&gt;</p>"),
    ("**bold** and *it* and `code`", "<p><strong>bold</strong> and <em>it</em> and <code>code</code></p>"),
    ("`**x**`", "<p><code>**x**</code></p>"),
    ("`a<b&c`", "<p><code>a&lt;b&amp;c</code></p>"),
    ("**abc", "<p>**abc</p>"),
    ("*abc", "<p>*abc</p>"),
    ("snake_case_name", "<p>snake_case_name</p>"),
    ("_not italic_", "<p>_not italic_</p>"),
    ("\\*not italic\\*", "<p>*not italic*</p>"),
    ("a \\\\ b", "<p>a \\ b</p>"),
    ("[docs](https://x.com/a?b=1&c=2)", '<p><a href="https://x.com/a?b=1&amp;c=2">docs</a></p>'),
    ("[**b**](/p)", '<p><a href="/p"><strong>b</strong></a></p>'),
    ("[top](#top)", '<p><a href="#top">top</a></p>'),
    ("[x](javascript:alert(1))", "<p>[x](javascript:alert(1))</p>"),
    ("[x](ftp://host/file)", "<p>[x](ftp://host/file)</p>"),
    ("- a\n- b", "<ul><li>a</li><li>b</li></ul>"),
    ("* a\n* b", "<ul><li>a</li><li>b</li></ul>"),
    ("- **x** and `y`", "<ul><li><strong>x</strong> and <code>y</code></li></ul>"),
    ("1. a\n2. b\n10. c", "<ol><li>a</li><li>b</li><li>c</li></ol>"),
    ("text\n- a", "<p>text</p>\n<ul><li>a</li></ul>"),
    ("- a\ntext", "<ul><li>a</li></ul>\n<p>text</p>"),
    ("para\n# Head\nmore", "<p>para</p>\n<h1>Head</h1>\n<p>more</p>"),
    ("> a\n> b", "<blockquote><p>a\nb</p></blockquote>"),
    ("> *x*", "<blockquote><p><em>x</em></p></blockquote>"),
    ("---", "<hr>"),
    ("-----", "<hr>"),
    ("--", "<p>--</p>"),
    ("a\n---\nb", "<p>a</p>\n<hr>\n<p>b</p>"),
    ("```python\nprint('<hi>')\n\nx = 1\n```", "<pre><code>print('&lt;hi&gt;')\n\nx = 1</code></pre>"),
    ("```\n**not bold** & <b>\n```", "<pre><code>**not bold** &amp; &lt;b&gt;</code></pre>"),
    ("```\nunclosed\ncode", "<pre><code>unclosed\ncode</code></pre>"),
    ("```\n```", "<pre><code></code></pre>"),
    ("# T\n\ntext with **b**\n\n- i1\n- i2\n\n```\nc\n```",
     "<h1>T</h1>\n<p>text with <strong>b</strong></p>\n<ul><li>i1</li><li>i2</li></ul>\n<pre><code>c</code></pre>"),
]


@pytest.mark.parametrize("src,expected", CASES, ids=[f"case{i}" for i in range(len(CASES))])
def test_to_html(src, expected):
    assert to_html(src) == expected


def test_return_type_and_no_trailing_newline():
    out = to_html("# a\n\nb\n")
    assert isinstance(out, str) and not out.endswith("\n") and not out.startswith("\n")


def test_href_quotes_are_escaped():
    assert to_html('[x](/a"b)') == '<p><a href="/a&quot;b">x</a></p>'


def test_input_is_not_mutated_and_function_is_pure():
    s = "# a\n\n- b"
    assert to_html(s) == to_html(s) == "<h1>a</h1>\n<ul><li>b</li></ul>"
