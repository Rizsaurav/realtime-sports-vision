# mdlite: a small Markdown converter

Please write a Python 3 module `mdlite.py` (standard library only) that exposes one function,
`to_html(text: str) -> str`, converting a small subset of Markdown to HTML. The output must follow the rules below
exactly, because other programs compare it character by character.

## Blocks

Blocks are separated by one or more blank lines (a blank line is empty or contains only spaces). The HTML of the
blocks is joined with a single `\n`, with no leading or trailing newline. Empty input, or input with only blank
lines, gives `""`. Lines may end in `\n` or `\r\n`; treat both as line breaks.

1. **Heading**: a line that starts with 1 to 6 `#` characters followed by a space. `## Title` gives
   `<h2>Title</h2>`. The text is trimmed of surrounding spaces. A heading line is always a block of its own, even with
   no blank line before or after it. Seven or more `#`, or no space after the `#`s, is not a heading.
2. **Horizontal rule**: a line that consists only of three or more `-` characters gives `<hr>`.
3. **Fenced code**: a line starting with three backticks (optionally followed by a language word, which is ignored)
   opens a block that ends at the next line that is exactly three backticks, or at the end of the input if there is no
   such line. The result is `<pre><code>` + the raw lines in between joined with `\n` + `</code></pre>`, where only
   `&`, `<` and `>` are replaced by `&amp;`, `&lt;` and `&gt;`. Nothing else is processed inside, and blank lines inside
   belong to the block.
4. **Blockquote**: consecutive lines that start with `> ` (or are exactly `>`). Remove the marker, join the lines with
   `\n`, process inline formatting, and give `<blockquote><p>` + text + `</p></blockquote>`.
5. **Unordered list**: consecutive lines that start with `- ` or `* `. The result is `<ul>` followed by one
   `<li>` + inline text + `</li>` per line, followed by `</ul>`, all with no newlines in between.
6. **Ordered list**: consecutive lines that start with digits, a dot and a space (such as `1. `). Same as above with
   `<ol>`. The numbers themselves are ignored.
7. **Paragraph**: anything else. A paragraph continues over consecutive lines (joined with `\n`) until a blank line or
   the start of any other block type, and gives `<p>` + inline text + `</p>`.

## Inline formatting

Inline formatting applies to the text of headings, paragraphs, list items and blockquotes. First `&`, `<` and `>` in the
text are replaced by `&amp;`, `&lt;` and `&gt;`; there is never any raw HTML in the output. Then:

- `` `code` `` gives `<code>code</code>`. Inside a code span nothing else is processed.
- `**text**` gives `<strong>text</strong>`.
- `*text*` gives `<em>text</em>`. Only asterisks make italics; underscores are ordinary characters.
- `[text](url)` gives `<a href="url">text</a>` when `url` starts with `http://`, `https://`, `/` or `#`. The text may
  contain the other inline formats. In the `href`, a `"` is written `&quot;` (an `&` is already `&amp;` from the first
  step). If the url starts with anything else, the whole `[text](url)` stays as literal text.
- A backslash before `*`, `` ` ``, `[`, `]` or `\` produces that character literally and removes the backslash.
- Markers that are not closed stay as literal text (`**abc` stays `**abc`).

## Examples

| Input | Output |
|---|---|
| `# Hello` | `<h1>Hello</h1>` |
| `a < b & c` | `<p>a &lt; b &amp; c</p>` |
| `**bold** and *it* and `` `x` `` | `<p><strong>bold</strong> and <em>it</em> and <code>x</code></p>` |
| `- a`⏎`- b` | `<ul><li>a</li><li>b</li></ul>` |
| `> q`⏎`> r` | `<blockquote><p>q\nr</p></blockquote>` |
| `[d](https://x.com/?a=1&b=2)` | `<p><a href="https://x.com/?a=1&amp;b=2">d</a></p>` |
