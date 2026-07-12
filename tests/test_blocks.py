"""Chandra layout chunks: HTML line splitting and page synthesis."""

from __future__ import annotations

from paperless_chandra.engine.blocks import (
    block_lines,
    markdown_sidecar,
    page_from_chunks,
    page_sidecar,
)

# ---------------------------------------------------------------- block_lines


def test_paragraphs_become_separate_lines():
    assert block_lines("<p>first</p><p>second</p>") == ["first", "second"]


def test_br_splits_lines_inside_a_paragraph():
    assert block_lines("<p>first<br>second</p>") == ["first", "second"]


def test_list_items_become_lines():
    assert block_lines("<ul><li>alpha</li><li>beta</li></ul>") == ["alpha", "beta"]


def test_table_rows_one_line_cells_space_joined():
    html = "<table><tr><td>a1</td><td>a2</td></tr><tr><td>b1</td><td>b2</td></tr></table>"
    assert block_lines(html) == ["a1 a2", "b1 b2"]


def test_math_keeps_latex_source():
    assert block_lines("<p>area <math>x^2 + y_1</math> end</p>") == ["area x^2 + y_1 end"]


def test_checkboxes_render_as_brackets():
    html = '<p><input type="checkbox" checked/> yes <input type="checkbox"/> no</p>'
    assert block_lines(html) == ["[x] yes [ ] no"]


def test_plain_text_without_tags_is_one_line():
    assert block_lines("just words") == ["just words"]


def test_whitespace_is_collapsed_and_empties_dropped():
    assert block_lines("<p>  a   b  </p><p>   </p>") == ["a b"]


# ---------------------------------------------------------- page_from_chunks


def _chunk(label="Text", bbox=(100, 100, 500, 200), content="<p>hello world</p>"):
    return {"label": label, "bbox": list(bbox), "content": content}


def test_page_synthesises_lines_and_words():
    page = page_from_chunks([_chunk()], 1000, 1000, "eng")
    assert len(page.blocks) == 1
    line = page.blocks[0].lines[0]
    assert line.text == "hello world"
    assert [w.text for w in line.words] == ["hello", "world"]
    assert line.box == (100, 100, 500, 200)
    assert all(w.confidence == 95 for w in line.words)


def test_multi_line_block_splits_box_vertically():
    page = page_from_chunks(
        [_chunk(content="<p>one</p><p>two</p>", bbox=(0, 0, 400, 100))], 1000, 1000, "eng"
    )
    first, second = page.blocks[0].lines
    assert first.box == (0, 0, 400, 50)
    assert second.box == (0, 50, 400, 100)


def test_non_text_labels_are_skipped():
    chunks = [
        _chunk(label="Image", content='<img alt="a cat"/>'),
        _chunk(label="Figure", content="<p>figure narration</p>"),
        _chunk(label="Blank-Page", content=""),
        _chunk(label="Text", content="<p>kept</p>"),
    ]
    page = page_from_chunks(chunks, 1000, 1000, "eng")
    assert len(page.blocks) == 1
    assert page.blocks[0].lines[0].text == "kept"


def test_unknown_labels_are_kept():
    page = page_from_chunks([_chunk(label="block")], 1000, 1000, "eng")
    assert len(page.blocks) == 1


def test_degenerate_bbox_widens_to_full_page():
    page = page_from_chunks([_chunk(bbox=(0, 0, 1, 1))], 800, 600, "eng")
    assert page.blocks[0].box == (0, 0, 800, 600)


def test_bbox_is_clamped_to_page():
    page = page_from_chunks([_chunk(bbox=(500, 500, 2000, 2000))], 800, 600, "eng")
    assert page.blocks[0].box == (500, 500, 800, 600)


def test_empty_content_produces_no_block():
    page = page_from_chunks([_chunk(content="  ")], 1000, 1000, "eng")
    assert page.blocks == []


# ----------------------------------------------------------------- sidecars


def test_page_sidecar_joins_blocks_with_blank_lines():
    chunks = [
        _chunk(content="<p>one</p><p>two</p>", bbox=(0, 0, 400, 100)),
        _chunk(content="<p>three</p>", bbox=(0, 200, 400, 300)),
    ]
    page = page_from_chunks(chunks, 1000, 1000, "eng")
    assert page_sidecar(page) == "one\ntwo\n\nthree"


def test_markdown_sidecar_converts_headings_and_keeps_footers():
    raw = (
        '<div data-label="Section-Header" data-bbox="0 0 500 50"><h1>Title</h1></div>'
        '<div data-label="Page-Footer" data-bbox="0 900 500 950"><p>page 1 of 2</p></div>'
    )
    md = markdown_sidecar(raw)
    assert "# Title" in md
    assert "page 1 of 2" in md
