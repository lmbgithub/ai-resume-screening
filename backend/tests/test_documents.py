import pytest

from resume_screen.documents import Chunk, chunk_document, load_document, normalise


def test_empty_document_yields_no_chunks():
    assert chunk_document("") == []


def test_whitespace_only_document_yields_no_chunks():
    assert chunk_document("\n\n   \n\t\n") == []


def test_bullets_become_separate_chunks():
    chunks = chunk_document("## Experience\n- first claim here\n- second claim here")
    assert [c.text for c in chunks] == ["first claim here", "second claim here"]


def test_bullets_are_flagged_as_bullets():
    chunks = chunk_document("## Experience\n- a bullet claim\nprose line follows here")
    assert [c.bullet for c in chunks] == [True, False]


def test_consecutive_prose_lines_join_into_one_chunk():
    chunks = chunk_document("Some prose here\ncontinuing on the next line")
    assert len(chunks) == 1
    assert chunks[0].text == "Some prose here continuing on the next line"


def test_blank_line_separates_paragraphs():
    chunks = chunk_document("first paragraph here\n\nsecond paragraph here")
    assert len(chunks) == 2


def test_markdown_heading_sets_section():
    chunks = chunk_document("# Experience\n- shipped a thing to production")
    assert chunks[0].section == "Experience"


def test_caps_heading_sets_section():
    chunks = chunk_document("WORK EXPERIENCE\n- shipped a thing to production")
    assert chunks[0].section == "WORK EXPERIENCE"


def test_caps_sentence_is_not_a_heading():
    chunks = chunk_document("THIS IS A SHOUTED SENTENCE.")
    assert chunks[0].section == "body"


def test_heading_itself_is_never_a_chunk():
    chunks = chunk_document("## Skills\n- python and sql")
    assert all(c.text != "Skills" for c in chunks)


@pytest.mark.parametrize("marker", ["-", "*", "•", "1.", "2)"])
def test_all_bullet_markers_are_stripped(marker):
    chunks = chunk_document(f"## S\n{marker} the claim itself here")
    assert chunks[0].text == "the claim itself here"


def test_short_fragments_are_dropped():
    assert chunk_document("## S\n- ok") == []


def test_chunk_indices_are_contiguous():
    chunks = chunk_document("## S\n- claim number one\n- claim number two\n- claim number three")
    assert [c.index for c in chunks] == [0, 1, 2]


def test_default_section_is_used_before_any_heading():
    chunks = chunk_document("a line of prose before headings", default_section="resume")
    assert chunks[0].section == "resume"


def test_normalise_collapses_runs_of_spaces_and_tabs():
    assert normalise("  a\t\t b   c ") == "a b c"


def test_chunk_is_frozen():
    chunk = Chunk(text="t", section="s", index=0, bullet=True)
    with pytest.raises(AttributeError):
        chunk.text = "other"


def test_load_document_reads_utf8(tmp_path):
    path = tmp_path / "r.md"
    path.write_text("café — naïve", encoding="utf-8")
    assert load_document(str(path)) == "café — naïve"


def test_load_document_raises_on_missing_file(tmp_path):
    with pytest.raises(OSError):
        load_document(str(tmp_path / "nope.md"))
