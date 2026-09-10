import io
import re
import zipfile
import zlib

import pytest

from resume_screen.extract import (
    MAX_UPLOAD_BYTES,
    MIN_PLAUSIBLE_WORDS,
    ExtractionError,
    extract,
    sniff_format,
)

CV_TEXT = "\n".join(
    ["# Alex Rivera", "## Experience"]
    + [f"- shipped production system number {i} with measurable impact" for i in range(12)]
)


def make_docx(paragraphs, *, body_only=False):
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(
        "<w:p>" + "".join(f"<w:t>{run}</w:t>" for run in runs) + "</w:p>" for runs in paragraphs
    )
    xml = f'<?xml version="1.0"?><w:document {ns}><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if not body_only:
            archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def make_pdf(lines, *, compress=True, extra_streams=()):
    """A minimal PDF with one content stream of Td/Tj operators."""
    ops = (
        "BT /F1 12 Tf\n"
        + "".join(f"1 0 0 1 72 {700 - 14 * i} Td ({line}) Tj\n" for i, line in enumerate(lines))
        + "ET"
    )
    payload = ops.encode("latin-1")
    if compress:
        payload = zlib.compress(payload)
    parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Length 1 >>\nstream\n",
        payload,
        b"\nendstream\nendobj\n",
    ]
    for blob in extra_streams:
        parts += [b"2 0 obj\n<< /Length 1 >>\nstream\n", blob, b"\nendstream\nendobj\n"]
    parts.append(b"trailer\n<< /Root 1 0 R >>\n%%EOF")
    return b"".join(parts)


LONG_LINES = [f"shipped production system number {i} with measurable impact" for i in range(12)]


# --- format sniffing ---------------------------------------------------------


def test_pdf_is_detected_by_magic_bytes_not_extension():
    # A CV exported as PDF but named .txt must not be decoded as text.
    assert sniff_format("cv.txt", b"%PDF-1.7\n rest") == "pdf"


def test_docx_is_detected_by_its_zip_entry():
    assert sniff_format("cv.bin", make_docx([["hello"]])) == "docx"


def test_a_zip_without_a_word_document_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(ExtractionError, match=re.escape("not a .docx")):
        sniff_format("cv.zip", buffer.getvalue())


def test_legacy_doc_is_rejected_with_advice():
    with pytest.raises(ExtractionError, match=re.escape("save as .docx")):
        sniff_format("cv.doc", b"plain bytes")


@pytest.mark.parametrize("name", ["cv.txt", "cv.md", "cv.markdown", "CV.TXT"])
def test_text_extensions(name):
    assert sniff_format(name, b"hello") == "text"


def test_unknown_extension_falls_back_to_text():
    assert sniff_format("cv", b"hello") == "text"


# --- plain text --------------------------------------------------------------


def test_text_round_trips_exactly():
    result = extract("cv.md", CV_TEXT.encode("utf-8"))
    assert result.text == CV_TEXT
    assert (result.format, result.confidence) == ("text", "exact")


def test_text_has_no_warnings():
    assert extract("cv.md", CV_TEXT.encode("utf-8")).warnings == ()


def test_utf16_is_decoded():
    assert "Alex Rivera" in extract("cv.txt", CV_TEXT.encode("utf-16")).text


def test_latin1_fallback():
    assert extract("cv.txt", (CV_TEXT + "\ncafé").encode("latin-1")).text.endswith("café")


def test_word_count_is_reported():
    assert extract("cv.md", CV_TEXT.encode("utf-8")).words == len(CV_TEXT.split())


# --- docx --------------------------------------------------------------------


def test_docx_extracts_paragraph_text():
    data = make_docx([["Alex Rivera"], ["Built a streaming gateway"]])
    assert extract("cv.docx", data).text == "Alex Rivera\nBuilt a streaming gateway"


def test_docx_joins_runs_within_a_paragraph():
    # Word splits a styled sentence across runs; joining them with newlines
    # would turn one bullet into several chunks.
    data = make_docx([["Built a ", "streaming ", "gateway"]])
    assert extract("cv.docx", data).text == "Built a streaming gateway"


def test_docx_keeps_paragraphs_on_separate_lines():
    data = make_docx([["first bullet here"], ["second bullet here"]])
    assert len(extract("cv.docx", data).text.splitlines()) == 2


def test_docx_drops_empty_paragraphs():
    data = make_docx([["real text here"], [], ["more real text"]])
    assert extract("cv.docx", data).text == "real text here\nmore real text"


def test_docx_is_exact_confidence():
    assert extract("cv.docx", make_docx([[CV_TEXT]])).confidence == "exact"


def test_a_docx_with_no_text_is_rejected():
    with pytest.raises(ExtractionError, match="no text could be extracted"):
        extract("cv.docx", make_docx([[]]))


def test_corrupt_docx_xml_is_reported():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", "<w:document><unclosed>")
    with pytest.raises(ExtractionError, match="malformed XML"):
        extract("cv.docx", buffer.getvalue())


# --- pdf ---------------------------------------------------------------------


def test_pdf_extracts_text():
    result = extract("cv.pdf", make_pdf(LONG_LINES))
    assert "shipped production system number 0" in result.text


def test_pdf_is_always_flagged_approximate():
    result = extract("cv.pdf", make_pdf(LONG_LINES))
    assert result.confidence == "approximate"
    assert any("approximate" in w for w in result.warnings)


def test_pdf_puts_each_positioned_run_on_its_own_line():
    result = extract("cv.pdf", make_pdf(LONG_LINES))
    assert len(result.text.splitlines()) == len(LONG_LINES)


def test_uncompressed_pdf_streams_are_read():
    result = extract("cv.pdf", make_pdf(LONG_LINES, compress=False))
    assert "measurable impact" in result.text


def test_embedded_font_binaries_are_not_extracted_as_text():
    # A font program is a Flate stream full of parentheses; parsing it as a
    # content stream produced pages of mojibake before this was filtered.
    font = zlib.compress(bytes(range(256)) * 20)
    result = extract("cv.pdf", make_pdf(LONG_LINES, extra_streams=[font]))
    assert result.words == sum(len(line.split()) for line in LONG_LINES)


def test_a_horizontal_move_does_not_break_a_word():
    # Generators emit Td for kerning as well as for a new line. Breaking on
    # every Td split words mid-token ("sess" / "ions"); breaking only when the
    # y operand changes is what a reader would call a line.
    same_line = b"BT /F1 12 Tf\n72 700 Td (conc) Tj\n90 700 Td (urrent) Tj\nET"
    data = b"%PDF-1.4\n1 0 obj\nstream\n" + same_line + b"\nendstream\n%%EOF"
    assert "concurrent" in extract("cv.pdf", data).text


def test_a_vertical_move_does_break_the_line():
    two_lines = b"BT /F1 12 Tf\n72 700 Td (Experience) Tj\n72 686 Td (Skills) Tj\nET"
    data = b"%PDF-1.4\n1 0 obj\nstream\n" + two_lines + b"\nendstream\n%%EOF"
    assert extract("cv.pdf", data).text.splitlines() == ["Experience", "Skills"]


def test_garbled_output_is_warned_about():
    # What a subset-encoded font actually yields: symbols and control codes,
    # not letters.
    garbled = "".join(chr(c) for c in range(0xA1, 0xBF)) * 6
    result = extract("cv.pdf", make_pdf([garbled]))
    assert any("garbled" in w for w in result.warnings)


def test_accented_latin_text_is_not_called_garbled():
    # A Spanish or French CV is full of high bytes and is perfectly fine.
    spanish = ["Ingeniería de software con énfasis en producción y evaluación"] * 4
    result = extract("cv.pdf", make_pdf(spanish))
    assert not any("garbled" in w for w in result.warnings)


def test_a_pdf_with_no_text_streams_is_rejected():
    with pytest.raises(ExtractionError, match="no text could be extracted"):
        extract("cv.pdf", b"%PDF-1.4\ntrailer\n%%EOF")


def test_scanned_pdf_advice_mentions_images():
    with pytest.raises(ExtractionError, match="images, not text"):
        extract("cv.pdf", b"%PDF-1.4\ntrailer\n%%EOF")


def test_octal_escapes_are_decoded():
    result = extract("cv.pdf", make_pdf([*LONG_LINES, r"caf\351 latte"]))
    assert "café" in result.text


def test_escaped_parentheses_do_not_end_the_string():
    result = extract("cv.pdf", make_pdf([*LONG_LINES, r"built \(fast\) systems"]))
    assert "built (fast) systems" in result.text


# --- guards ------------------------------------------------------------------


def test_empty_file_is_rejected():
    with pytest.raises(ExtractionError, match="empty"):
        extract("cv.txt", b"")


def test_oversized_file_is_rejected():
    with pytest.raises(ExtractionError, match="exceeds"):
        extract("cv.txt", b"x" * (MAX_UPLOAD_BYTES + 1))


def test_a_suspiciously_short_extraction_is_warned_about():
    result = extract("cv.txt", b"Alex Rivera, engineer")
    assert any("short for a CV" in w for w in result.warnings)


def test_a_full_length_cv_gets_no_length_warning():
    assert extract("cv.md", CV_TEXT.encode()).words >= MIN_PLAUSIBLE_WORDS
    assert not any("short for a CV" in w for w in extract("cv.md", CV_TEXT.encode()).warnings)


def test_to_dict_shape():
    body = extract("cv.md", CV_TEXT.encode()).to_dict()
    assert set(body) == {"text", "format", "confidence", "warnings", "words"}
    assert isinstance(body["warnings"], list)


# --- pdf: hex strings, CMaps and encryption ----------------------------------


def pdf_with(streams):
    """A PDF whose objects are the given (already-encoded) stream bodies."""
    parts = [b"%PDF-1.7\n"]
    for i, blob in enumerate(streams, start=1):
        parts += [
            f"{i} 0 obj\n<< /Length {len(blob)} >>\nstream\n".encode(),
            blob,
            b"\nendstream\nendobj\n",
        ]
    parts.append(b"trailer\n<< /Root 1 0 R >>\n%%EOF")
    return b"".join(parts)


def content(ops: str) -> bytes:
    return zlib.compress(ops.encode("latin-1"))


TOUNICODE = b"""/CIDInit /ProcSet findresource begin
1 begincmap
2 beginbfchar
<0044> <0048> <0045> <0065>
endbfchar
1 beginbfrange
<0046> <0048> <006C>
endbfrange
endcmap
"""


def test_a_hex_string_is_decoded_through_the_tounicode_map():
    # bfchar maps 0044->H and 0045->e; bfrange maps 0046..0048 -> l, m, n.
    ops = "BT /F1 12 Tf\n72 700 Td <00440045004600470048> Tj\nET"
    data = pdf_with([content(ops), TOUNICODE])
    assert extract("cv.pdf", data).text == "Helmn"


def test_bfrange_maps_a_span_of_codes():
    ops = "BT /F1 12 Tf\n72 700 Td <0046> Tj\n72 686 Td <0048> Tj\nET"
    data = pdf_with([content(ops), TOUNICODE])
    lines = extract("cv.pdf", data).text.splitlines()
    assert lines == ["l", "n"]


def test_a_hex_string_without_a_cmap_falls_back_to_byte_codes():
    ops = "BT /F1 12 Tf\n72 700 Td <48656C6C6F20776F726C6420616761696E> Tj\nET"
    assert "Hello world again" in extract("cv.pdf", pdf_with([content(ops)])).text


def test_whitespace_inside_a_hex_string_is_ignored():
    ops = "BT /F1 12 Tf\n72 700 Td <48 65 6C 6C 6F 20 77 6F 72 6C 64> Tj\nET"
    assert "Hello world" in extract("cv.pdf", pdf_with([content(ops)])).text


def test_an_odd_length_hex_string_is_padded_not_rejected():
    ops = "BT /F1 12 Tf\n72 700 Td <48656C6C6F20776F726C6> Tj\nET"
    assert "Hello worl" in extract("cv.pdf", pdf_with([content(ops)])).text


def test_literal_and_hex_strings_mix_in_one_stream():
    ops = "BT /F1 12 Tf\n72 700 Td (plain text here) Tj\n<616E64206865782074657874> Tj\nET"
    text = extract("cv.pdf", pdf_with([content(ops)])).text
    assert "plain text here" in text and "and hex text" in text


def test_an_encrypted_pdf_says_so_instead_of_blaming_a_scan():
    # The old message sent people looking for a scanner problem they did not
    # have; an encrypted PDF is a different fix entirely.
    data = pdf_with([content("BT (hi) Tj ET")]).replace(
        b"trailer\n<< /Root 1 0 R >>", b"trailer\n<< /Root 1 0 R /Encrypt 9 0 R >>"
    )
    with pytest.raises(ExtractionError, match="encrypted"):
        extract("cv.pdf", data)


def test_the_encryption_message_suggests_a_way_out():
    data = pdf_with([content("BT (hi) Tj ET")]).replace(
        b"trailer\n<< /Root 1 0 R >>", b"trailer\n<< /Root 1 0 R /Encrypt 9 0 R >>"
    )
    with pytest.raises(ExtractionError, match="without a password"):
        extract("cv.pdf", data)


def test_a_raw_deflate_stream_is_still_read():
    # Some producers omit the zlib header; refusing those loses the page.
    compressor = zlib.compressobj(wbits=-15)
    raw = compressor.compress(b"BT /F1 12 Tf\n72 700 Td (raw deflate content) Tj\nET")
    raw += compressor.flush()
    assert "raw deflate content" in extract("cv.pdf", pdf_with([raw])).text


def test_a_truncated_stream_yields_what_it_can():
    blob = zlib.compress(b"BT /F1 12 Tf\n72 700 Td (recoverable prefix here) Tj\nET" * 40)
    assert "recoverable prefix here" in extract("cv.pdf", pdf_with([blob[:-20]])).text


def test_the_apostrophe_show_operator_counts_as_content():
    ops = "BT /F1 12 Tf\n72 700 Td (shown with the quote operator) '\nET"
    assert "shown with the quote operator" in extract("cv.pdf", pdf_with([content(ops)])).text


def test_a_font_program_stream_is_still_excluded():
    font = zlib.compress(bytes(range(256)) * 20)
    ops = "BT /F1 12 Tf\n72 700 Td (the only real text on this page) Tj\nET"
    result = extract("cv.pdf", pdf_with([content(ops), font]))
    assert result.text.strip() == "the only real text on this page"


# --- pdf: octal escapes ------------------------------------------------------


@pytest.mark.parametrize(
    "literal,expected",
    [
        (rb"(\101)", "A"),
        (rb"(\0)", "\x00"),
        (rb"(\377)", "ÿ"),
        (rb"(\7a)", "\x07a"),
        (rb"(\78)", "\x078"),
        # The one that was wrong: filtering non-octal bytes out of the 3-byte
        # window merged `1` and `2` across the `a`, decoding \12 and dropping
        # the letter entirely.
        (rb"(\1a2)", "\x01a2"),
        (rb"(\12abc)", "\nabc"),
    ],
)
def test_octal_escapes_stop_at_the_first_non_octal_digit(literal, expected):
    from resume_screen.extract import _decode_pdf_string

    assert _decode_pdf_string(literal).decode("latin-1") == expected
