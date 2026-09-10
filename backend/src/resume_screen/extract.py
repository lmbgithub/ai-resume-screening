"""Turning an uploaded CV into text, and being honest about how well it worked.

Three formats, three very different confidence levels, and the caller is told
which one it got:

* **text/markdown** — decode and you are done. `exact`.
* **docx** — a zip of XML. `w:t` elements hold the runs and `w:p` the
  paragraphs, so the structure survives. `exact`.
* **pdf** — approximate, and unavoidably so. A PDF stores glyph-placement
  operators, not paragraphs; a two-column CV has no reading order to recover
  without layout analysis. The extractor is best-effort and says so, and the
  UI shows the result for the user to correct before anything is scored.

The alternative — a PDF extractor that silently interleaves two columns into
one line — poisons every downstream score while looking like it worked. Whether
extraction is trustworthy is the caller's decision to make, so it is returned
rather than hidden.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import dataclass, field
from io import BytesIO

MAX_UPLOAD_BYTES = 5_000_000
# A CV with fewer words than this is almost certainly a failed extraction.
MIN_PLAUSIBLE_WORDS = 40

WORD_DOC = "word/document.xml"
DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class ExtractionError(ValueError):
    """The file could not be turned into text at all."""


@dataclass(frozen=True)
class Extracted:
    """Extracted text, plus what the caller needs to judge it."""

    text: str
    format: str
    confidence: str  # "exact" | "approximate"
    warnings: tuple[str, ...] = field(default=())

    @property
    def words(self) -> int:
        return len(re.findall(r"\S+", self.text))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "format": self.format,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "words": self.words,
        }


def sniff_format(filename: str, data: bytes) -> str:
    """Identify by magic bytes first, extension second.

    A file named `.txt` that begins with `%PDF-` is a PDF, and treating it as
    text would hand the model a screenful of binary rather than a CV.
    """
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"PK\x03\x04"):
        # Every OOXML file is a zip; only a docx has word/document.xml.
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                if WORD_DOC in archive.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        raise ExtractionError(
            "this looks like a zip archive but not a .docx "
            "(no word/document.xml). Old .doc files are not supported."
        )
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {"txt", "md", "markdown", "text"}:
        return "text"
    if suffix in {"doc"}:
        raise ExtractionError("legacy .doc is not supported — save as .docx or .pdf")
    # Unknown extension, no magic bytes: try text and let decoding decide.
    return "text"


def extract(filename: str, data: bytes) -> Extracted:
    """Extract text from an uploaded CV."""
    if not data:
        raise ExtractionError("the uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ExtractionError(f"file exceeds {MAX_UPLOAD_BYTES // 1_000_000} MB")

    fmt = sniff_format(filename, data)
    extracted = {"text": _extract_text, "docx": _extract_docx, "pdf": _extract_pdf}[fmt](data)

    if not extracted.text.strip():
        raise ExtractionError(
            f"no text could be extracted from this {fmt} file. "
            "If it is a scanned document it contains images, not text."
        )
    return _add_length_warning(extracted)


def _add_length_warning(extracted: Extracted) -> Extracted:
    if extracted.words >= MIN_PLAUSIBLE_WORDS:
        return extracted
    return Extracted(
        text=extracted.text,
        format=extracted.format,
        confidence=extracted.confidence,
        warnings=(
            *extracted.warnings,
            f"only {extracted.words} words were extracted, which is short for a CV — "
            "check the text below before analysing",
        ),
    )


def _extract_text(data: bytes) -> Extracted:
    """Decode, trying UTF-16 only when a BOM asks for it.

    `bytes.decode("utf-16")` accepts almost any even-length input and returns
    CJK-looking mojibake, so offering it as a blind fallback silently mangles
    Latin-1 files. The BOM is the only reliable signal available here.
    """
    encodings = ["utf-8"]
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings.insert(0, "utf-16")
    encodings.append("latin-1")
    for encoding in encodings:
        try:
            return Extracted(text=data.decode(encoding), format="text", confidence="exact")
        except UnicodeDecodeError:
            continue
    raise ExtractionError("the file is not readable as text in UTF-8, UTF-16 or Latin-1")


def _extract_docx(data: bytes) -> Extracted:
    """Read `word/document.xml`, one line per `w:p` paragraph.

    Paragraph boundaries matter: chunking downstream splits on lines, so
    joining every run into one blob would turn a whole CV into one chunk.
    """
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            document = archive.read(WORD_DOC)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ExtractionError(f"not a readable .docx file: {exc}") from exc
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise ExtractionError(f"the .docx contains malformed XML: {exc}") from exc

    lines: list[str] = []
    for paragraph in root.iter(f"{{{DOCX_NS['w']}}}p"):
        runs = [node.text or "" for node in paragraph.iter(f"{{{DOCX_NS['w']}}}t")]
        line = "".join(runs).strip()
        if line:
            lines.append(line)
    warnings: tuple[str, ...] = ()
    if not lines:
        warnings = ("the document body contained no text runs",)
    return Extracted(text="\n".join(lines), format="docx", confidence="exact", warnings=warnings)


# --- PDF ---------------------------------------------------------------------

_STREAM = re.compile(rb"stream\r?\n(.*?)endstream", re.DOTALL)
# Show-text operands — literal `(...)` and hex `<...>` strings — plus the
# operators that move the text cursor.
_SHOW_TEXT = re.compile(
    rb"\((?:\\.|[^\\()])*\)"
    rb"|<[0-9A-Fa-f\s]*>"
    rb"|(-?[\d.]+)\s+(-?[\d.]+)\s+(TD|Td)\b"
    rb"|\bT\*|\bET\b"
)
_PDF_ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f"}
_ENCRYPT = re.compile(rb"/Encrypt\s+\d+\s+\d+\s+R")

# ToUnicode CMaps: the only reliable way to read text set in a subset font.
_BFCHAR = re.compile(rb"beginbfchar(.*?)endbfchar", re.DOTALL)
_BFRANGE = re.compile(rb"beginbfrange(.*?)endbfrange", re.DOTALL)
_HEX = re.compile(rb"<([0-9A-Fa-f]+)>")


def _decode_pdf_string(raw: bytes) -> bytes:
    """Unescape a PDF literal string, minus the surrounding parentheses."""
    body = raw[1:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        char = body[i : i + 1]
        if char != b"\\":
            out += char
            i += 1
            continue
        nxt = body[i + 1 : i + 2]
        if nxt in _PDF_ESCAPES:
            out += _PDF_ESCAPES[nxt]
            i += 2
        elif nxt.isdigit():  # octal escape, up to three digits
            # The run must stop at the first non-octal byte rather than
            # filtering them out: filtering turned `\1a2` into `\12`, which
            # both decoded wrongly and swallowed the `a`.
            octal = bytearray()
            for byte in body[i + 1 : i + 4]:
                if not 0x30 <= byte <= 0x37:
                    break
                octal.append(byte)
            out += bytes([int(octal, 8) & 0xFF]) if octal else b""
            i += 1 + max(len(octal), 1)
        else:
            out += nxt
            i += 2
    return bytes(out)


def _inflate(blob: bytes) -> bytes:
    """Decompress a stream, tolerating the ways real PDFs are malformed.

    `zlib.decompress` refuses a stream that is truncated or missing its zlib
    header, both of which occur in the wild; `decompressobj` returns what it
    managed to read instead of nothing at all.
    """
    for wbits in (15, -15):
        try:
            return zlib.decompressobj(wbits).decompress(blob)
        except zlib.error:
            continue
    return blob


def _parse_tounicode(blob: bytes) -> dict[int, str]:
    """Read one ToUnicode CMap into {character code: text}."""
    mapping: dict[int, str] = {}
    for match in _BFCHAR.finditer(blob):
        pairs = _HEX.findall(match.group(1))
        for src, dst in zip(pairs[0::2], pairs[1::2], strict=False):
            mapping[int(src, 16)] = _utf16_be(dst)
    for match in _BFRANGE.finditer(blob):
        codes = _HEX.findall(match.group(1))
        # `<lo> <hi> <dst>` triples; the array form of bfrange is rarer and
        # is skipped rather than guessed at.
        for lo, hi, dst in zip(codes[0::3], codes[1::3], codes[2::3], strict=False):
            start, stop, base = int(lo, 16), int(hi, 16), int(dst, 16)
            if stop < start or stop - start > 0xFFFF:
                continue
            for offset in range(stop - start + 1):
                mapping[start + offset] = chr(base + offset)
    return mapping


def _utf16_be(hex_digits: bytes) -> str:
    raw = bytes.fromhex(hex_digits.decode("ascii"))
    if len(raw) % 2:
        raw += b"\x00"
    try:
        return raw.decode("utf-16-be")
    except UnicodeDecodeError:
        return ""


def _extract_pdf(data: bytes) -> Extracted:
    """Best-effort text from PDF content streams.

    Streams are inflated, the show-text operators are read in file order, and
    hex strings are mapped through the document's ToUnicode CMaps. The limits
    are real and reported: no layout analysis, so column order is whatever the
    producer wrote, and a font with no ToUnicode cannot be decoded at all.
    """
    if _ENCRYPT.search(data):
        raise ExtractionError(
            "this PDF is encrypted, so its text cannot be read. Re-export it "
            "without a password, or print it to a new PDF, or paste the text."
        )

    blobs = [_inflate(match.group(1)) for match in _STREAM.finditer(data)]

    # ToUnicode CMaps are collected across the whole document and merged.
    # Per-font maps would need font-resource tracking through `Tf`; merging is
    # wrong only where two subset fonts assign different text to one code,
    # which is rare next to the alternative of reading nothing.
    unicode_map: dict[int, str] = {}
    for blob in blobs:
        if b"beginbfchar" in blob or b"beginbfrange" in blob:
            unicode_map.update(_parse_tounicode(blob))

    pieces = [_read_show_text(blob, unicode_map) for blob in blobs if _is_content_stream(blob)]
    text = _tidy_pdf_text("\n".join(p for p in pieces if p.strip()))

    warnings = [
        "PDF text extraction is approximate: reading order is not recovered, so "
        "multi-column layouts may interleave. Check the text before analysing.",
    ]
    if blobs and not pieces:
        warnings.append(
            "no page-content streams were found — this PDF may use an unsupported "
            "compression filter, or be a scan containing images rather than text"
        )
    if _looks_like_mojibake(text):
        warnings.append(
            "the extracted characters look garbled, which usually means the PDF "
            "embeds subset fonts with no ToUnicode map — paste the text instead"
        )
    return Extracted(text=text, format="pdf", confidence="approximate", warnings=tuple(warnings))


def _is_content_stream(blob: bytes) -> bool:
    """True for a page description, false for a font program or an image.

    The test is structural: a content stream opens a text object with `BT` and
    shows glyphs with `Tj`/`TJ`/`'`/`"`. An earlier version also required the
    bytes to be mostly printable ASCII, which correctly rejected font programs
    and then rejected the one case worth keeping — a page whose text extracts
    as high-byte mojibake, which the caller needs to see warned rather than
    silently dropped.
    """
    if b"BT" not in blob:
        return False
    return any(op in blob for op in (b"Tj", b"TJ", b"'", b'"'))


def _read_show_text(blob: bytes, unicode_map: dict[int, str]) -> str:
    """Concatenate show-text operands, breaking on a real vertical move.

    Generators emit `Td` both to start a new line *and* to nudge the cursor
    horizontally for kerning within one. Breaking on every `Td` splits words
    mid-token; breaking only when the y operand changes matches what a reader
    would call a line.
    """
    out: list[str] = []
    last_y: float | None = None
    for match in _SHOW_TEXT.finditer(blob):
        token = match.group(0)
        if token.startswith(b"("):
            out.append(_decode_bytes(_decode_pdf_string(token), unicode_map))
            continue
        if token.startswith(b"<"):
            out.append(_decode_hex_string(token, unicode_map))
            continue
        if token in (b"T*", b"ET"):
            out.append("\n")
            last_y = None
            continue
        try:
            y = float(match.group(2))
        except (TypeError, ValueError):
            continue
        if last_y is not None and y != last_y:
            out.append("\n")
        last_y = y
    return "".join(out)


def _decode_hex_string(token: bytes, unicode_map: dict[int, str]) -> str:
    """Decode `<...>`, through the ToUnicode map when there is one.

    Hex strings in a subset font hold glyph ids, not characters, so without a
    CMap there is nothing meaningful to decode; two-byte codes are tried first
    because CID fonts are the common reason a PDF uses hex at all.
    """
    digits = re.sub(rb"\s", b"", token[1:-1])
    if len(digits) % 2:
        digits += b"0"
    try:
        raw = bytes.fromhex(digits.decode("ascii"))
    except ValueError:
        return ""
    if unicode_map:
        wide = [int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw) - 1, 2)]
        if wide and all(code in unicode_map for code in wide):
            return "".join(unicode_map[code] for code in wide)
        if all(byte in unicode_map for byte in raw):
            return "".join(unicode_map[byte] for byte in raw)
    return _decode_bytes(raw, unicode_map)


def _decode_bytes(raw: bytes, unicode_map: dict[int, str]) -> str:
    """A simple-font string: byte codes, mapped where the document says how."""
    if unicode_map and all(byte in unicode_map for byte in raw):
        return "".join(unicode_map[byte] for byte in raw)
    return raw.decode("latin-1")


def _tidy_pdf_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _looks_like_mojibake(text: str) -> bool:
    """Heuristic: mostly non-alphabetic output means the encoding was wrong."""
    stripped = [c for c in text if not c.isspace()]
    if len(stripped) < 40:
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    return letters / len(stripped) < 0.5
