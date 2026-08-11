"""Markdown chunker for the file knowledge base.

Splits a markdown file into pieces bounded by `chunk_size` characters while
respecting heading boundaries, so a chunk never starts mid-section. Each chunk
carries `file`, `heading` (the nearest heading path) and `text` folded into
the text itself when `include_sources` is on.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    file: str
    heading: str
    text: str

    def to_display_text(self, include_sources: bool = True) -> str:
        if not include_sources:
            return self.text
        head = f"[{self.file}" + (f" > {self.heading}]" if self.heading else "]")
        return f"{head}\n{self.text}"


def _strip_code_fences(text: str) -> str:
    lines = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def _split_into_units(text: str) -> list[str]:
    """Split text into sentence-ish units: lines first, then Thai/English
    sentence enders (`.`, `!`, `?`) within each line. Chunks built from
    these units never cut mid-sentence or mid-word."""
    units: list[str] = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if sent:
                units.append(sent)
    return units


def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Split text at sentence boundaries into chunks of ~`size` characters.

    Each piece ends on a unit boundary, so no chunk starts mid-sentence or
    mid-word. `overlap` (when > 0) keeps one extra unit at the start of the
    next chunk so context around the boundary is preserved for retrieval.
    """
    if len(text) <= size:
        return [text.strip()]
    units = _split_into_units(text)
    pieces: list[str] = []
    i = 0
    n = len(units)
    overlap_units = 1 if overlap > 0 else 0
    while i < n:
        piece = units[i]
        j = i + 1
        while j < n and len(piece) + len(units[j]) + 1 <= size:
            piece += " " + units[j]
            j += 1
        pieces.append(piece)
        i = max(j - overlap_units, i + 1)
    return pieces


def chunk_markdown(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Chunk a markdown document into a list of Chunk objects."""
    text = _strip_code_fences(text or "")

    headings = []  # stack of (level, title)
    sections = []  # list of (heading_path, section_text)

    current_heading = ""
    current_lines = []

    def flush():
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))
        current_lines = []

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            headings = [(lvl, h) for lvl, h in headings if lvl < level] + [
                (level, title)
            ]
            current_heading = " > ".join(h for _, h in headings)
        else:
            current_lines.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, body in sections:
        for piece in _split_with_overlap(body, chunk_size, chunk_overlap):
            chunks.append(Chunk(file=source, heading=heading, text=piece))

    # Files with no headings at all -> whole body is one section
    if not chunks:
        for piece in _split_with_overlap(text, chunk_size, chunk_overlap):
            chunks.append(Chunk(file=source, heading="", text=piece))
    return chunks


def load_folder_files(
    folder_path: str,
    extension: str = ".md",
) -> list[tuple[str, str]]:
    """Return [(relative_path, content)] for every file with `extension`
    under folder_path (recursively)."""
    root = Path(folder_path)
    result = []
    if not root.is_dir():
        return result
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() == extension.lower():
            try:
                content = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = p.read_text(encoding="utf-8-sig", errors="replace")
            result.append((str(p.relative_to(root)).replace("\\", "/"), content))
    return result
