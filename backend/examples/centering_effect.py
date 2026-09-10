"""Measure what double-centering changes. Requires a running Ollama server.

Prints the raw-cosine range over every (requirement, resume-line) pair — the
"floor" that makes a raw cosine unreadable as a percentage — and how often
centering picks different evidence than raw similarity would.
"""

from pathlib import Path

from resume_screen import OllamaClient, chunk_document, parse_requirements
from resume_screen.scoring import double_center, rank_chunks, similarity_matrix

DATA = Path(__file__).resolve().parent.parent / "data"
TOP_K = 3

client = OllamaClient()
chunks = chunk_document((DATA / "resume.md").read_text(encoding="utf-8"), default_section="resume")
requirements = parse_requirements((DATA / "job.md").read_text(encoding="utf-8"))

vectors = client.embed([r.text for r in requirements] + [c.text for c in chunks])
raw = similarity_matrix(vectors[: len(requirements)], vectors[len(requirements) :])
centered = double_center(raw)

flat = [value for row in raw for value in row]
print(f"{len(requirements)} requirements x {len(chunks)} resume lines")
print(f"raw cosine: min {min(flat):.3f}  max {max(flat):.3f}  mean {sum(flat) / len(flat):.3f}")

changed_top1 = changed_set = 0
for i, requirement in enumerate(requirements):
    by_raw = [c for c, _, _ in rank_chunks(raw[i], raw[i], top_k=TOP_K)]
    by_centered = [c for c, _, _ in rank_chunks(raw[i], centered[i], top_k=TOP_K)]
    changed_top1 += by_raw[0] != by_centered[0]
    changed_set += by_raw != by_centered
    if by_raw[0] != by_centered[0]:
        print(f"\n  {requirement.text}")
        print(f"    raw picks:      {chunks[by_raw[0]].text[:70]}")
        print(f"    centered picks: {chunks[by_centered[0]].text[:70]}")

print(f"\ntop-{TOP_K} evidence set changed for {changed_set}/{len(requirements)} requirements")
print(f"top-1 evidence changed for {changed_top1}/{len(requirements)} requirements")

column_means = [(sum(row[j] for row in raw) / len(raw), chunks[j].text) for j in range(len(chunks))]
generic_mean, generic_text = max(column_means)
print(f"\nmost generic resume line (mean cosine {generic_mean:.3f}): {generic_text[:80]}")
