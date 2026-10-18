"""Structure-aware chunking with deterministic IDs and content hashes."""
import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import tiktoken

NAMESPACE = uuid.UUID("5d7c3c1e-6c55-4b1a-9a43-0a1e0b1a5e01")
TARGET_MAX = 500
MERGE_BELOW = 80
OVERLAP = 50
ATOMIC_MAX = 800

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


@dataclass
class Section:
    """One heading section of a page or file before size normalisation."""

    url: str
    heading_path: list[str]
    text: str
    h2: str | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class Chunk:
    chunk_id: str
    url: str
    doc_url: str
    title: str
    heading_path: list[str]
    text: str
    source_type: str
    repo: str | None
    path: str | None
    component: str
    connector: str | None
    content_hash: str
    last_updated: str | None
    commit_sha: str | None
    token_count: int
    indexed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def embed_text(self) -> str:
        return f"[{' > '.join(self.heading_path)}]\n{self.text}"

    def payload(self) -> dict:
        return asdict(self)


def make_chunk_id(canonical_url: str, heading_path: list[str], part: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{canonical_url}|{' > '.join(heading_path)}|{part}"))


def content_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _blocks(text: str) -> list[str]:
    """Split markdown into paragraph-level blocks without breaking fenced code or tables."""
    blocks, cur, in_fence = [], [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            cur.append(line)
            continue
        if not in_fence and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def split_text(text: str) -> list[str]:
    """Split oversized text at block boundaries with a small trailing overlap."""
    if count_tokens(text) <= TARGET_MAX:
        return [text]
    parts, cur, cur_tok = [], [], 0
    for block in _blocks(text):
        bt = count_tokens(block)
        if cur and cur_tok + bt > TARGET_MAX:
            parts.append("\n\n".join(cur))
            tail = cur[-1] if count_tokens(cur[-1]) <= OVERLAP and not cur[-1].lstrip().startswith("```") else None
            cur, cur_tok = ([tail], count_tokens(tail)) if tail else ([], 0)
        if bt > TARGET_MAX and not block.lstrip().startswith(("```", "|")):
            # plain prose block that's huge: hard split by lines
            lines, buf = block.split("\n"), []
            for ln in lines:
                buf.append(ln)
                if count_tokens("\n".join(buf)) > TARGET_MAX:
                    parts.append("\n".join(buf))
                    buf = []
            if buf:
                cur.append("\n".join(buf))
                cur_tok += count_tokens(cur[-1])
            continue
        cur.append(block)
        cur_tok += bt
    if cur:
        parts.append("\n\n".join(cur))
    return [p for p in parts if p.strip()]


def normalise_sections(sections: list[Section]) -> list[Section]:
    """Merge tiny sections into the next sibling under the same H2, then split big ones."""
    merged: list[Section] = []
    pending: Section | None = None
    for s in sections:
        if pending is not None:
            if pending.h2 == s.h2:
                s = Section(
                    url=pending.url,
                    heading_path=pending.heading_path,
                    text=f"{pending.text}\n\n#### {s.heading_path[-1]}\n\n{s.text}",
                    h2=s.h2,
                    line_start=pending.line_start,
                    line_end=s.line_end,
                )
            else:
                merged.append(pending)
            pending = None
        if count_tokens(s.text) < MERGE_BELOW:
            pending = s
        else:
            merged.append(s)
    if pending is not None:
        merged.append(pending)
    return merged


def build_chunks(sections: list[Section], *, doc_url: str, title: str, source_type: str, component: str,
                 connector: str | None, repo: str | None = None, path: str | None = None,
                 last_updated: str | None = None, commit_sha: str | None = None,
                 line_anchor: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s in normalise_sections(sections):
        for i, part in enumerate(split_text(s.text)):
            url = s.url
            if line_anchor and s.line_start:
                url = f"{doc_url}#L{s.line_start}-L{s.line_end or s.line_start}"
            chunks.append(Chunk(
                chunk_id=make_chunk_id(s.url, s.heading_path, i),
                url=url,
                doc_url=doc_url,
                title=title,
                heading_path=s.heading_path,
                text=part,
                source_type=source_type,
                repo=repo,
                path=path,
                component=component,
                connector=connector,
                content_hash=content_hash(part),
                last_updated=last_updated,
                commit_sha=commit_sha,
                token_count=count_tokens(part),
            ))
    return chunks
