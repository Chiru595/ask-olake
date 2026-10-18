"""GitHub repos: shallow clone, include/exclude rules, file -> chunks, commit/compare API, raw fetch."""
import asyncio
import re
import shutil
from pathlib import Path

import httpx
import structlog

from app.config import Settings
from app.ingest.chunker import Chunk, Section, build_chunks, count_tokens
from app.ingest.classify import classify

log = structlog.get_logger()

EXCLUDE_PARTS = {".git", ".github", "node_modules", "vendor", "dist", "build", "test", "tests", "__tests__",
                 "testdata", "mocks", "site", "licenses-binary"}
EXCLUDE_NAMES = {"go.sum", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "CLAUDE.md", "LICENSE", "LICENCE"}
MAX_BYTES = 200_000

# olake-fusion is a fork of Apache Amoro; skip upstream docs for features Fusion does not ship
FUSION_EXCLUDE = re.compile(
    r"^(docs/engines/flink/|docs/formats/(mixed-|paimon)|docs/admin-guides/using-kyuubi|amoro-web/|amoro-common/|"
    r"docs/engines/spark/spark-(ddl|queries|writes)|docs/user-guides/cdc-ingestion)"
)


def is_included(repo: str, path: str, size: int) -> bool:
    p = Path(path)
    if size > MAX_BYTES or p.name in EXCLUDE_NAMES or any(part in EXCLUDE_PARTS for part in p.parts):
        return False
    if p.name.endswith(("_test.go", ".min.js")) or ".spec." in p.name or ".test." in p.name:
        return False
    if repo == "olake-fusion" and FUSION_EXCLUDE.match(path):
        return False
    name = p.name.lower()
    if p.suffix.lower() in (".md", ".mdx"):
        return True
    if name in ("spec.json", "chart.yaml", "values.yaml", "notes.txt", ".env.example", "api-contract.md"):
        return True
    if name.startswith("docker-compose") and name.endswith((".yml", ".yaml")) and "drivers" not in p.parts:
        return True
    if name == "config.go" and ("drivers" in p.parts or "destination" in p.parts):
        return True
    return False


def _github_slug(text: str, seen: dict[str, int]) -> str:
    slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
    n = seen.get(slug, 0)
    seen[slug] = n + 1
    return slug if n == 0 else f"{slug}-{n}"


def _clean_markdown(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images / badges
    text = re.sub(r"<img[^>]*>", "", text, flags=re.I)
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)  # frontmatter
    return re.sub(r"\n{3,}", "\n\n", text)


def markdown_sections(text: str, url: str, title: str) -> list[Section]:
    text = _clean_markdown(text)
    sections: list[Section] = []
    seen: dict[str, int] = {}
    cur_path, cur_anchor, cur_h2, buf = [title], None, None, []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(url=url + (f"#{cur_anchor}" if cur_anchor else ""), heading_path=list(cur_path),
                                    text=body, h2=cur_h2))

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^(#{1,3})\s+(.+?)\s*#*$", line)
        if m:
            flush()
            buf = []
            level, head = len(m.group(1)), re.sub(r"[*`]", "", m.group(2)).strip()
            cur_anchor = _github_slug(head, seen)
            if level == 1:
                cur_path, cur_h2 = [title], None
            elif level == 2:
                cur_h2 = head
                cur_path = [title, head]
            else:
                cur_path = [title] + ([cur_h2] if cur_h2 else []) + [head]
            continue
        buf.append(line)
    flush()
    return sections


def _line_blocks(lines: list[str], starts: re.Pattern) -> list[tuple[int, int]]:
    idx = [i for i, ln in enumerate(lines) if starts.match(ln)] or [0]
    if idx[0] != 0:
        idx.insert(0, 0)
    bounds = idx + [len(lines)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def code_sections(text: str, url: str, title: str, kind: str) -> list[Section]:
    """Config/spec/Go files: whole file if small, else grouped top-level blocks with line ranges."""
    lines = text.split("\n")
    if count_tokens(text) <= 500:
        return [Section(url=url, heading_path=[title], text=f"```{kind}\n{text.strip()}\n```", line_start=1, line_end=len(lines))]
    if kind == "go":
        pat = re.compile(r"^(type|func|var|const)\b")
    elif kind == "json":
        pat = re.compile(r'^\s{0,8}"[\w\-]+"\s*:\s*\{')
    else:
        pat = re.compile(r"^[A-Za-z_][\w\-]*:")
    sections: list[Section] = []
    group: list[tuple[int, int]] = []

    def flush() -> None:
        if not group:
            return
        a, b = group[0][0], group[-1][1]
        body = "\n".join(lines[a:b]).strip()
        if body:
            first = lines[a].strip()[:80]
            sections.append(Section(url=url, heading_path=[title, first], text=f"```{kind}\n{body}\n```",
                                    line_start=a + 1, line_end=b))

    tok = 0
    for a, b in _line_blocks(lines, pat):
        t = count_tokens("\n".join(lines[a:b]))
        if group and tok + t > 450:
            flush()
            group, tok = [], 0
        group.append((a, b))
        tok += t
    flush()
    return sections


def file_to_chunks(repo: str, branch: str, sha: str | None, path: str, text: str, s: Settings,
                   last_updated: str | None = None) -> list[Chunk]:
    url = f"https://github.com/{s.github_org}/{repo}/blob/{branch}/{path}"
    title = f"{repo} > {path}"
    suffix = Path(path).suffix.lower()
    component, connector = classify(url, repo, path)
    if suffix in (".md", ".mdx"):
        sections = markdown_sections(text, url, title)
        line_anchor = False
    else:
        kind = {".go": "go", ".json": "json", ".yaml": "yaml", ".yml": "yaml"}.get(suffix, "text")
        sections = code_sections(text, url, title, kind)
        line_anchor = True
    if sum(count_tokens(x.text) for x in sections) < 50:
        return []
    return build_chunks(sections, doc_url=url, title=title, source_type="repo", component=component,
                        connector=connector, repo=repo, path=path, last_updated=last_updated, commit_sha=sha,
                        line_anchor=line_anchor)


async def run(cmd: list[str], cwd: str | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(proc.communicate(), timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {err.decode()[:300]}")
    return out.decode().strip()


async def clone_repo(repo: str, s: Settings) -> tuple[Path, str, str]:
    """Fresh shallow clone. Returns (dir, branch, sha). Blobs over 1 MB are skipped to keep clones small."""
    if shutil.which("git") is None:  # e.g. serverless runtimes; caller keeps the previous state
        raise RuntimeError("git is not available for a full repo resync")
    root = Path(s.workdir) / "repos"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / repo
    if dest.exists():
        await run(["rm", "-rf", str(dest)])
    await run(["git", "clone", "--depth", "1", "--filter=blob:limit=1m", "--no-tags", "-q",
               f"https://github.com/{s.github_org}/{repo}.git", str(dest)])
    branch = await run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(dest))
    sha = await run(["git", "rev-parse", "HEAD"], cwd=str(dest))
    return dest, branch, sha


def walk_clone(repo: str, dest: Path, branch: str, sha: str, s: Settings, committed_at: str | None) -> dict[str, list[Chunk]]:
    """Return {path: chunks} for every included file in a clone."""
    out: dict[str, list[Chunk]] = {}
    for p in sorted(dest.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel = str(p.relative_to(dest))
        try:
            if not is_included(repo, rel, p.stat().st_size):
                continue
            text = p.read_text(errors="replace")
        except OSError:
            continue
        chunks = file_to_chunks(repo, branch, sha, rel, text, s, committed_at)
        if chunks:
            out[rel] = chunks
    return out


class GitHub:
    def __init__(self, client: httpx.AsyncClient, s: Settings):
        self.c, self.s = client, s
        self.h = {"Accept": "application/vnd.github+json", "User-Agent": "AskOLakeBot/1.0"}
        if s.github_token:
            self.h["Authorization"] = f"Bearer {s.github_token}"

    async def head_commit(self, repo: str, branch: str, etag: str | None) -> tuple[int, str | None, str | None, str | None]:
        """Returns (status, sha, etag, committed_at). 304 = unchanged."""
        h = dict(self.h)
        if etag:
            h["If-None-Match"] = etag
        r = await self.c.get(f"https://api.github.com/repos/{self.s.github_org}/{repo}/commits/{branch}", headers=h, timeout=15)
        if r.status_code == 304:
            return 304, None, etag, None
        r.raise_for_status()
        j = r.json()
        return 200, j["sha"], r.headers.get("etag"), j["commit"]["committer"]["date"]

    async def compare(self, repo: str, base: str, head: str) -> list[dict] | None:
        """Changed files between two SHAs, or None when the diff is too large / unavailable."""
        r = await self.c.get(f"https://api.github.com/repos/{self.s.github_org}/{repo}/compare/{base}...{head}",
                             headers=self.h, timeout=20)
        if r.status_code != 200:
            return None
        files = r.json().get("files", [])
        return None if len(files) >= 300 else files

    async def raw(self, repo: str, sha: str, path: str) -> str | None:
        r = await self.c.get(f"https://raw.githubusercontent.com/{self.s.github_org}/{repo}/{sha}/{path}",
                             headers={"User-Agent": "AskOLakeBot/1.0"}, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text

