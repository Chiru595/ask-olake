"""Background sync: docs sitemap + GitHub repos every POLL_INTERVAL_SECONDS, re-indexing only what changed."""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import structlog

from app.cache import AnswerCache
from app.config import Settings
from app.ingest.chunker import Chunk
from app.ingest.docs_crawler import fetch_page, fetch_sitemap, page_to_chunks
from app.ingest.indexer import Indexer, f_doc, f_eq, f_repo_path
from app.ingest.repo_loader import GitHub, clone_repo, file_to_chunks, is_included, walk_clone
from app.sync.state import StateStore

log = structlog.get_logger()
SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SyncResult:
    changed: list[Chunk] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    dense: dict[str, list[float]] = field(default_factory=dict)

    def add(self, r: tuple[list[Chunk], list[str], dict[str, list[float]]]) -> None:
        self.changed += r[0]
        self.deleted += r[1]
        self.dense.update(r[2])

    @property
    def any(self) -> bool:
        return bool(self.changed or self.deleted)


class Poller:
    def __init__(self, s: Settings, http: httpx.AsyncClient, indexer: Indexer, state: StateStore, cache: AnswerCache):
        self.s, self.http, self.ix, self.state, self.cache = s, http, indexer, state, cache
        self.gh = GitHub(http, s)
        self.lock = asyncio.Lock()
        self.cycle = 0
        self.ready = False
        self.info: dict = {"last_poll": None, "last_change_detected": None, "last_cycle_seconds": None,
                           "last_error": None, "evicted_last_sync": 0, "sources": {}}
        self._task: asyncio.Task | None = None
        self._lazy_task: asyncio.Task | None = None
        self._last_cycle_wall = 0.0

    # ---------- lifecycle ----------
    async def start(self) -> None:
        await self.ix.ensure_collections()
        meta = await self.state.get("meta")
        mismatch = (meta.get("embed_model"), meta.get("embed_dim"), meta.get("schema_version")) != (
            self.s.embed_model, self.s.embed_dim, SCHEMA_VERSION)
        if mismatch and meta:
            log.warning("embedding_config_changed_full_reindex", old=meta)
            await self.ix.ensure_collections(recreate=True)
        if mismatch or await self.ix.count() == 0:
            await self.run_cycle(full=True)
            await self.state.put("meta", {"embed_model": self.s.embed_model, "embed_dim": self.s.embed_dim,
                                          "schema_version": SCHEMA_VERSION, "last_full_reindex": now()})
        else:
            await self.load_status()
        self.ready = await self.ix.count() > 0
        self._task = asyncio.create_task(self._loop())

    async def start_serverless(self) -> None:
        """Serverless: the index is prebuilt in Qdrant; never full-index or loop, just load state."""
        await self.load_status()
        self.ready = await self.ix.count() > 0

    def maybe_sync(self) -> None:
        """Serverless freshness: kick off an incremental cycle when the last one is older than the poll interval."""
        if self.lock.locked() or (self._lazy_task and not self._lazy_task.done()):
            return
        if time.time() - self._last_cycle_wall < self.s.poll_interval_seconds:
            return
        self._last_cycle_wall = time.time()
        self._lazy_task = asyncio.create_task(self._safe("lazy_sync", self.run_cycle(full=False)))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.poll_interval_seconds)
            try:
                await self.run_cycle(full=False)
                self.ready = self.ready or await self.ix.count() > 0
            except Exception as e:  # noqa: BLE001 - poller must never die
                log.exception("poll_cycle_failed", error=str(e))
                self.info["last_error"] = str(e)[:300]

    async def load_status(self) -> None:
        docs = await self.state.get("docs")
        if docs.get("last_checked"):
            self.info["last_poll"] = self.info["last_poll"] or docs["last_checked"]
            try:
                self._last_cycle_wall = max(self._last_cycle_wall, datetime.fromisoformat(docs["last_checked"]).timestamp())
            except ValueError:
                pass
        self.info["sources"]["docs"] = {k: docs.get(k) for k in ("pages", "chunks", "last_checked", "last_changed")}
        for repo in self.s.repo_list:
            st = await self.state.get(f"repo:{repo}")
            self.info["sources"][repo] = {"sha": st.get("sha"), "branch": st.get("branch"), "files": st.get("files"),
                                          "chunks": st.get("chunks"), "last_checked": st.get("last_checked"),
                                          "last_changed": st.get("last_changed")}

    # ---------- one cycle ----------
    async def run_cycle(self, full: bool = False) -> dict:
        if self.lock.locked():
            log.info("poll_skipped_previous_running")
            return {"skipped": True}
        async with self.lock:
            t0 = time.monotonic()
            self.cycle += 1
            result = SyncResult()
            tasks = [self._safe("docs", self.sync_docs(full, result))]
            tasks += [self._safe(r, self.sync_repo(r, full, result)) for r in self.s.repo_list]
            await asyncio.gather(*tasks)
            evicted = 0
            if full:
                evicted = self.cache.clear()
            elif result.any:
                evicted = self.cache.invalidate(result.changed, result.deleted, result.dense)
            self._last_cycle_wall = time.time()
            self.info.update(last_poll=now(), last_cycle_seconds=round(time.monotonic() - t0, 2), evicted_last_sync=evicted)
            if result.any:
                self.info["last_change_detected"] = now()
            log.info("poll_cycle_done", cycle=self.cycle, full=full, changed=len(result.changed),
                     deleted=len(result.deleted), evicted=evicted, seconds=self.info["last_cycle_seconds"])
            return {"changed": len(result.changed), "deleted": len(result.deleted), "evicted": evicted}

    async def _safe(self, name: str, coro) -> None:
        try:
            await coro
        except Exception as e:  # noqa: BLE001 - one source failing must not block the others
            log.exception("source_sync_failed", source=name, error=str(e))
            self.info["last_error"] = f"{name}: {str(e)[:300]}"

    # ---------- docs ----------
    async def sync_docs(self, full: bool, result: SyncResult) -> None:
        st = await self.state.get("docs")
        pages: dict[str, dict] = st.get("pages_map", {})
        sm = await fetch_sitemap(self.http, self.s, None if full else st.get("sitemap_etag"))
        checked = now()
        if sm.not_modified and not (self.cycle % 5 == 0):
            st["last_checked"] = checked
            await self.state.put("docs", st)
            self.info["sources"]["docs"] = {k: st.get(k) for k in ("pages", "chunks", "last_checked", "last_changed")}
            return
        if sm.not_modified:  # every 5th cycle re-check pages with no lastmod
            entries = {u: v.get("lastmod") for u, v in pages.items()}
        else:
            entries = sm.entries
        todo = [u for u, lm in entries.items()
                if full or u not in pages or lm is None or pages[u].get("lastmod") != lm]
        sem = asyncio.Semaphore(8)
        changed_any = False

        async def one(url: str) -> None:
            nonlocal changed_any
            async with sem:
                try:
                    html = await fetch_page(self.http, url)
                except Exception as e:  # noqa: BLE001
                    log.warning("docs_page_failed", url=url, error=str(e)[:200])
                    return
            page_hash, chunks = page_to_chunks(html, url, entries.get(url), self.s)
            pages[url] = {"lastmod": entries.get(url), "page_hash": page_hash, "chunks": len(chunks)}
            if full:  # batch all pages into one sync so embedding calls stay few and large
                all_chunks.extend(chunks)
                return
            if old_hashes.get(url) == page_hash:
                return
            r = await self.ix.sync_chunks(f_doc(url), chunks)
            result.add(r)
            if r[0] or r[1]:
                changed_any = True

        all_chunks: list[Chunk] = []
        old_hashes = {u: v.get("page_hash") for u, v in pages.items()}
        await asyncio.gather(*(one(u) for u in todo))
        if full:
            for st_name in ("docs", "blog"):
                result.add(await self.ix.sync_chunks(f_eq("source_type", st_name),
                                                     [c for c in all_chunks if c.source_type == st_name]))
        if not sm.not_modified:
            for gone in [u for u in pages if u not in entries]:
                result.deleted += await self.ix.delete_where(f_doc(gone))
                pages.pop(gone)
                changed_any = True
        st.update(pages_map=pages, pages=len(pages), chunks=sum(p.get("chunks", 0) for p in pages.values()),
                  last_checked=checked, sitemap_etag=sm.etag or st.get("sitemap_etag"))
        if changed_any or full:
            st["last_changed"] = checked
        await self.state.put("docs", st)
        self.info["sources"]["docs"] = {k: st.get(k) for k in ("pages", "chunks", "last_checked", "last_changed")}

    # ---------- repos ----------
    async def sync_repo(self, repo: str, full: bool, result: SyncResult) -> None:
        st = await self.state.get(f"repo:{repo}")
        checked = now()
        if not full and st.get("sha"):
            status, sha, etag, committed = await self.gh.head_commit(repo, st["branch"], st.get("etag"))
            if status == 304 or sha == st["sha"]:
                st.update(last_checked=checked, etag=etag or st.get("etag"))
                await self.state.put(f"repo:{repo}", st)
                self._repo_info(repo, st)
                return
            files = await self.gh.compare(repo, st["sha"], sha)
            if files is not None:
                for f in files:
                    path, status_f = f["filename"], f["status"]
                    if status_f == "renamed" and f.get("previous_filename"):
                        result.deleted += await self.ix.delete_where(f_repo_path(repo, f["previous_filename"]))
                    if status_f == "removed":
                        result.deleted += await self.ix.delete_where(f_repo_path(repo, path))
                        continue
                    if not is_included(repo, path, 0):
                        continue
                    text = await self.gh.raw(repo, sha, path)
                    chunks = file_to_chunks(repo, st["branch"], sha, path, text, self.s, committed) if text else []
                    result.add(await self.ix.sync_chunks(f_repo_path(repo, path), chunks))
                st.update(sha=sha, etag=etag, last_checked=checked, last_changed=checked,
                          chunks=await self.ix.count(f_eq("repo", repo)))
                await self.state.put(f"repo:{repo}", st)
                self._repo_info(repo, st)
                return
        # full path: fresh shallow clone and hash diff of every chunk
        dest, branch, sha = await clone_repo(repo, self.s)
        by_path = await asyncio.to_thread(walk_clone, repo, dest, branch, sha, self.s, None)
        chunks = [c for cs in by_path.values() for c in cs]
        result.add(await self.ix.sync_chunks(f_eq("repo", repo), chunks))
        try:
            _, _, etag, _ = await self.gh.head_commit(repo, branch, None)
        except httpx.HTTPError:
            etag = None
        st.update(branch=branch, sha=sha, etag=etag, files=len(by_path), chunks=len(chunks),
                  last_checked=checked, last_changed=checked)
        await self.state.put(f"repo:{repo}", st)
        self._repo_info(repo, st)

    def _repo_info(self, repo: str, st: dict) -> None:
        self.info["sources"][repo] = {"sha": (st.get("sha") or "")[:12], "branch": st.get("branch"),
                                      "files": st.get("files"), "chunks": st.get("chunks"),
                                      "last_checked": st.get("last_checked"), "last_changed": st.get("last_changed")}
