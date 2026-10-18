"""Chunker, cache invalidation, citation post-processing and the /ask contract."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.cache import AnswerCache, CachedAnswer
from app.ingest.chunker import Section, build_chunks, split_text
from app.llm.prompts import REFUSAL
from app.retrieval.pipeline import AskResult, Block, InvalidQuestion, postprocess


# ---------- chunker ----------
def test_split_never_breaks_code_fence():
    code = "```yaml\n" + "\n".join(f"key{i}: value{i}" for i in range(400)) + "\n```"
    text = "Intro paragraph.\n\n" + code + "\n\nOutro paragraph."
    for part in split_text(text):
        assert part.count("```") % 2 == 0


def test_chunk_ids_are_stable():
    secs = [Section(url="https://olake.io/docs/x/#a", heading_path=["X", "A"], text="word " * 200, h2="A")]
    a = build_chunks(secs, doc_url="https://olake.io/docs/x/", title="X", source_type="docs", component="ingestion", connector=None)
    b = build_chunks(secs, doc_url="https://olake.io/docs/x/", title="X", source_type="docs", component="ingestion", connector=None)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.content_hash for c in a] == [c.content_hash for c in b]


# ---------- cache ----------
def _entry(chunks, emb=None, refusal=False):
    return CachedAnswer(answer="a", sources=[], titles=[], chunk_ids=chunks, embedding=emb, is_refusal=refusal)


def test_cache_evicts_only_affected_answers():
    c = AnswerCache(100, 3600, 0.75)
    c.put("k1", _entry(["c1"], [1.0, 0.0]))
    c.put("k2", _entry(["c2"], [0.0, 1.0]))
    c.put("k3", _entry([], [0.0, 1.0], refusal=True))
    changed = [SimpleNamespace(chunk_id="c1")]
    evicted = c.invalidate(changed, [], {"c1": [1.0, 0.0]})
    assert evicted == 2  # k1 (used c1) and k3 (refusal); k2 untouched
    assert c.get("k2") is not None and c.get("k1") is None and c.get("k3") is None


def test_cache_no_changes_evicts_nothing():
    c = AnswerCache(100, 3600, 0.75)
    c.put("k1", _entry(["c1"], [1.0, 0.0]))
    assert c.invalidate([], [], {}) == 0


# ---------- post-processing ----------
BLOCKS = [Block(1, "https://olake.io/docs/a/#x", "A", ["A", "X"], "text", ["id1"], 0.9),
          Block(2, "https://olake.io/docs/b/", "B", ["B"], "text", ["id2"], 0.8)]


def test_sources_are_cited_blocks_in_first_cited_order():
    text, sources, _, refusal = postprocess("Use B [2]. Also A [1][2].", BLOCKS)
    assert sources == ["https://olake.io/docs/b/", "https://olake.io/docs/a/#x"]
    assert "[1]" in text and not refusal


def test_uncited_answer_cites_top_blocks():
    text, sources, _, refusal = postprocess("Some answer with no citations.", BLOCKS)
    assert not refusal and sources == ["https://olake.io/docs/a/#x", "https://olake.io/docs/b/"]


def test_empty_answer_becomes_refusal():
    text, sources, _, refusal = postprocess("", BLOCKS)
    assert refusal and sources == [] and text.startswith(REFUSAL)


def test_invented_links_are_removed():
    text, _, _, _ = postprocess("See [docs](https://evil.example.com/x) [1].", BLOCKS)
    assert "evil.example.com" not in text


def test_secret_leak_blocked():
    text, sources, _, refusal = postprocess("my key is sk-or-v1-abcdef [1]", BLOCKS)
    assert refusal and "sk-or" not in text and sources == []


# ---------- /ask contract ----------
class FakePipeline:
    async def ask(self, q):
        if q == "boom":
            raise RuntimeError("x")
        if not q.strip():
            raise InvalidQuestion("empty")
        return AskResult("ans [1]", ["https://olake.io/docs/a/"], model="m")


@pytest.fixture(scope="module")
def client():
    # one lifespan per module: the MCP session manager can only start once per process
    import app.main as main

    async def noop(self):
        self.ready = True

    from app.sync.poller import Poller

    mp = pytest.MonkeyPatch()
    mp.setattr(Poller, "start", noop)
    with TestClient(main.app) as c:
        main.app.state.pipeline = FakePipeline()
        main.app.state.poller.ready = True
        yield c
    mp.undo()


def test_ask_contract_success(client):
    r = client.post("/ask", json={"question": "hi", "extra": 1})
    assert r.status_code == 200 and set(r.json()) == {"answer", "sources"}
    assert r.headers["X-Model"] == "m"


@pytest.mark.parametrize("body,status", [({"question": "   "}, 422), ({}, 422), ({"question": "boom"}, 503)])
def test_ask_contract_errors(client, body, status):
    r = client.post("/ask", json=body)
    assert r.status_code == status
    assert set(r.json()) == {"answer", "sources"} and r.json()["sources"] == []


def test_version_reads_env(client, monkeypatch):
    monkeypatch.setenv("VERCEL_GIT_COMMIT_SHA", "abc123")
    monkeypatch.setenv("VERCEL_DEPLOYMENT_ID", "dpl_x")
    assert client.get("/version").json() == {"commit": "abc123", "deployment": "dpl_x"}


def test_refusal_citing_support_list_becomes_not_supported():
    blocks = BLOCKS + [Block(3, "https://olake.io/docs/#destinations", "Overview", ["Overview", "Destinations"], "t", ["id3"], 0.0)]
    text, sources, _, refusal = postprocess(REFUSAL + " OLake writes Iceberg and Parquet [3].", blocks,
                                            "Can OLake write to Delta Lake?")
    assert not refusal and text.startswith("That is not supported by OLake") and sources[0] == "https://olake.io/docs/#destinations"


def test_offtopic_refusal_is_not_rewritten():
    blocks = BLOCKS + [Block(3, "https://olake.io/docs/#destinations", "Overview", ["Overview", "Destinations"], "t", ["id3"], 0.0)]
    text, sources, _, refusal = postprocess(REFUSAL + " I can help with OLake instead [3].", blocks, "Write me a poem")
    assert refusal and sources == []


def test_not_supported_answer_lists_support_sections():
    blocks = BLOCKS + [Block(3, "https://olake.io/docs/#supported-sources", "Overview", ["Overview", "Supported Sources"], "t", ["id3"], 0.0)]
    _, sources, _, refusal = postprocess("Cassandra is not supported by OLake [1].", blocks, "Cassandra in OLake?")
    assert not refusal and "https://olake.io/docs/#supported-sources" in sources
