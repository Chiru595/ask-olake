"""Rule-based component / connector tagging (no LLM)."""
import re

CONNECTORS = {
    "postgres": ["postgres", "postgresql", "pgoutput", "wal2json"],
    "mysql": ["mysql", "binlog", "aurora mysql", "mariadb"],
    "mongodb": ["mongodb", "mongo", "oplog"],
    "oracle": ["oracle"],
    "mssql": ["mssql", "sql server", "sqlserver"],
    "db2": ["db2"],
    "kafka": ["kafka"],
    "s3": ["s3 source", "/connectors/s3", "drivers/s3"],
}

COMPONENT_HINTS = {
    "fusion": ["fusion", "compaction", "optimizer", "table maintenance"],
    "helm": ["helm", "kubernetes", "k8s"],
    "ui": ["olake-ui", "olake ui"],
}


def classify(url: str, repo: str | None = None, path: str | None = None) -> tuple[str, str | None]:
    """Return (component, connector) from URL/repo/path only, so tags are stable across edits."""
    key = f"{url} {repo or ''} {path or ''}".lower()
    if repo == "olake-fusion" or "/docs/fusion/" in key:
        component = "fusion"
    elif repo == "olake-helm" or "/install/kubernetes" in key:
        component = "helm"
    elif repo == "olake-ui" or "/install/olake-ui" in key or "olake-ui-api" in key:
        component = "ui"
    elif repo == "olake" or "/docs/" in key:
        component = "ingestion"
    else:
        component = "general"
    connector = None
    m = re.search(r"/(?:connectors|drivers)/([a-z0-9]+)", key)
    if m and m.group(1) in CONNECTORS:
        connector = m.group(1)
    return component, connector


def detect_query_filters(question: str) -> tuple[str | None, str | None]:
    """Detect a connector / component named in the question for a soft retrieval boost."""
    q = question.lower()
    connector = next((c for c, words in CONNECTORS.items() if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)), None)
    component = next((c for c, words in COMPONENT_HINTS.items() if any(w in q for w in words)), None)
    return component, connector
