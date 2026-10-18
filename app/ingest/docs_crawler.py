"""olake.io docs: sitemap fetch (ETag aware), page fetch, rendered HTML -> heading sections."""
import asyncio
import hashlib
import re
from dataclasses import dataclass

import httpx
import structlog
from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify

from app.config import Settings
from app.ingest.chunker import Chunk, Section, build_chunks, count_tokens
from app.ingest.classify import classify

log = structlog.get_logger()
UA = {"User-Agent": "AskOLakeBot/1.0 (+https://github.com/datazip-inc)"}
ZW = re.compile(r"[​‌‍﻿]")


@dataclass
class SitemapResult:
    not_modified: bool
    etag: str | None
    entries: dict[str, str | None]  # url -> lastmod
    body_hash: str | None = None


async def fetch_sitemap(client: httpx.AsyncClient, s: Settings, etag: str | None = None) -> SitemapResult:
    headers = dict(UA)
    if etag:
        headers["If-None-Match"] = etag
    r = await client.get(s.docs_sitemap_url, headers=headers, timeout=15)
    if r.status_code == 304:
        return SitemapResult(True, etag, {})
    r.raise_for_status()
    entries: dict[str, str | None] = {}
    for m in re.finditer(r"<url>(.*?)</url>", r.text, re.S):
        block = m.group(1)
        loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block)
        if not loc or not s.wants_url(loc.group(1)):
            continue
        lm = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block)
        entries[loc.group(1)] = lm.group(1) if lm else None
    return SitemapResult(False, r.headers.get("etag"), entries, hashlib.sha256(r.content).hexdigest())


async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = await client.get(url, headers=UA, timeout=15, follow_redirects=True)
            r.raise_for_status()
            return r.text
        except Exception as e:  # noqa: BLE001
            last = e
            await asyncio.sleep(0.5 * 2**attempt)
    raise RuntimeError(f"fetch failed {url}: {last}")


def _clean(t: str) -> str:
    return ZW.sub("", t).strip()


def parse_docs_html(html: str, url: str, base_url: str) -> tuple[str, str, list[Section]]:
    """Return (title, main_text_hash, sections). Splits on every h2/h3, including ones inside tabs."""
    soup = BeautifulSoup(html, "lxml")
    # docs pages render into .theme-doc-markdown, blog posts into article .markdown
    md = soup.select_one(".theme-doc-markdown") or soup.select_one("article .markdown")
    if md is None:
        return url, "", []
    crumbs = [_clean(li.get_text(" ", strip=True)) for li in soup.select(".theme-doc-breadcrumbs li")]
    crumbs = [c for c in crumbs if c]
    h1 = md.select_one("h1") or soup.select_one("article header h1") or soup.select_one("h1")
    title = crumbs[-1] if crumbs else (_clean(h1.get_text(" ", strip=True)) if h1 else url)
    if "/blog/" in url:
        crumbs = ["Blog", title]
    for sel in ["a.hash-link", "button", "script", "style", "svg", "img", "nav", "iframe", "h1", "form"]:
        for el in md.select(sel):
            el.decompose()
    for a in md.select("a[href]"):
        if a["href"].startswith("/"):
            a["href"] = base_url + a["href"]
    for tabs in md.select(".tabs-container"):
        labels = [_clean(li.get_text(" ", strip=True)) for li in tabs.select("[role=tab]")]
        for i, panel in enumerate(tabs.select("[role=tabpanel]")):
            if i < len(labels):
                panel.insert(0, NavigableString(f"\n\nQQTAB{labels[i]}QQEND\n\n"))
        for ul in tabs.select("[role=tablist]"):
            ul.decompose()
    codes: list[tuple[str, str]] = []
    for pre in md.select("pre"):
        holder = pre if any(c.startswith("language-") for c in pre.get("class", [])) else pre.find_parent(class_=re.compile(r"language-"))
        m = re.search(r"language-(\S+)", " ".join(holder.get("class", []))) if holder else None
        lang = m.group(1) if m else ""
        lines = pre.select(".token-line")
        # Prism renders one div per line; get_text() alone would glue lines together
        code = "\n".join(ln.get_text() for ln in lines) if lines else pre.get_text()
        codes.append((lang, code.rstrip()))
        pre.replace_with(NavigableString(f"QQPRE{len(codes) - 1}QQEND"))
    for adm in md.select(".theme-admonition"):
        kind = next((c.split("-")[-1] for c in adm.get("class", []) if c.startswith("theme-admonition-")), "note")
        adm.insert(0, NavigableString(f"QQADM{kind.upper()}QQEND "))
    last_id = None
    for h in md.find_all(["h2", "h3"]):
        last_id = h.get("id") or last_id
        h.replace_with(NavigableString(f"\n\nQQH{h.name[1]}|{last_id or ''}|{_clean(h.get_text(' ', strip=True))}QQEND\n\n"))
    text = markdownify(str(md), heading_style="ATX", bullets="-", escape_underscores=False, escape_asterisks=False)
    text = re.sub(r"QQPRE(\d+)QQEND", lambda m: "\n\n```%s\n%s\n```\n\n" % codes[int(m.group(1))], text)
    text = re.sub(r"QQTAB(.*?)QQEND", r"**[Tab: \1]**", text)
    text = re.sub(r"QQADM(.*?)QQEND", r"**\1:**", text)
    text = re.sub(r"<!--.*?-->", "", ZW.sub("", text), flags=re.S)
    page_hash = hashlib.sha256(text.encode()).hexdigest()
    parts = re.split(r"QQH([23])\|([^|]*)\|(.*?)QQEND", text)
    base = crumbs or [title]
    sections: list[Section] = []
    h2 = None

    def add(anchor: str | None, path: list[str], body: str, h2_name: str | None) -> None:
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if body:
            sections.append(Section(url=url + (f"#{anchor}" if anchor else ""), heading_path=path, text=body, h2=h2_name))

    add(None, base, parts[0], None)
    for i in range(1, len(parts), 4):
        lvl, anchor, head, body = parts[i], parts[i + 1], parts[i + 2].replace("\\", ""), parts[i + 3]
        if lvl == "2":
            h2 = head
            path = base + [head]
        else:
            path = base + ([h2] if h2 else []) + [head]
        add(anchor or None, path, body, h2)
    return title, page_hash, sections


def page_to_chunks(html: str, url: str, lastmod: str | None, s: Settings) -> tuple[str, list[Chunk]]:
    title, page_hash, sections = parse_docs_html(html, url, s.docs_base_url)
    if sum(count_tokens(x.text) for x in sections) < 50:
        return page_hash, []
    component, connector = classify(url)
    source_type = "blog" if "/blog/" in url else "docs"
    chunks = build_chunks(sections, doc_url=url, title=title, source_type=source_type, component=component,
                          connector=connector, last_updated=lastmod)
    return page_hash, chunks
