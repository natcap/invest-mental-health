"""
pipeline/stage_b_collection.py — Page & PDF Collection

Fetches HTML pages and PDFs from candidate URLs, extracts text content,
and caches results to avoid redundant downloads.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    FETCH_RATE_LIMIT,
    PAGE_CHAR_LIMIT,
    PDF_CHAR_LIMIT,
    REQUEST_TIMEOUT,
    USER_AGENT,
    log,
)
from pipeline.db import PipelineDB
from pipeline.models import SourceDocument

# Optional Chrome-TLS-impersonation fallback for sites that 403 plain requests
# (Cloudflare / Akamai WAFs).  We only use this when the regular stack fails,
# so the common path is unchanged.
try:
    from curl_cffi import requests as _curl_cffi_requests  # type: ignore
    _HAS_CURL_CFFI = True
except Exception:  # pragma: no cover — soft dependency
    _curl_cffi_requests = None
    _HAS_CURL_CFFI = False


# ── HTTP Session ──────────────────────────────────────────

_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
)


# ── HTML Fetching ─────────────────────────────────────────

# Split connect/read timeouts: fail fast when the host is unreachable,
# but still allow slow servers to send their response. `REQUEST_TIMEOUT`
# (from config) is the read timeout; connect timeout is capped at 6s
# so a dead host burns ~6s × 3 retries = 18s instead of 30 × 3 = 90s.
_CONNECT_TIMEOUT = 6
_FETCH_TIMEOUT = (_CONNECT_TIMEOUT, REQUEST_TIMEOUT)

# Status codes where the server actively rejected us (WAF / bot block).
# We won't retry these with plain requests — we jump straight to the
# curl_cffi (Chrome TLS fingerprint) fallback.
_WAF_BLOCK_STATUSES = {401, 403, 429}


class _CurlCffiResponse:
    """Minimal requests.Response-shaped wrapper for curl_cffi responses so the
    rest of the pipeline (which only reads .text and .status_code) works
    without changes."""

    __slots__ = ("status_code", "text", "content", "url", "_cc")

    def __init__(self, cc_resp):
        self._cc = cc_resp
        self.status_code = cc_resp.status_code
        self.text = cc_resp.text
        self.content = cc_resp.content
        self.url = str(cc_resp.url)

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(
                f"{self.status_code} returned by curl_cffi fallback", response=self  # type: ignore[arg-type]
            )


def _curl_cffi_get(url: str, *, is_pdf: bool = False):
    """Fetch with Chrome TLS impersonation. Returns a response-like object.

    Raises the underlying exception on failure so the caller can log it."""
    if not _HAS_CURL_CFFI:
        raise RuntimeError("curl_cffi not installed")
    read_timeout = REQUEST_TIMEOUT * (2 if is_pdf else 1)
    resp = _curl_cffi_requests.get(  # type: ignore[union-attr]
        url,
        timeout=read_timeout,
        impersonate="chrome124",
        allow_redirects=True,
    )
    return _CurlCffiResponse(resp)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_html_plain(url: str) -> requests.Response:
    """Plain `requests` fetch. Retries on transient errors. Callers that want a
    WAF fallback should use `_fetch_html` instead."""
    resp = _session.get(url, timeout=_FETCH_TIMEOUT, allow_redirects=True)
    # Don't burn 3 retries on a WAF block — let the caller fall back to
    # curl_cffi.  Raise a normal HTTPError so tenacity sees it only once.
    if resp.status_code in _WAF_BLOCK_STATUSES:
        raise requests.HTTPError(
            f"{resp.status_code} WAF-block (plain requests)", response=resp
        )
    resp.raise_for_status()
    return resp


def _fetch_html(url: str):
    """Fetch HTML, transparently falling back to curl_cffi Chrome-TLS
    impersonation when plain requests is blocked by a WAF (403/429) or
    times out.  Returns either a requests.Response or a _CurlCffiResponse
    (both expose .text / .status_code / raise_for_status())."""
    try:
        return _fetch_html_plain(url)
    except Exception as plain_err:
        if not _HAS_CURL_CFFI:
            raise
        # Only worth escalating for errors that look like a WAF block or
        # connection-level rejection.  For ordinary 404s, skip the fallback.
        msg = str(plain_err).lower()
        blocked_signal = (
            "403" in msg
            or "429" in msg
            or "401" in msg
            or "connectionerror" in msg
            or "sslerror" in msg
            or "timeout" in msg
            or "connectiontimeout" in msg
            or "readtimeout" in msg
            or "retryerror" in msg
        )
        if not blocked_signal:
            raise
        try:
            log.debug(f"[Collection] curl_cffi fallback for {url[:80]}: {plain_err}")
            resp = _curl_cffi_get(url)
            if resp.status_code >= 400:
                raise requests.HTTPError(
                    f"{resp.status_code} via curl_cffi", response=resp  # type: ignore[arg-type]
                )
            return resp
        except Exception as cc_err:
            # Re-raise the *original* plain error if the fallback also failed;
            # it usually has more actionable info.
            raise plain_err from cc_err


def _extract_html_text(html: str) -> str:
    """Parse HTML and return clean text content."""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup(["nav", "footer", "header", "script", "style", "aside",
                     "noscript", "iframe", "form"]):
        tag.decompose()

    # Try to find main content
    main = soup.find("main") or soup.find("article") or soup.find(
        "div", {"role": "main"}
    )
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Collapse blank lines
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)[:PAGE_CHAR_LIMIT]


def _extract_page_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""


# ── PDF Fetching ──────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_pdf_bytes_plain(url: str) -> bytes:
    # PDFs can be large — use a longer read timeout but still fail fast on connect.
    resp = _session.get(url, timeout=(_CONNECT_TIMEOUT, REQUEST_TIMEOUT * 2))
    if resp.status_code in _WAF_BLOCK_STATUSES:
        raise requests.HTTPError(
            f"{resp.status_code} WAF-block (plain requests)", response=resp
        )
    resp.raise_for_status()
    return resp.content


def _fetch_pdf_bytes(url: str) -> bytes:
    """PDF fetch with the same curl_cffi fallback policy as `_fetch_html`."""
    try:
        return _fetch_pdf_bytes_plain(url)
    except Exception as plain_err:
        if not _HAS_CURL_CFFI:
            raise
        msg = str(plain_err).lower()
        blocked_signal = (
            "403" in msg or "429" in msg or "401" in msg
            or "connectionerror" in msg or "sslerror" in msg
            or "timeout" in msg or "retryerror" in msg
        )
        if not blocked_signal:
            raise
        try:
            log.debug(f"[Collection] curl_cffi PDF fallback for {url[:80]}")
            resp = _curl_cffi_get(url, is_pdf=True)
            if resp.status_code >= 400:
                raise requests.HTTPError(
                    f"{resp.status_code} via curl_cffi", response=resp  # type: ignore[arg-type]
                )
            return resp.content
        except Exception as cc_err:
            raise plain_err from cc_err


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    import pdfplumber

    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_bytes)
        f.flush()
        pages_text = []
        with pdfplumber.open(f.name) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
    return "\n".join(pages_text)[:PDF_CHAR_LIMIT]


# ── PDF Link Discovery ───────────────────────────────────


def _find_pdf_links(html: str, base_url: str) -> List[str]:
    """Find PDF links in an HTML page."""
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "lxml")
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            full_url = urljoin(base_url, href)
            pdf_links.append(full_url)
    return list(dict.fromkeys(pdf_links))[:20]  # cap at 20


# ── Content-Type Detection ───────────────────────────────


def _is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


# ── URL validity filter ──────────────────────────────────

# Patterns that indicate the LLM left a placeholder in the URL it suggested
# (e.g. `/DocumentCenter/View/???/Tree-Master-Plan`, `?id=<ID>`, `{slug}`).
# These URLs will never resolve, so skip them before paying the network cost.
import re as _re

_PLACEHOLDER_MARKERS = (
    "???",
    "<id>", "<ID>", "<slug>", "<SLUG>",
    "{id}", "{slug}", "{year}", "{doc_id}", "{number}",
    "[id]", "[slug]",
    "XXXXX", "xxxxx", "YYYYY", "yyyyy", "NNNNN", "nnnnn",
    "PLACEHOLDER", "placeholder",
)

# Runs of ? or _ that indicate the LLM filled in a blank.
# 3+ consecutive `?` characters, or 4+ consecutive `_` inside a path segment,
# are almost never legitimate (URLs don't normally use `____` as a slug).
_PLACEHOLDER_RE = _re.compile(r"\?{3,}|_{4,}")


def _has_placeholder(url: str) -> bool:
    if any(m in url for m in _PLACEHOLDER_MARKERS):
        return True
    return bool(_PLACEHOLDER_RE.search(url))


# ── Non-text media filter ────────────────────────────────
#
# We can't extract useful text from audio/video/archive/image files, and many
# city DocumentCenter URLs encode the media type in the filename slug
# (e.g. `.../View/123/02-11-2020-City-Council-Meeting-MP3`).  Skip them before
# paying the network + pdfplumber cost.

_BINARY_EXTS = (
    # audio
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma",
    # video
    ".mp4", ".mov", ".avi", ".wmv", ".flv", ".webm", ".mkv", ".m4v",
    # image
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".tiff", ".tif",
    ".ico", ".webp", ".heic",
    # archive
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".tgz",
    # CAD / GIS / misc binary
    ".dwg", ".dxf", ".shp", ".kmz", ".dmg", ".exe", ".iso",
)

# Slug suffixes like `-MP3`, `-Video`, `-Audio-Recording` (CivicEngage pattern
# where the filename ends with the file-type label).  Tested against the final
# path segment lowercased.
_MEDIA_SUFFIX_RE = _re.compile(
    r"(?:^|[-_])(?:mp3|mp4|wav|m4a|aac|wmv|mov|avi|video|audio|recording|"
    r"podcast|webcast|mpeg|mpg)$"
)


def _is_non_text_media(url: str) -> bool:
    """True if the URL is obviously audio / video / image / archive — i.e.
    nothing our HTML or PDF extractors can turn into useful text."""
    path = urlparse(url).path.lower()
    if path.endswith(_BINARY_EXTS):
        return True
    # Last path segment: check for `-mp3`, `-video`, `-audio-recording`, etc.
    last = path.rsplit("/", 1)[-1] if "/" in path else path
    # Strip trailing slash/empty
    if not last:
        return False
    return bool(_MEDIA_SUFFIX_RE.search(last))


# ── 404 / Error-Page Detection ───────────────────────────

_ERROR_TITLE_MARKERS = (
    "page not found",
    "404",
    "not found",
    "page cannot be found",
    "error page",
    "404 error",
)


def _looks_like_error_page(title: str, text: str) -> bool:
    """Heuristic check: does a successfully-fetched HTML page actually contain
    a 404 / error-page soft response?  (Some city sites return 200 OK with a
    404 template.)"""
    tl = (title or "").lower()
    if any(m in tl for m in _ERROR_TITLE_MARKERS):
        return True
    # Very short pages with common 404 phrases in body
    if text and len(text) < 800:
        bl = text[:800].lower()
        if ("page not found" in bl or "404" in bl) and "sorry" in bl:
            return True
    return False


# ── Public API ────────────────────────────────────────────


def fetch_source(
    url: str,
    city: str,
    state: str,
    db: PipelineDB,
    is_official: bool = False,
) -> Optional[SourceDocument]:
    """
    Fetch a single URL (HTML or PDF), extract text, cache in DB.

    Returns SourceDocument or None on failure.
    """
    # Check cache first — but ignore cached failures and previously-cached
    # 404 soft-error pages (these were polluting extraction from an earlier run).
    cached = db.get_cached_source(url)
    if cached and cached.get("text_content"):
        cached_status = cached.get("http_status") or 0
        cached_ct = cached.get("content_type") or ""
        cached_title = cached.get("title") or ""
        cached_text = cached.get("text_content") or ""
        is_error = (
            cached_ct == "error"
            or cached_status == 0
            or cached_status >= 400
            or _looks_like_error_page(cached_title, cached_text)
        )
        if not is_error:
            log.debug(f"[Collection] Cache hit: {url[:80]}")
            return SourceDocument(
                url=url,
                title=cached_title,
                content_type=cached["content_type"],
                text=cached_text,
                fetch_timestamp=cached.get("fetch_timestamp", ""),
                http_status=cached_status or 200,
                is_official=bool(cached.get("is_official", False)),
            )
        # Stale bad cache — fall through and re-fetch (won't be cached again
        # as a success unless the URL now resolves properly).
        log.debug(f"[Collection] Ignoring stale error cache: {url[:80]}")

    log.info(f"[Collection] Fetching: {url[:80]}")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        if _is_pdf_url(url):
            pdf_bytes = _fetch_pdf_bytes(url)
            text = _extract_pdf_text(pdf_bytes)
            doc = SourceDocument(
                url=url,
                title=urlparse(url).path.split("/")[-1],
                content_type="pdf",
                text=text,
                fetch_timestamp=now,
                http_status=200,
                is_official=is_official,
            )
        else:
            resp = _fetch_html(url)
            text = _extract_html_text(resp.text)
            title = _extract_page_title(resp.text)

            # Guard against soft-404s: some city CMSes return HTTP 200 with a
            # "Page Not Found" template.  Drop these so we don't pay the LLM
            # to extract zero records from them.
            if _looks_like_error_page(title, text):
                log.info(f"[Collection] Soft-404 detected, skipping: {url[:80]}")
                db.cache_source(
                    url=url,
                    city=city,
                    state=state,
                    content_type="error",
                    text_content="",
                    title=title,
                    http_status=404,
                    is_official=is_official,
                )
                time.sleep(1.0 / max(FETCH_RATE_LIMIT, 1))
                return None

            doc = SourceDocument(
                url=url,
                title=title,
                content_type="html",
                text=text,
                fetch_timestamp=now,
                http_status=resp.status_code,
                is_official=is_official,
            )

        # Cache in DB
        db.cache_source(
            url=url,
            city=city,
            state=state,
            content_type=doc.content_type,
            text_content=doc.text,
            title=doc.title,
            http_status=doc.http_status,
            is_official=is_official,
        )

        # Rate limiting
        time.sleep(1.0 / max(FETCH_RATE_LIMIT, 1))
        return doc

    except Exception as e:
        log.warning(f"[Collection] Failed to fetch {url[:80]}: {e}")
        # Cache the failure so we don't retry endlessly
        db.cache_source(
            url=url,
            city=city,
            state=state,
            content_type="error",
            text_content="",
            title="",
            http_status=0,
            is_official=is_official,
        )
        return None


def collect_sources(
    city: str,
    state: str,
    urls: List[str],
    db: PipelineDB,
    official_domains: List[str] | None = None,
) -> List[SourceDocument]:
    """
    Fetch all candidate URLs for a city and return successfully-fetched documents.

    Also discovers PDF links within fetched HTML pages and fetches those.
    """
    official_domains = official_domains or []
    documents: List[SourceDocument] = []

    # ── Per-host circuit breaker ────────────────────────────
    # If a host fails `_DEAD_HOST_THRESHOLD` times in a row we mark it dead and
    # skip every remaining URL on that host.  This prevents one unreachable
    # domain (e.g. paramountcity.com ConnectTimeout) from eating hours when the
    # LLM over-suggested 80+ URLs on it.
    _DEAD_HOST_THRESHOLD = 3
    host_fail_streak: dict[str, int] = {}
    dead_hosts: set[str] = set()

    skipped_placeholders = 0
    skipped_media = 0
    skipped_dead_host = 0
    for url in urls:
        if _has_placeholder(url):
            skipped_placeholders += 1
            continue
        if _is_non_text_media(url):
            skipped_media += 1
            log.debug(f"[Collection] Skipping non-text media: {url[:100]}")
            continue

        host = urlparse(url).netloc.lower()
        if host in dead_hosts:
            skipped_dead_host += 1
            continue

        is_official = any(d in url for d in official_domains)
        doc = fetch_source(url, city, state, db, is_official=is_official)

        if doc and doc.text.strip():
            host_fail_streak[host] = 0
            documents.append(doc)
        else:
            host_fail_streak[host] = host_fail_streak.get(host, 0) + 1
            if host_fail_streak[host] >= _DEAD_HOST_THRESHOLD:
                dead_hosts.add(host)
                log.warning(
                    f"[Collection] {city}, {state} — host {host} marked "
                    f"dead after {host_fail_streak[host]} consecutive failures; "
                    f"skipping remaining URLs on this host"
                )

    if skipped_placeholders:
        log.info(
            f"[Collection] {city}, {state} — skipped {skipped_placeholders} "
            f"URL(s) containing LLM placeholders"
        )
    if skipped_media:
        log.info(
            f"[Collection] {city}, {state} — skipped {skipped_media} "
            f"non-text media URL(s) (audio/video/image/archive)"
        )
    if skipped_dead_host:
        log.info(
            f"[Collection] {city}, {state} — skipped {skipped_dead_host} "
            f"URL(s) on unreachable hosts: {sorted(dead_hosts)}"
        )

    log.info(
        f"[Collection] {city}, {state} — collected {len(documents)} documents "
        f"from {len(urls)} URLs"
    )
    return documents
