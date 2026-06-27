"""
pipeline/stage_a_discovery.py — Source Discovery

For each city, generates search queries and finds candidate URLs from:
  1. Web search API (SerpAPI / Google Custom Search)  
  2. LLM-assisted URL suggestion  
  3. Known domain templates  

Outputs a list of official + secondary candidate URLs per city.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import requests
from requests.exceptions import HTTPError, RequestException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import (
    SERP_API_KEY,
    serp_key_mgr,
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SEARCH_RATE_LIMIT,
    USER_AGENT,
    log,
)
from pipeline.models import DiscoveryResponse


# ── Search Query Templates ────────────────────────────────

OFFICIAL_QUERY_TEMPLATES = [
    'site:{domain} "tree canopy"',
    'site:{domain} "urban forest" target OR goal',
    'site:{domain} "green infrastructure" plan',
    'site:{domain} filetype:pdf "climate action plan" trees',
    'site:{domain} filetype:pdf "urban forest" plan',
    'site:{domain} "sustainability" trees canopy',
    'site:{domain} "tree planting" goal',
]

SECONDARY_QUERY_TEMPLATES = [
    '"{city}" "{state}" "tree canopy" goal OR target',
    '"{city}" "urban forest master plan"',
    '"{city}" "greening target" OR "canopy goal"',
    '"{city}" "tree planting" goal USDA OR "Arbor Day"',
    '"{city}" "climate action plan" trees canopy',
]

STATE_FALLBACK_TEMPLATES = [
    '"{state}" state "urban forest" statewide goal',
    '"{state}" DNR "tree canopy" target',
    '"{state}" "urban forestry" state plan',
]

UNOFFICIAL_QUERY_TEMPLATES = [
    '"{city}" "tree canopy" goal OR target site:americanforests.org OR site:treeequityscore.org OR site:curbcanopy.com',
    '"{city}" "{state}" "urban forest" plan OR goal site:arborday.org',
    '"{city}" "tree canopy" OR "urban forest" goal news OR report -site:{domain}',
    '"{city}" "{state}" "tree planting" initiative OR campaign OR partnership',
    '"{city}" "tree equity" score OR goal OR canopy',
    '"{city}" "{state}" "tree canopy" nonprofit OR foundation OR coalition',
]


def _build_queries(
    city: str, state: str, domains: List[str], phase: str = "official"
) -> List[str]:
    """Generate search queries for a given phase."""
    queries: List[str] = []
    if phase == "official":
        for domain in domains:
            for tpl in OFFICIAL_QUERY_TEMPLATES:
                queries.append(tpl.format(domain=domain, city=city, state=state))
    elif phase == "secondary":
        for tpl in SECONDARY_QUERY_TEMPLATES:
            queries.append(tpl.format(city=city, state=state))
    elif phase == "state":
        for tpl in STATE_FALLBACK_TEMPLATES:
            queries.append(tpl.format(city=city, state=state))
    elif phase == "unofficial":
        primary_domain = domains[0] if domains else ""
        for tpl in UNOFFICIAL_QUERY_TEMPLATES:
            queries.append(tpl.format(city=city, state=state, domain=primary_domain))
    return queries


# ── SerpAPI Search ────────────────────────────────────────

# Status codes that indicate the current SerpAPI key is bad / exhausted and
# rotation should be attempted (not a transient error).
_KEY_ROTATE_STATUS = {401, 402, 403, 429}


def _rotate_key(api_key: str, reason: str) -> Optional[str]:
    """Rotate to the next SerpAPI key. Returns the new key or None if exhausted."""
    old_tail = api_key[-6:] if api_key else "??????"
    next_key = serp_key_mgr.rotate()
    if next_key:
        log.warning(
            f"SerpAPI key ...{old_tail} {reason} — "
            f"rotating to key ...{next_key[-6:]} "
            f"({serp_key_mgr.remaining} keys remaining)"
        )
        return next_key
    log.error(f"All SerpAPI keys exhausted ({reason}) — search disabled for remainder of run")
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type((RequestException,)),
    reraise=True,
)
def _serp_request(query: str, api_key: str, num_results: int) -> requests.Response:
    """Low-level SerpAPI request with transient-error retry (network only)."""
    return requests.get(
        "https://serpapi.com/search",
        params={
            "q": query,
            "api_key": api_key,
            "engine": "google",
            "num": num_results,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )


def _serp_search(query: str, num_results: int = 10) -> List[Dict]:
    """
    Execute a single search via SerpAPI with automatic key rotation.

    Rotation triggers on:
      • HTTP 401 / 402 / 403 / 429 (invalid or quota-exhausted key)
      • HTTP 200 with JSON body containing quota/limit/account error
    Transient network errors are retried up to 3 times with the same key.
    """
    if not serp_key_mgr:
        log.debug("No SERP_API_KEY set — skipping web search")
        return []

    # Try up to N = len(keys) different keys for this single query
    max_key_attempts = max(len(serp_key_mgr._keys), 1)

    for _ in range(max_key_attempts):
        api_key = serp_key_mgr.current_key
        if not api_key:
            return []

        try:
            resp = _serp_request(query, api_key, num_results)
        except RequestException as e:
            # Transient network failure — retried 3x already; give up on this query
            log.warning(f"SerpAPI network error for query '{query[:60]}...': {e}")
            return []

        # Key-level failure → rotate and retry the same query with the next key
        if resp.status_code in _KEY_ROTATE_STATUS:
            if _rotate_key(api_key, f"HTTP {resp.status_code}") is None:
                return []
            continue

        # Other non-success HTTP status — not a key problem, don't rotate
        if resp.status_code >= 400:
            log.warning(
                f"SerpAPI returned HTTP {resp.status_code} for query '{query[:60]}...'"
            )
            return []

        # 200 OK — check JSON body for account-level error
        try:
            data = resp.json()
        except ValueError:
            log.warning(f"SerpAPI returned non-JSON for query '{query[:60]}...'")
            return []

        err = data.get("error")
        if err:
            err_l = str(err).lower()
            if any(k in err_l for k in ("limit", "quota", "exceed", "account", "invalid api key")):
                if _rotate_key(api_key, f"error: {err}") is None:
                    return []
                continue
            # Non-key error (e.g. bad query) — skip this query
            log.warning(f"SerpAPI error for query '{query[:60]}...': {err}")
            return []

        return data.get("organic_results", [])

    log.error("All SerpAPI keys exhausted — could not complete search")
    return []


def _extract_urls_from_serp(results: List[Dict]) -> List[str]:
    """Pull unique URLs from SerpAPI organic results."""
    seen = set()
    urls = []
    for r in results:
        link = r.get("link", "")
        if link and link not in seen:
            seen.add(link)
            urls.append(link)
    return urls


# ── LLM-assisted discovery (optional enrichment) ─────────

def _llm_discover(
    city: str, state: str, domain_candidates: List[str]
) -> DiscoveryResponse:
    """Ask the LLM to suggest likely official URLs for a city."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        return DiscoveryResponse()

    from config import OPENAI_BASE_URL  # noqa: PLC0415
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    prompt = f"""You are a municipal policy research assistant.

Task: Identify official city-government sources for urban greening targets.

City: {city}
State: {state}
Official domain candidates: {', '.join(domain_candidates)}

Return ONLY likely relevant source URLs from:
- .gov official city websites
- Adopted city plans (sustainability, climate, urban forestry, comprehensive)
- City PDFs (plans, reports, assessments)
- City council resolutions or minutes
- Official city dashboards
- Official mayor/city press releases

Prioritize sources likely to contain:
- Tree canopy cover goals
- Tree planting targets
- Urban forest management goals
- Green infrastructure targets

Return JSON with keys: official_candidate_urls (list), secondary_candidate_urls (list), notes (string).
"""

    # 4000 tokens ≈ 16 000 chars — enough for ~60+ URLs.
    # Retry once on JSON parse errors (the most common failure mode is the
    # model running out of budget mid-string and returning truncated JSON).
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
                max_completion_tokens=4000,
            )
            data = json.loads(response.choices[0].message.content)
            return DiscoveryResponse(**data)
        except json.JSONDecodeError as e:
            last_exc = e
            log.warning(
                f"LLM discovery JSON parse failed for {city}, {state} "
                f"(attempt {attempt}/2): {e}"
            )
            continue
        except Exception as e:
            log.warning(f"LLM discovery failed for {city}, {state}: {e}")
            return DiscoveryResponse()

    log.warning(
        f"LLM discovery gave up for {city}, {state} after 2 attempts: {last_exc}"
    )
    return DiscoveryResponse()


# ── Known-Domain URL Generation ───────────────────────────

def _generate_known_urls(city: str, state: str, domains: List[str]) -> List[str]:
    """Generate common government page paths to try directly."""
    paths = [
        "/sustainability",
        "/environment",
        "/trees",
        "/urban-forestry",
        "/parks",
        "/climate",
        "/climate-action-plan",
        "/green-infrastructure",
    ]
    urls = []
    for domain in domains:
        for path in paths:
            urls.append(f"https://www.{domain}{path}")
            urls.append(f"https://{domain}{path}")
    return urls


# ── Public API ────────────────────────────────────────────


def discover_sources(
    city: str,
    state: str,
    domains: List[str],
    use_llm: bool = True,
    phase: str = "official",
) -> Dict[str, List[str]]:
    """
    Run source discovery for a single city.

    Returns:
        {"official": [url, ...], "secondary": [url, ...]}
    """
    log.info(f"[Discovery] {city}, {state} — phase={phase}")

    official_urls: List[str] = []
    secondary_urls: List[str] = []

    # 1. Web search (SerpAPI) — skip entirely if no keys remain
    queries = _build_queries(city, state, domains, phase=phase)
    if serp_key_mgr and serp_key_mgr.current_key:
        for q in queries:
            # Bail out of the inner loop as soon as the key pool is drained
            if not serp_key_mgr.current_key:
                log.debug("SerpAPI key pool drained — skipping remaining queries")
                break
            try:
                results = _serp_search(q, num_results=5)
                urls = _extract_urls_from_serp(results)
                for u in urls:
                    if any(d in u for d in domains):
                        official_urls.append(u)
                    else:
                        secondary_urls.append(u)
                time.sleep(1.0 / max(SEARCH_RATE_LIMIT, 1))
            except Exception as e:
                log.warning(f"Search failed for query '{q[:60]}...': {e}")
    else:
        log.debug(f"[Discovery] {city}, {state} — SerpAPI unavailable, using LLM + known URLs only")

    # 2. Known domain URLs
    if phase == "official":
        official_urls.extend(_generate_known_urls(city, state, domains))

    # 3. LLM-assisted discovery (optional)
    if use_llm and phase in ("official", "secondary", "unofficial"):
        llm_resp = _llm_discover(city, state, domains)
        official_urls.extend(llm_resp.official_candidate_urls)
        secondary_urls.extend(llm_resp.secondary_candidate_urls)

    # Deduplicate
    official_urls = list(dict.fromkeys(official_urls))
    secondary_urls = list(dict.fromkeys(u for u in secondary_urls if u not in set(official_urls)))

    log.info(
        f"[Discovery] {city}, {state} — {len(official_urls)} official, "
        f"{len(secondary_urls)} secondary URLs"
    )

    return {"official": official_urls, "secondary": secondary_urls}
