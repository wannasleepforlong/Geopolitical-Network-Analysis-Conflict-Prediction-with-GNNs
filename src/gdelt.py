from __future__ import annotations

import io
import re
import time
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

import logging

try:
    from ..utils.logger import get_logger
    from ..utils.llm_client import LLMClient
    logger = get_logger("mirofish.gdelt")
except ImportError:
    # Fallback when utils package does not exist (new architecture)
    logger = logging.getLogger("mirofish.gdelt")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    class LLMClient:
        """Stub LLM client — GDELTEventFeed will degrade gracefully."""
        def chat_json(self, *args, **kwargs):
            logger.warning("LLMClient stub: no LLM available")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitise_gdelt_query(raw: str) -> str:
    """
    Clean a free-text query so it is safe for the GDELT Doc API fulltext search.

    GDELT's parser rules (simplified):
      - Parentheses are only allowed wrapping OR'd sub-expressions.
        Bare parentheses around a plain phrase cause a 400-like HTML response.
      - Bare boolean operators (AND / OR / NOT) outside of a recognised
        compound expression confuse the parser.
      - Colons and special chars have no meaning in fulltext mode.

    Strategy: keep only alphanumeric chars, spaces, and hyphens; collapse
    runs of whitespace; strip leading/trailing whitespace.
    """
    # Remove characters the GDELT parser chokes on
    cleaned = re.sub(r'[^\w\s\-]', ' ', raw, flags=re.UNICODE)
    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Remove bare AND / OR / NOT that could trip the parser
    cleaned = re.sub(r'\b(AND|OR|NOT)\b', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _backoff_delay(attempt: int, base: float = 2.0, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter."""
    delay = min(base ** attempt, cap)
    return delay * (0.5 + random.random() * 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Existing thin wrapper (kept 100 % compatible with /api/news/live)
# ─────────────────────────────────────────────────────────────────────────────

class GDELTFetcher:
    """
    Lightweight GDELT Doc API v2 wrapper.
    Used by the existing /api/news/live endpoint – do NOT change its public API.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    _cache: Dict[Tuple, Tuple[float, List]] = {}
    _CACHE_TTL = 60  # seconds

    # Maximum retries for a single API call
    _MAX_RETRIES = 3
    # Timeout for a single GDELT request (seconds)
    _REQUEST_TIMEOUT = 30

    # ── GDELT theme presets ──────────────────────────────────────────────────
    # These are passed as the `theme` query-param — NOT appended to the query
    # string — so they are valid GDELT theme identifiers.
    THEME_PRESETS: Dict[str, str] = {
        "conflict":    "MILITARY",
        "diplomacy":   "DIPLOMACY",
        "economy":     "ECON_BANKRUPTCY",
        "environment": "ENV_CLIMATECHANGE",
        "protest":     "PROTEST",
        "health":      "HEALTH_PANDEMIC",
        "technology":  "WB_2538_SCIENCE_TECHNOLOGY_AND_INNOVATION",
        "general":     "",   # no theme filter
    }

    @classmethod
    def fetch_war_news(cls, countries: List[str], max_rows: int = 50) -> List[Dict[str, Any]]:
        """Fetch conflict-related news for specific countries. (legacy, unchanged)"""
        if not countries:
            return []

        cache_key = tuple(sorted(countries))
        if cache_key in cls._cache:
            ts, data = cls._cache[cache_key]
            if time.time() - ts < cls._CACHE_TTL:
                logger.info(f"Returning cached news for {countries}")
                return data

        # Build a safe query: "war Ukraine" or "war Ukraine OR Russia"
        sanitised = [_sanitise_gdelt_query(c) for c in countries if c]
        if len(sanitised) > 1:
            countries_part = " OR ".join(sanitised)
            query = f"war {countries_part}"
        else:
            query = f"war {sanitised[0]}" if sanitised else "war"

        return cls._query_doc_api(query, max_rows, cache_key)

    @classmethod
    def fetch_by_queries(
        cls,
        queries: List[str],
        max_rows: int = 20,
        theme: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch GDELT articles for a list of free-text queries.
        Optionally restrict to a GDELT THEMES category.

        Args:
            queries:  1-5 search strings (LLM-generated or manual)
            max_rows: articles per query (de-duped across queries)
            theme:    optional preset key from THEME_PRESETS
                      or a raw GDELT theme string like "MILITARY"

        Returns:
            De-duplicated list of article dicts.

        IMPORTANT: theme is passed as a separate `theme` query-param to GDELT,
        NOT concatenated into the query string. This avoids the
        "Parentheses may only be used around OR'd statements" error.
        """
        # Resolve theme → GDELT theme identifier
        theme_param: Optional[str] = None
        if theme:
            preset = cls.THEME_PRESETS.get(theme.lower())
            if preset is not None:
                theme_param = preset or None   # empty string → no filter
            else:
                # Caller passed a raw GDELT theme string (e.g. "MILITARY")
                theme_param = theme if theme else None

        seen_urls: set = set()
        results: List[Dict[str, Any]] = []

        for q in queries[:5]:
            safe_q = _sanitise_gdelt_query(q)
            if not safe_q:
                logger.warning(f"GDELT: query '{q}' was empty after sanitisation, skipping")
                continue
            try:
                articles = cls._query_doc_api(safe_q, max_rows, theme_param=theme_param)
                for art in articles:
                    url = art.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(art)
            except Exception as exc:
                logger.warning(f"GDELT query failed for '{q}': {exc}")

        logger.info(f"GDELT fetch_by_queries → {len(results)} unique articles")
        return results

    @classmethod
    def _query_doc_api(
        cls,
        query: str,
        max_rows: int,
        cache_key: Optional[tuple] = None,
        theme_param: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Call the GDELT Doc API with retry + back-off.

        Retry policy
        ------------
        - Retries up to _MAX_RETRIES times.
        - HTTP 429 (rate-limited): waits _RATE_LIMIT_PAUSE seconds then retries.
        - Timeout / connection error: exponential backoff with jitter.
        - Non-JSON response (parser rejection): no retry — the query itself is bad.
        - HTTP 5xx: exponential backoff + retry.
        """
        params: Dict[str, Any] = {
            "query":   query,
            "mode":    "artlist",
            "format":  "json",
            "maxrows": max_rows,
        }
        if theme_param:
            params["theme"] = theme_param

        last_exc: Optional[Exception] = None

        for attempt in range(cls._MAX_RETRIES):
            try:
                resp = requests.get(
                    cls.BASE_URL,
                    params=params,
                    timeout=cls._REQUEST_TIMEOUT,
                )

                # ── Rate-limit handling ───────────────────────────────────
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    wait = max(retry_after, 10) + random.uniform(0, 3)
                    logger.warning(
                        f"GDELT rate-limited (429). "
                        f"Waiting {wait:.1f}s before retry {attempt + 1}/{cls._MAX_RETRIES}"
                    )
                    time.sleep(wait)
                    continue

                # ── Server errors: retry ──────────────────────────────────
                if resp.status_code >= 500:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        f"GDELT HTTP {resp.status_code}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{cls._MAX_RETRIES})"
                    )
                    time.sleep(delay)
                    last_exc = Exception(f"HTTP {resp.status_code}")
                    continue

                resp.raise_for_status()

                ct = resp.headers.get("Content-Type", "")
                if "json" not in ct.lower():
                    # GDELT returns HTML when the query is syntactically invalid.
                    # Log the reason and do NOT retry — the query itself is wrong.
                    snippet = resp.text[:200].replace("\n", " ")
                    logger.warning(
                        f"GDELT non-JSON response ({ct}) for query='{query}': {snippet}"
                    )
                    return []

                data = resp.json()
                articles = data.get("articles", [])

                processed = [
                    {
                        "title":        art.get("title", "No Title"),
                        "url":          art.get("url", ""),
                        "source":       art.get("domain", art.get("source", "Unknown")),
                        "published_at": art.get("seendate", ""),
                        "description":  art.get("title", ""),
                        "tone":         art.get("tone", None),
                    }
                    for art in articles
                ]

                if cache_key is not None:
                    cls._cache[cache_key] = (time.time(), processed)

                return processed

            except requests.exceptions.Timeout as exc:
                delay = _backoff_delay(attempt)
                logger.warning(
                    f"GDELT request timed out (attempt {attempt + 1}/{cls._MAX_RETRIES}). "
                    f"Retrying in {delay:.1f}s"
                )
                last_exc = exc
                time.sleep(delay)

            except requests.exceptions.ConnectionError as exc:
                delay = _backoff_delay(attempt)
                logger.warning(
                    f"GDELT connection error (attempt {attempt + 1}/{cls._MAX_RETRIES}): {exc}. "
                    f"Retrying in {delay:.1f}s"
                )
                last_exc = exc
                time.sleep(delay)

            except Exception as exc:
                logger.error(f"GDELT Doc API unexpected error: {exc}")
                return []

        # All retries exhausted
        logger.error(
            f"GDELT Doc API failed after {cls._MAX_RETRIES} retries. "
            f"Last error: {last_exc}"
        )
        return []

    @classmethod
    def fetch_live_ticker(cls, countries: List[str], count: int = 10) -> List[Dict[str, Any]]:
        """Fetch latest news for a live ticker. (legacy, unchanged)"""
        return cls.fetch_war_news(countries, max_rows=count)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GKG (Global Knowledge Graph) monitor – your TL's original class,
#     kept intact with minor improvements (better logging, type hints)
# ─────────────────────────────────────────────────────────────────────────────

class GDELTGraphMonitor:
    """
    Reads the GDELT GKG (Global Knowledge Graph) v2 live feed.
    Downloads the raw 15-minute GKG CSV, filters by theme + country pair,
    and optionally scrapes full article text in parallel.
    """

    LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

    # GKG column indices (zero-based)
    COL_DATE      = 1
    COL_THEMES    = 7
    COL_LOCATIONS = 9
    COL_PERSONS   = 11
    COL_TONE      = 15
    COL_SOURCEURL = 26

    HEADERS = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # ── URL / title helpers ──────────────────────────────────────────────────

    def extract_url(self, raw: str) -> str:
        if pd.isna(raw):
            return ""
        raw = str(raw)
        match = re.search(r"<PAGE_LINKS>(https?://[^<]+)</PAGE_LINKS>", raw)
        raw_url = match.group(1) if match else ""
        if not raw_url:
            match = re.search(r"https?://[^\s<>\"]+", raw)
            if not match:
                return ""
            raw_url = match.group(0)
        first = raw_url.split(";")[0].strip()
        first = re.sub(r"/amp/?$", "", first, flags=re.IGNORECASE)
        first = first.replace("?amp=1", "").replace("&amp=1", "")
        return first

    def extract_title(self, raw: str) -> str:
        if pd.isna(raw):
            return ""
        match = re.search(r"<PAGE_TITLE>([^<]+)</PAGE_TITLE>", str(raw))
        return match.group(1).strip() if match else ""

    # ── Article scraping ─────────────────────────────────────────────────────

    def scrape_article(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        result: Dict[str, Any] = {"url": url, "title": "", "text": "", "error": None}
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "iframe", "noscript", "figure"]):
                tag.decompose()

            title_tag = soup.find("h1") or soup.find("title")
            result["title"] = title_tag.get_text(strip=True) if title_tag else ""

            text = ""
            for selector in [
                "article",
                '[class*="article-body"]', '[class*="article_body"]',
                '[class*="story-body"]',   '[class*="story_body"]',
                '[class*="post-content"]', '[class*="post_content"]',
                '[class*="entry-content"]',
                "main",
                '[role="main"]',
            ]:
                container = soup.select_one(selector)
                if container:
                    paras = container.find_all("p")
                    text = " ".join(p.get_text(strip=True) for p in paras)
                    if len(text) > 200:
                        break

            if len(text) < 200:
                text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

            result["text"] = text.strip()

        except requests.HTTPError as exc:
            result["error"] = f"HTTP {exc.response.status_code}"
        except requests.RequestException as exc:
            result["error"] = f"Request failed: {exc}"
        except Exception as exc:
            result["error"] = f"Parse error: {exc}"

        return result

    def scrape_articles_parallel(self, urls: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
        results: List[Optional[Dict]] = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.scrape_article, url): i
                             for i, url in enumerate(urls)}
            for future in as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        return results  # type: ignore[return-value]

    # ── GKG fetch ─────────────────────────────────────────────────────────────

    def fetch_country_tension(
        self,
        country1: str = "IN",
        country2: str = "CN",
        theme: str = "MILITARY",
    ) -> pd.DataFrame:
        """
        Download the latest GKG 15-minute snapshot and filter for records
        that mention both countries and the specified GDELT theme.

        Returns a DataFrame with columns:
            Date, Themes, Locations, Persons, Tone_Score, URL, Title
        """
        logger.info(f"Fetching GKG tension data: {country1}–{country2} / {theme}")

        r = requests.get(self.LAST_UPDATE_URL, timeout=15)
        r.raise_for_status()
        # Line 3 is the GKG file
        gkg_link = r.text.strip().split("\n")[2].split(" ")[2].strip()

        file_res = requests.get(gkg_link, timeout=60)
        file_res.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(file_res.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep="\t", encoding="latin-1",
                                 header=None, low_memory=False)

        df = df[[self.COL_DATE, self.COL_THEMES, self.COL_LOCATIONS,
                 self.COL_PERSONS, self.COL_TONE, self.COL_SOURCEURL]]
        df.columns = ["Date", "Themes", "Locations", "Persons", "Tone", "RawURL"]

        # Filter theme
        df = df[df["Themes"].str.contains(theme, na=False)]

        # Filter both countries
        def has_both(locs: str) -> bool:
            if pd.isna(locs):
                return False
            return (country1 in locs) and (country2 in locs)

        tension_df = df[df["Locations"].apply(has_both)].copy()
        tension_df["Tone_Score"] = tension_df["Tone"].str.split(",").str[0].astype(float)
        tension_df["URL"]        = tension_df["RawURL"].apply(self.extract_url)
        tension_df["Title"]      = tension_df["RawURL"].apply(self.extract_title)
        tension_df = tension_df[tension_df["URL"] != ""]

        logger.info(f"GKG returned {len(tension_df)} matching records")
        return tension_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GDELTEventFeed – LLM-driven queries → graph injection
#     Mirrors GraphBuilderService.fetch_and_add_news() but uses GDELT (free)
# ─────────────────────────────────────────────────────────────────────────────

class GDELTEventFeed:
    """
    High-level service that wires GDELT into the graph-build pipeline.

    Usage (in GraphBuilderService or graph.py build endpoint):

        feed = GDELTEventFeed()
        episode_uuid = feed.fetch_and_inject(
            zep_client  = self.client,
            graph_id    = graph_id,
            simulation_requirement = project.simulation_requirement,
            context_text           = text,
            project_id             = project_id,
        )

    How queries are generated
    -------------------------
    The LLM produces 3 concise keyword queries (same pattern as NewsAPI's
    _generate_smart_news_queries) PLUS picks the most relevant GDELT theme
    category. Queries are sanitised before hitting the API, so LLM output
    with natural-language phrasing won't cause parser rejections.
    """

    NUM_QUERIES       = 3
    ARTICLES_PER_QUERY = 5
    MAX_FINAL_ARTICLES = 4

    def generate_gdelt_queries(
        self,
        simulation_requirement: str,
        context_text: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """
        Use the LLM to produce GDELT-optimised search queries AND pick the
        most relevant GDELT theme category for this scenario.

        Returns:
            (queries, theme_key)  where theme_key is one of GDELTFetcher.THEME_PRESETS
        """
        context_snippet = (context_text or "")[:2000]
        theme_options   = list(GDELTFetcher.THEME_PRESETS.keys())

        system_prompt = (
            "You are an expert at formulating search queries for the GDELT Project "
            "database, which indexes global news by geopolitical events, actors, "
            "and themes. GDELT search works best with concise entity/topic keywords "
            "rather than full sentences. Respond ONLY with valid JSON."
        )
        user_prompt = f"""Given this simulation requirement and document context, produce:

1. Exactly {self.NUM_QUERIES} GDELT search queries (3–6 keywords each, no sentences).
   - Focus on key geopolitical actors, countries, organisations, or event types.
   - Avoid filler words like "current", "latest", "information about".
   - Do NOT use parentheses, AND/OR/NOT operators, or special characters.
     Plain space-separated keywords only.

2. The single best GDELT theme category from this list: {theme_options}
   - "conflict"    → military, wars, arms
   - "diplomacy"   → international relations, treaties, summits
   - "economy"     → finance, sanctions, trade
   - "environment" → climate, disasters, deforestation
   - "protest"     → civil unrest, regime change, strikes
   - "health"      → pandemics, vaccines, disease outbreaks
   - "technology"  → cyber, AI, tech policy
   - "general"     → use when none of the above fit clearly

Simulation Requirement:
{simulation_requirement}

Document Context (excerpt):
{context_snippet}

Return ONLY this JSON (no markdown):
{{
  "queries": ["query 1", "query 2", "query 3"],
  "theme":   "conflict"
}}"""

        try:
            llm    = LLMClient()
            result = llm.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                operation_name="gdelt_query_generation",
                project_id=project_id,
                simulation_id=None,
            )
            queries = result.get("queries", [])
            theme   = result.get("theme", "general")

            if not isinstance(queries, list) or not queries:
                raise ValueError("LLM returned empty queries")

            # Sanitise every query before returning
            queries = [
                _sanitise_gdelt_query(str(q).strip())
                for q in queries if q
            ]
            queries = [q for q in queries if q][:self.NUM_QUERIES]

            if not queries:
                raise ValueError("All queries were empty after sanitisation")

            if theme not in GDELTFetcher.THEME_PRESETS:
                theme = "general"

            logger.info(f"GDELT LLM queries: {queries}  theme: {theme}")
            return queries, theme

        except Exception as exc:
            logger.warning(f"GDELT query generation failed, falling back: {exc}")
            fallback = _sanitise_gdelt_query(simulation_requirement[:80])
            return [fallback] if fallback else ["geopolitical events"], "general"

    def filter_relevant_articles(
        self,
        articles: List[Dict[str, Any]],
        simulation_requirement: str,
        limit: int = 6,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """LLM-rank articles by relevance to the simulation and return top N."""
        if not articles:
            return []

        article_list = "\n".join(
            f"[{i+1}] {art.get('title', '')} | {art.get('source', '')} | {art.get('published_at', '')}"
            for i, art in enumerate(articles[:30])
        )

        system_prompt = (
            "You are an expert at selecting the most relevant global news for "
            "agent-based simulations. Respond ONLY with a JSON array of integers."
        )
        user_prompt = f"""Simulation goal: {simulation_requirement}

News articles (title | source | date):
{article_list}

Select the {limit} article numbers (1-based) most relevant to this simulation goal.
Return ONLY a JSON array of integers, e.g. [2, 5, 7, 1, 4, 9]"""

        try:
            llm     = LLMClient()
            indices = llm.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                operation_name="gdelt_article_filter",
                project_id=project_id,
                simulation_id=None,
            )
            if isinstance(indices, list):
                selected = []
                for idx in indices:
                    i = int(idx) - 1
                    if 0 <= i < len(articles):
                        selected.append(articles[i])
                return selected[:limit]
        except Exception as exc:
            logger.warning(f"GDELT article filtering failed: {exc}")

        return articles[:limit]

    def fetch_and_inject(
        self,
        zep_client: Any,
        graph_id: str,
        simulation_requirement: str,
        context_text: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Full pipeline:
          1. LLM generates GDELT queries + theme
          2. Sanitise queries
          3. Fetch articles from GDELT Doc API (with retry + rate-limit handling)
          4. LLM filters for relevance
          5. Format as background context (same style as NewsAPI inject)
          6. Inject into Zep graph as text episode

        Returns the Zep episode UUID on success, None on failure.
        """
        try:
            logger.info(f"GDELTEventFeed.fetch_and_inject → graph_id={graph_id}")

            # Step 1: generate + sanitise queries
            queries, theme = self.generate_gdelt_queries(
                simulation_requirement, context_text, project_id
            )

            # Step 2: fetch from GDELT
            # theme="general" means no theme filter; pass None so the param is omitted
            articles = GDELTFetcher.fetch_by_queries(
                queries,
                max_rows=self.ARTICLES_PER_QUERY,
                theme=theme if theme != "general" else None,
            )

            if not articles:
                logger.warning("GDELTEventFeed: no articles returned, skipping inject")
                return None

            # Step 3: LLM filter
            selected = self.filter_relevant_articles(
                articles,
                simulation_requirement,
                limit=self.MAX_FINAL_ARTICLES,
                project_id=project_id,
            )

            # Step 4: format (same style as NewsAPI inject so graph is consistent)
            lines = [
                "### BACKGROUND CONTEXT – GDELT GLOBAL EVENT FEED (Supplementary only) ###",
                f"Date: {time.strftime('%Y-%m-%d %H:%M UTC')}",
                f"Simulation Requirement: {simulation_requirement}",
                f"GDELT Theme Filter: {theme}",
                f"Queries Used: {', '.join(queries)}",
                "Note: GDELT data is supplementary background. "
                "Primary source remains the uploaded documents.",
                "",
                "--- GDELT GLOBAL EVENT ARTICLES ---",
                "",
            ]

            for i, art in enumerate(selected, 1):
                lines += [
                    f"[Global Event {i}]:",
                    f"Title:  {art.get('title', 'N/A')}",
                    f"Source: {art.get('source', 'N/A')}",
                    f"Date:   {art.get('published_at', 'N/A')}",
                    f"URL:    {art.get('url', '')}",
                    f"Tone:   {art.get('tone', 'N/A')}",
                    "",
                ]

            lines.append("--- END GDELT GLOBAL EVENT FEED ---")
            report_text = "\n".join(lines)

            logger.info(
                f"GDELTEventFeed: injecting {len(selected)} articles "
                f"({len(report_text)} chars)"
            )

            # Step 5: inject into Zep
            uuids = zep_client.graph.add(
                graph_id=graph_id,
                type="text",
                data=report_text,
            )

            logger.info(
                f"GDELTEventFeed: Zep returned "
                f"type={type(uuids).__name__} "
                f"uuid_={getattr(uuids, 'uuid_', None)}"
            )

            # graph.add() returns a single Episode object — extract UUID string
            if isinstance(uuids, list) and uuids:
                ep = uuids[0]
                return getattr(ep, "uuid_", None) or getattr(ep, "uuid", None) or ep
            if uuids is not None:
                return getattr(uuids, "uuid_", None) or getattr(uuids, "uuid", None) or uuids
            return None

        except Exception as exc:
            logger.error(f"GDELTEventFeed.fetch_and_inject failed: {exc}")
            return None