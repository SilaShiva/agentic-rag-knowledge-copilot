"""
Glean connector — cookie-based backup search, no API key.

Auth priority:
  1. Auto-extract from Chrome cookie DB via browser-cookie3 (Chrome must be closed)
  2. Fall back to .env vars GLEAN_SESSION_COOKIE / GLEAN_AZURE_COOKIE

The 401 path raises GleanAuthError with a friendly message that the UI shows
instead of crashing — cookies expire (~24h) and need a re-copy.
"""
from __future__ import annotations

import os
import urllib3
from typing import Any

import requests

from app.core.config import GLEAN_SERVER_URL
from app.core.logging import logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def _glean_domain() -> str:
    """Host of the configured Glean instance (from GLEAN_SERVER_URL)."""
    from urllib.parse import urlparse

    return urlparse(GLEAN_SERVER_URL).netloc


def _headers() -> dict[str, str]:
    """Request headers — origin/referer point at the instance's web app."""
    domain = _glean_domain()
    root = domain.split(".", 1)[-1] if domain else "glean.example.com"
    return {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "origin": f"https://app.{root}",
        "referer": f"https://app.{root}/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/148.0.0.0 Safari/537.36"
        ),
    }


class GleanAuthError(RuntimeError):
    """Raised when Glean cookies are missing or expired."""


def _extract_chrome_cookies() -> dict[str, str]:
    try:
        import browser_cookie3

        jar = browser_cookie3.chrome(domain_name=_glean_domain())
        return {c.name: c.value for c in jar}
    except Exception:
        return {}


def _get_cookies() -> dict[str, str]:
    base = {"login_remember": "true", "locale": "en"}
    chrome = _extract_chrome_cookies()
    if chrome.get("glean-session-store"):
        return {**base, **chrome}
    session = os.getenv("GLEAN_SESSION_COOKIE")
    azure = os.getenv("GLEAN_AZURE_COOKIE")
    if session:
        base["glean-session-store"] = session
    if azure:
        base["azure-hosted-login-session-store"] = azure
    return base


def is_configured() -> bool:
    """True if Glean cookies are available (Chrome or .env)."""
    chrome = _extract_chrome_cookies()
    if chrome.get("glean-session-store"):
        return True
    return bool(os.getenv("GLEAN_SESSION_COOKIE"))


def _check_auth(cookies: dict) -> None:
    if not cookies.get("glean-session-store"):
        raise GleanAuthError(
            "Glean session cookie not found. Close Chrome and restart the app "
            "(auto-extract), or set GLEAN_SESSION_COOKIE in .env."
        )


def _handle_status(resp: requests.Response) -> None:
    if resp.status_code == 401:
        raise GleanAuthError(
            "Glean cookies expired — close Chrome, restart the app to "
            "auto-refresh, or re-copy the cookie into .env."
        )
    if resp.status_code == 403:
        raise GleanAuthError("Glean returned 403 — insufficient permissions.")
    resp.raise_for_status()


def _post(path: str, payload: dict, timeout: int = 20) -> dict:
    cookies = _get_cookies()
    _check_auth(cookies)
    resp = requests.post(
        f"{GLEAN_SERVER_URL}{path}",
        headers=_headers(),
        cookies=cookies,
        json=payload,
        timeout=timeout,
        verify=False,
    )
    _handle_status(resp)
    return resp.json()


def _extract_results(data: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in data.get("results", []):
        doc = r.get("document")
        if not doc:
            for sr in r.get("structuredResults", []):
                if "document" in sr:
                    doc = sr["document"]
                    break
        if not doc:
            continue
        meta = doc.get("metadata", {})
        raw_snippets = r.get("snippets", [])
        snippet = " … ".join(
            (s if isinstance(s, str) else s.get("text", ""))
            for s in raw_snippets
            if (s if isinstance(s, str) else s.get("text", ""))
        )
        out.append({
            "title": doc.get("title", "Untitled"),
            "url": doc.get("url", ""),
            "snippet": snippet[:600],
            "source": doc.get("datasource", "") or meta.get("datasource", ""),
            "last_updated": meta.get("updateTime", "")[:10],
        })
    return out


def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """General Glean search — Slack, Drive, Confluence, GitHub, …"""
    data = _post(
        "/api/v1/search",
        {"query": query, "pageSize": min(max_results, 10)},
    )
    results = _extract_results(data)
    logger.info("glean: %d results for %r", len(results), query[:60])
    return results


def chat(message: str) -> str:
    """Glean AI synthesis (includes citations inline)."""
    data = _post(
        "/api/v1/chat",
        {"messages": [{
            "author": "USER",
            "messageType": "CONTENT",
            "fragments": [{"text": message}],
        }]},
        timeout=45,
    )
    parts: list[str] = []
    citations: list[str] = []
    for msg in data.get("messages", []):
        if msg.get("author", "").upper() == "GLEAN_ASSISTANT":
            parts.extend(
                f["text"] for f in msg.get("fragments", []) if f.get("text")
            )
        for c in msg.get("citations", []):
            src = c.get("sourceDocument", {})
            title = src.get("title", "Unknown")
            url = src.get("url", "")
            citations.append(f"- {title}" + (f" ({url})" if url else ""))
    answer = "\n".join(parts).strip() or "No response received from Glean."
    if citations:
        answer += "\n\nSources:\n" + "\n".join(citations)
    return answer


def format_results(results: list[dict[str, Any]]) -> str:
    """Compact markdown for the agent prompt."""
    if not results:
        return "No results found."
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"**{i}. {r['title']}**  [{r.get('source', '')}]")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
    return "\n".join(lines)
