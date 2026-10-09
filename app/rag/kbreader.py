"""
Knowledge-base reader — enumerates .md/.yaml/.yml files across all local git branches
without ever checking a branch out.

Every file read goes through `git show <branch>:<path>` against the local clone
(local git clone, see KB_LOCAL_PATH in config).
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from app.core.config import KB_LOCAL_PATH
from app.core.logging import logger

DOC_EXTENSIONS = (".md", ".yaml", ".yml")
# Skip dirs that carry no business knowledge
SKIP_PREFIXES = (".git/", ".claude/scripts/", "audit/raw/")


def _git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or KB_LOCAL_PATH),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()[:300]}")
    return result.stdout


def list_branches() -> list[str]:
    """All local branch heads (the clone keeps one branch per CG function)."""
    out = _git(["for-each-ref", "refs/heads/", "--format=%(refname:short)"])
    branches = [b.strip() for b in out.splitlines() if b.strip()]
    logger.info("kb: %d local branches", len(branches))
    return branches


def list_files(branch: str) -> list[str]:
    """Knowledge files on one branch, filtered to doc extensions."""
    out = _git(["ls-tree", "-r", "--name-only", branch])
    files = []
    for path in out.splitlines():
        p = path.strip()
        if not p or not p.endswith(DOC_EXTENSIONS):
            continue
        if p.startswith(SKIP_PREFIXES):
            continue
        files.append(p)
    return files


def read_file(branch: str, path: str) -> str:
    return _git(["show", f"{branch}:{path}"])


def branch_head(branch: str) -> str:
    """Current commit sha of a branch (for change detection)."""
    return _git(["rev-parse", branch]).strip()


def fetch_origin() -> bool:
    """git fetch — updates local branch heads if remote moved. Returns success."""
    try:
        _git(["fetch", "origin", "--prune"])
        return True
    except Exception as exc:  # offline / auth failure → reindex still works locally
        logger.warning("kb: git fetch failed (%s) — using local branch heads", exc)
        return False


def enumerate_knowledge() -> list[dict]:
    """
    Walk every branch × doc file and return one record per unique content:

        {content_hash, path, branches: [..], content, head}

    The same file frequently exists identically on several branches — dedupe by
    sha1 of content and merge branch lists into one record.
    """
    records: dict[str, dict] = {}
    files_seen = 0
    for branch in list_branches():
        try:
            files = list_files(branch)
        except RuntimeError as exc:
            logger.warning("kb: skipping branch %s (%s)", branch, exc)
            continue
        for path in files:
            files_seen += 1
            try:
                content = read_file(branch, path)
            except RuntimeError as exc:
                logger.debug("kb: unreadable %s@%s (%s)", path, branch, exc)
                continue
            if not content.strip():
                continue
            content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
            rec = records.get(content_hash)
            if rec is None:
                records[content_hash] = {
                    "content_hash": content_hash,
                    "path": path,
                    "branches": [branch],
                    "content": content,
                }
            elif branch not in rec["branches"]:
                rec["branches"].append(branch)
    logger.info(
        "kb: %d files scanned -> %d unique documents", files_seen, len(records)
    )
    return list(records.values())
