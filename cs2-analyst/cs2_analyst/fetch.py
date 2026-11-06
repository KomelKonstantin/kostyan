"""Получение HTML-страниц HLTV.

Политика:
* только вежливые запросы: пауза ≥ 3 с между запросами, честный User-Agent,
  дисковый кэш, никаких повторных «долбёжек»;
* если HLTV отвечает защитой Cloudflare (403/429/503 или страница-челлендж),
  инструмент НЕ пытается её обойти (никаких прокси, подмены IP, решения капчи) —
  выбрасывается :class:`BlockedError`, и пользователю предлагаются
  легальные альтернативы: сохранить страницы через браузер (Claude in Chrome)
  и передать каталог через ``--html-dir`` либо заполнить данные вручную.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol

BASE_URL = "https://www.hltv.org"
MIN_DELAY_SECONDS = 3.0
DEFAULT_CACHE_TTL = 6 * 3600
USER_AGENT = "cs2-analyst/0.1 (personal match research; polite, cached, >=3s between requests)"

_CHALLENGE_MARKERS = (
    "just a moment...",
    "cf-chl",
    "challenge-platform",
    "cf_chl_opt",
    "attention required! | cloudflare",
    "enable javascript and cookies to continue",
)


class BlockedError(RuntimeError):
    """HLTV закрыл доступ (Cloudflare). Обход запрещён — нужен другой источник."""


class SourceMissingError(RuntimeError):
    """Страница отсутствует в локальном источнике."""


class HtmlSource(Protocol):
    def get(self, role: str, url: str) -> str: ...


def looks_like_challenge(status: int, body: str) -> bool:
    if status in (403, 429, 503):
        return True
    head = body[:5000].lower()
    return any(m in head for m in _CHALLENGE_MARKERS)


class PoliteFetcher:
    """HTTP-источник с паузами и кэшем на диске."""

    def __init__(
        self,
        cache_dir: Path,
        min_delay: float = MIN_DELAY_SECONDS,
        ttl: int = DEFAULT_CACHE_TTL,
        offline: bool = False,
        verbose: bool = False,
    ) -> None:
        if min_delay < MIN_DELAY_SECONDS:
            raise ValueError(f"Пауза между запросами не может быть меньше {MIN_DELAY_SECONDS} с")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay = min_delay
        self.ttl = ttl
        self.offline = offline
        self.verbose = verbose
        self._last_request = 0.0
        self._session = None
        self.blocked = False

    # --- кэш -------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def _read_cache(self, url: str) -> str | None:
        p = self._cache_path(url)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not self.offline and time.time() - data.get("ts", 0) > self.ttl:
            return None
        return data.get("body")

    def _write_cache(self, url: str, body: str) -> None:
        p = self._cache_path(url)
        p.write_text(json.dumps({"url": url, "ts": time.time(), "body": body}), encoding="utf-8")

    # --- сеть ------------------------------------------------------------
    def _session_obj(self):
        if self._session is None:
            import requests

            s = requests.Session()
            s.headers.update({
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
            })
            self._session = s
        return self._session

    def get(self, role: str, url: str) -> str:
        cached = self._read_cache(url)
        if cached is not None:
            if self.verbose:
                print(f"  [кэш] {role}: {url}")
            return cached
        if self.offline:
            raise SourceMissingError(f"{role}: нет в кэше, а включён --offline ({url})")
        if self.blocked:
            # После первого отказа больше не стучимся — это и есть «вежливость».
            raise BlockedError("HLTV уже ответил защитой Cloudflare; повторные запросы не отправляются")

        wait = self.min_delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        if self.verbose:
            print(f"  [сеть] {role}: {url}")
        try:
            resp = self._session_obj().get(url, timeout=20)
        finally:
            self._last_request = time.monotonic()
        body = resp.text
        if looks_like_challenge(resp.status_code, body):
            self.blocked = True
            raise BlockedError(
                f"HLTV вернул {resp.status_code} / страницу проверки Cloudflare для {url}"
            )
        resp.raise_for_status()
        self._write_cache(url, body)
        return body


class DirectorySource:
    """Локальные HTML-файлы, сохранённые из браузера (например, через Claude in Chrome).

    Имена файлов = роли: ``match.html``, ``ranking.html``, ``team1_results.html``,
    ``team2_results.html``, ``team1_maps.html``, ``team2_maps.html``,
    ``team1_page.html``, ``team2_page.html``, ``player_<ник>.html``.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Каталог не найден: {directory}")

    def get(self, role: str, url: str) -> str:
        for ext in (".html", ".htm", ".json"):
            p = self.directory / f"{role}{ext}"
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        raise SourceMissingError(f"{role}: файл {role}.html не найден в {self.directory}")


class ChainSource:
    """Сначала локальный каталог, затем (если разрешено) сеть."""

    def __init__(self, *sources: HtmlSource) -> None:
        self.sources = [s for s in sources if s is not None]

    def get(self, role: str, url: str) -> str:
        last: Exception | None = None
        for s in self.sources:
            try:
                return s.get(role, url)
            except SourceMissingError as e:
                last = e
        raise last or SourceMissingError(role)
