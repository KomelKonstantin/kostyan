"""Парсинг страниц HLTV (best-effort).

Вёрстка HLTV меняется; каждый парсер возвращает только то, что реально нашёл.
Не найдено → поле остаётся ``None`` и в отчёте будет помечено «[нет данных]».
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlparse

from bs4 import BeautifulSoup

from .fetch import BASE_URL, BlockedError, HtmlSource, SourceMissingError
from .models import normalize_map

MATCH_URL_RE = re.compile(r"^/matches/(\d+)/([\w-]+)")
TEAM_HREF_RE = re.compile(r"/team/(\d+)/([\w-]+)")
PLAYER_HREF_RE = re.compile(r"/player/(\d+)/([\w-]+)")


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el is not None else ""


def _first_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def _first_float(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", "."))
    return float(m.group()) if m else None


def parse_match_url(url: str) -> tuple[int, str]:
    path = urlparse(url).path
    m = MATCH_URL_RE.match(path)
    if not m or "hltv.org" not in (urlparse(url).netloc or "hltv.org"):
        raise ValueError(f"Не похоже на ссылку матча HLTV: {url}")
    return int(m.group(1)), m.group(2)


# ---------------------------------------------------------------- match page
def parse_match_page(html: str) -> dict[str, Any]:
    s = _soup(html)
    out: dict[str, Any] = {"team1": {}, "team2": {}}

    for idx in (1, 2):
        box = s.select_one(f"div.team{idx}-gradient")
        if box is None:
            continue
        name = _text(box.select_one(".teamName"))
        a = box.select_one("a[href^='/team/']")
        team: dict[str, Any] = {"name": name or None}
        if a is not None:
            m = TEAM_HREF_RE.search(a["href"])
            if m:
                team["hltv_id"], team["slug"] = int(m.group(1)), m.group(2)
        out[f"team{idx}"] = team

    fmt = s.select_one("div.padding.preformatted-text")
    fmt_text = _text(fmt)
    m = re.search(r"Best of (\d)", fmt_text, re.I)
    out["best_of"] = int(m.group(1)) if m else None
    if re.search(r"\(LAN\)", fmt_text, re.I):
        out["lan"] = True
    elif re.search(r"\(Online\)", fmt_text, re.I):
        out["lan"] = False
    else:
        out["lan"] = None

    out["event"] = _text(s.select_one("div.event a")) or None
    d = s.select_one("div.date[data-unix]") or s.select_one("div.time[data-unix]")
    if d is not None:
        try:
            out["date"] = datetime.fromtimestamp(int(d["data-unix"]) / 1000, tz=timezone.utc).date().isoformat()
        except (ValueError, KeyError):
            pass

    # Ранги из блока составов: "World rank: #5"
    lineups = s.select("div.lineups div.lineup")
    for idx, lu in enumerate(lineups[:2], start=1):
        rank = _first_int(_text(lu.select_one(".teamRanking")))
        if rank is not None:
            out[f"team{idx}"]["rank"] = rank
        players, seen = [], set()
        for a in lu.select("a[href*='/player/']"):
            m = PLAYER_HREF_RE.search(a.get("href", ""))
            if not m or m.group(1) in seen:
                continue
            seen.add(m.group(1))
            nick = _text(a.select_one(".text-ellipsis")) or _text(a) or m.group(2)
            players.append({"name": nick, "hltv_id": int(m.group(1)), "slug": m.group(2)})
        if players:
            out[f"team{idx}"]["players"] = players[:5]

    # Вето: "3. Team picked Mirage", "7. Ancient was left over"
    veto: list[dict[str, Any]] = []
    t1n = (out["team1"].get("name") or "").lower()
    t2n = (out["team2"].get("name") or "").lower()
    for box in s.select("div.veto-box"):
        for line in box.get_text("\n", strip=True).split("\n"):
            pm = re.match(r"\d+\.\s*(.+?)\s+picked\s+(.+)$", line, re.I)
            lm = re.match(r"\d+\.\s*(.+?)\s+was left over", line, re.I)
            if pm:
                who = pm.group(1).strip().lower()
                picked_by = 1 if who == t1n else 2 if who == t2n else None
                veto.append({"name": normalize_map(pm.group(2)), "picked_by": picked_by})
            elif lm:
                veto.append({"name": normalize_map(lm.group(1)), "picked_by": None})
    if not veto:
        for mh in s.select("div.mapholder .mapname"):
            name = _text(mh)
            if name and name.upper() != "TBA":
                veto.append({"name": normalize_map(name), "picked_by": None})
    out["veto"] = veto

    # H2H
    h2h = []
    for row in s.select("div.head-to-head-listing tr"):
        date_txt = _text(row.select_one(".date"))
        won = row.select_one(".won")
        winner_name = _text(row.select_one(".winner")) or ""
        if not date_txt:
            continue
        try:
            dt = datetime.strptime(date_txt, "%d/%m/%y").date().isoformat()
        except ValueError:
            dt = None
        winner = None
        if winner_name.lower() == t1n:
            winner = 1
        elif winner_name.lower() == t2n:
            winner = 2
        if winner is None and won is not None:
            continue
        if winner:
            h2h.append({"date": dt, "winner": winner})
    out["h2h"] = h2h
    return out


# --------------------------------------------------------------- ranking page
def parse_ranking(html: str) -> dict[str, dict[str, int]]:
    """{имя_в_нижнем_регистре: {"rank": .., "points": ..}}"""
    s = _soup(html)
    res: dict[str, dict[str, int]] = {}
    for team in s.select("div.ranked-team"):
        rank = _first_int(_text(team.select_one(".position")))
        name = _text(team.select_one(".teamLine .name")) or _text(team.select_one(".name"))
        pts = _first_int(_text(team.select_one(".points")))
        if rank and name:
            res[name.lower()] = {"rank": rank, "points": pts}
    return res


# ---------------------------------------------------------------- team results
_RESULTS_DATE_RE = re.compile(r"Results for \w+ (\d+)\w* (\d{4})", re.I)


def parse_team_results(html: str, team_name: str, ranking: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Страница /results?team=<id>: дата, соперник, счёт."""
    s = _soup(html)
    out = []
    for sub in s.select("div.results-sublist"):
        headline = _text(sub.select_one(".standard-headline"))
        dt = None
        m = re.search(r"Results for (\w+) (\d+)\w* (\d{4})", headline)
        if m:
            try:
                dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").date()
            except ValueError:
                dt = None
        for con in sub.select("div.result-con"):
            t1 = _text(con.select_one(".team1 .team"))
            t2 = _text(con.select_one(".team2 .team"))
            scores = [_first_int(_text(x)) for x in con.select(".result-score span")]
            if len(scores) != 2 or None in scores or not t1 or not t2:
                continue
            if t1.lower() == team_name.lower():
                opp, mine, theirs = t2, scores[0], scores[1]
            elif t2.lower() == team_name.lower():
                opp, mine, theirs = t1, scores[1], scores[0]
            else:
                continue
            if mine == theirs:
                continue
            # Для BO1 счёт — раунды (16-10), для серий — карты (2-1).
            maps_won, maps_lost = (mine, theirs) if max(mine, theirs) <= 3 else (None, None)
            out.append({
                "date": dt.isoformat() if dt else None,
                "opponent": opp,
                "won": mine > theirs,
                "opponent_rank": (ranking.get(opp.lower()) or {}).get("rank"),
                "maps_won": maps_won,
                "maps_lost": maps_lost,
            })
    return out


# ------------------------------------------------------------ team map stats
def parse_team_maps(html: str) -> dict[str, dict[str, Any]]:
    """Страница /stats/teams/maps/<id>/<slug>: сыграно и winrate по картам."""
    s = _soup(html)
    out: dict[str, dict[str, Any]] = {}
    for col in s.select("div.two-grid div.col"):
        title = _text(col.select_one(".map-pool-map-name"))
        if not title:
            continue
        name = normalize_map(title.split(" - ")[0])
        stats: dict[str, Any] = {}
        for row in col.select("div.stats-row"):
            spans = row.select("span")
            if len(spans) < 2:
                continue
            label, value = _text(spans[0]).lower(), _text(spans[-1])
            if "times played" in label:
                stats["played"] = _first_int(value)
            elif label.startswith("win rate"):
                v = _first_float(value)
                stats["winrate"] = v / 100 if v is not None else None
        if stats:
            out[name] = stats
    return out


# ---------------------------------------------------------------- team page
def parse_team_page(html: str) -> dict[str, Any]:
    s = _soup(html)
    out: dict[str, Any] = {}
    rank = _first_int(_text(s.select_one(".profile-team-stat a[href*='ranking']")))
    if rank:
        out["rank"] = rank
    players, seen = [], set()
    for a in s.select(".bodyshot-team a[href*='/player/']"):
        m = PLAYER_HREF_RE.search(a.get("href", ""))
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        nick = a.get("title") or _text(a) or m.group(2)
        players.append({"name": nick, "hltv_id": int(m.group(1)), "slug": m.group(2)})
    if players:
        out["players"] = players[:5]
    return out


# ------------------------------------------------------------- player stats
def parse_player_rating(html: str) -> float | None:
    s = _soup(html)
    for box in s.select(".summaryStatBreakdown"):
        if "rating" in _text(box.select_one(".summaryStatBreakdownSubHeader")).lower():
            return _first_float(_text(box.select_one(".summaryStatBreakdownDataValue")))
    el = s.select_one(".summaryStatBreakdownDataValue")
    return _first_float(_text(el)) if el else None


# ------------------------------------------------------------------- search
def parse_search(body: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    teams = []
    for block in data if isinstance(data, list) else [data]:
        for t in block.get("teams", []) if isinstance(block, dict) else []:
            m = TEAM_HREF_RE.search(t.get("location", ""))
            if m:
                teams.append({"name": t.get("name"), "hltv_id": int(m.group(1)), "slug": m.group(2)})
    return teams


# --------------------------------------------------------------- collector
class Collector:
    """Собирает данные матча из источника HTML и пишет предупреждения."""

    def __init__(self, source: HtmlSource, fetch_players: bool = True, today: date | None = None) -> None:
        self.source = source
        self.fetch_players = fetch_players
        self.today = today or date.today()
        self.notes: list[str] = []

    def _try(self, role: str, url: str) -> str | None:
        try:
            return self.source.get(role, url)
        except SourceMissingError as e:
            self.notes.append(f"{role}: {e}")
            return None
        # BlockedError пробрасывается наверх намеренно — CLI остановится.

    def _period(self) -> str:
        start = self.today - timedelta(days=90)
        return f"startDate={start.isoformat()}&endDate={self.today.isoformat()}"

    def from_match_url(self, url: str) -> dict[str, Any]:
        parse_match_url(url)
        html = self.source.get("match", url)
        data = parse_match_page(html)
        data["url"] = url
        self._enrich(data)
        return data

    def from_team_names(self, name1: str, name2: str) -> dict[str, Any]:
        data: dict[str, Any] = {"team1": {"name": name1}, "team2": {"name": name2}}
        for idx, name in ((1, name1), (2, name2)):
            body = self._try(f"search_team{idx}", f"{BASE_URL}/search?term={quote(name)}")
            found = parse_search(body) if body else []
            exact = [t for t in found if (t["name"] or "").lower() == name.lower()]
            pick = (exact or found or [None])[0]
            if pick:
                data[f"team{idx}"].update(pick)
                page = self._try(f"team{idx}_page", f"{BASE_URL}/team/{pick['hltv_id']}/{pick['slug']}")
                if page:
                    data[f"team{idx}"].update(parse_team_page(page))
            else:
                self.notes.append(f"Команда «{name}» не найдена в поиске HLTV")
        self._enrich(data)
        return data

    def _enrich(self, data: dict[str, Any]) -> None:
        ranking_html = self._try("ranking", f"{BASE_URL}/ranking/teams")
        ranking = parse_ranking(ranking_html) if ranking_html else {}
        for idx in (1, 2):
            team = data[f"team{idx}"]
            name = team.get("name") or ""
            r = ranking.get(name.lower())
            if r:
                team.setdefault("rank", r["rank"])
                team["points"] = r.get("points")
            tid, slug = team.get("hltv_id"), team.get("slug")
            if tid and slug:
                res = self._try(f"team{idx}_results", f"{BASE_URL}/results?team={tid}")
                if res:
                    team["recent_matches"] = parse_team_results(res, name, ranking)
                maps = self._try(f"team{idx}_maps", f"{BASE_URL}/stats/teams/maps/{tid}/{slug}?{self._period()}")
                if maps:
                    team["maps"] = parse_team_maps(maps)
            if self.fetch_players:
                for p in team.get("players") or []:
                    if not p.get("hltv_id"):
                        continue
                    ph = self._try(
                        f"player_{p['slug']}",
                        f"{BASE_URL}/stats/players/{p['hltv_id']}/{p['slug']}?{self._period()}",
                    )
                    if ph:
                        p["rating"] = parse_player_rating(ph)


__all__ = ["Collector", "BlockedError", "parse_match_url", "parse_match_page", "parse_ranking",
           "parse_team_results", "parse_team_maps", "parse_player_rating", "parse_search"]
