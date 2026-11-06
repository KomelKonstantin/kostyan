"""Модели данных матча.

Правило: значение ``None`` означает «[нет данных]». Ничего не подставляется
«по умолчанию» молча — отсутствие данных учитывается в уверенности модели и
явно выводится в отчёте.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

NO_DATA = "[нет данных]"


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19] if "T" in text else text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Не удалось разобрать дату: {value!r} (ожидается YYYY-MM-DD)")


def _pct(value: Any) -> float | None:
    """Принимает 0.55, 55, "55%" → 0.55."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%").replace(",", ".")
        if not value:
            return None
    v = float(value)
    if v > 1.0:
        v /= 100.0
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"Процент вне диапазона: {value!r}")
    return v


@dataclass
class MapStat:
    name: str
    played: int | None = None
    winrate: float | None = None          # доля выигранных карт, 0..1
    ct_round_win: float | None = None     # доля выигранных раундов за CT
    t_round_win: float | None = None      # доля выигранных раундов за T
    pick_rate: float | None = None        # доля вето, где команда пикает карту
    ban_rate: float | None = None         # доля вето, где команда банит карту

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "MapStat":
        played = d.get("played")
        return cls(
            name=normalize_map(name),
            played=int(played) if played is not None else None,
            winrate=_pct(d.get("winrate")),
            ct_round_win=_pct(d.get("ct_round_win")),
            t_round_win=_pct(d.get("t_round_win")),
            pick_rate=_pct(d.get("pick_rate")),
            ban_rate=_pct(d.get("ban_rate")),
        )


@dataclass
class RecentMatch:
    date: date | None
    opponent: str
    won: bool
    opponent_rank: int | None = None
    maps_won: int | None = None
    maps_lost: int | None = None
    lan: bool | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecentMatch":
        r = d.get("opponent_rank")
        return cls(
            date=_parse_date(d.get("date")),
            opponent=str(d.get("opponent", "?")),
            won=bool(d["won"]),
            opponent_rank=int(r) if r not in (None, "") else None,
            maps_won=d.get("maps_won"),
            maps_lost=d.get("maps_lost"),
            lan=d.get("lan"),
        )


@dataclass
class Player:
    name: str
    rating: float | None = None   # HLTV Rating за ~3 месяца
    is_standin: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | str) -> "Player":
        if isinstance(d, str):
            return cls(name=d)
        rating = d.get("rating")
        return cls(
            name=str(d["name"]),
            rating=float(rating) if rating not in (None, "") else None,
            is_standin=bool(d.get("is_standin", False)),
        )


@dataclass
class Team:
    name: str
    hltv_id: int | None = None
    slug: str | None = None
    rank: int | None = None
    points: int | None = None
    players: list[Player] = field(default_factory=list)
    recent_matches: list[RecentMatch] = field(default_factory=list)
    maps: dict[str, MapStat] = field(default_factory=dict)
    lan_winrate: float | None = None
    lan_matches: int | None = None
    online_winrate: float | None = None
    online_matches: int | None = None
    pistol_winrate: float | None = None
    days_since_roster_change: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Team":
        maps_raw = d.get("maps") or {}
        if isinstance(maps_raw, list):
            maps_raw = {m["name"]: m for m in maps_raw}
        maps = {}
        for name, m in maps_raw.items():
            ms = MapStat.from_dict(name, m)
            maps[ms.name] = ms

        def _int(key: str) -> int | None:
            v = d.get(key)
            return int(v) if v not in (None, "") else None

        return cls(
            name=str(d["name"]),
            hltv_id=_int("hltv_id"),
            slug=d.get("slug"),
            rank=_int("rank"),
            points=_int("points"),
            players=[Player.from_dict(p) for p in d.get("players") or []],
            recent_matches=[RecentMatch.from_dict(m) for m in d.get("recent_matches") or []],
            maps=maps,
            lan_winrate=_pct(d.get("lan_winrate")),
            lan_matches=_int("lan_matches"),
            online_winrate=_pct(d.get("online_winrate")),
            online_matches=_int("online_matches"),
            pistol_winrate=_pct(d.get("pistol_winrate")),
            days_since_roster_change=_int("days_since_roster_change"),
        )


@dataclass
class VetoMap:
    name: str
    picked_by: int | None = None  # 1, 2 или None (десайдер / неизвестно)


@dataclass
class H2HMatch:
    date: date | None
    winner: int  # 1 или 2


@dataclass
class Match:
    team1: Team
    team2: Team
    best_of: int | None = None
    lan: bool | None = None
    event: str | None = None
    date: date | None = None
    url: str | None = None
    veto: list[VetoMap] = field(default_factory=list)   # известное вето (если опубликовано)
    map_pool: list[str] = field(default_factory=list)
    h2h: list[H2HMatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)       # источники, предупреждения

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Match":
        veto = []
        for v in d.get("veto") or []:
            if isinstance(v, str):
                veto.append(VetoMap(normalize_map(v)))
            else:
                veto.append(VetoMap(normalize_map(v["name"]), v.get("picked_by")))
        h2h = [H2HMatch(_parse_date(h.get("date")), int(h["winner"])) for h in d.get("h2h") or []]
        bo = d.get("best_of")
        return cls(
            team1=Team.from_dict(d["team1"]),
            team2=Team.from_dict(d["team2"]),
            best_of=int(bo) if bo not in (None, "") else None,
            lan=d.get("lan"),
            event=d.get("event"),
            date=_parse_date(d.get("date")),
            url=d.get("url"),
            veto=veto,
            map_pool=[normalize_map(m) for m in d.get("map_pool") or []],
            h2h=h2h,
            notes=list(d.get("notes") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        def conv(o: Any) -> Any:
            if isinstance(o, date):
                return o.isoformat()
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [conv(v) for v in o]
            return o

        return conv(asdict(self))


_MAP_ALIASES = {
    "dust2": "Dust2", "dust 2": "Dust2", "dust ii": "Dust2", "de_dust2": "Dust2",
    "mirage": "Mirage", "inferno": "Inferno", "nuke": "Nuke", "ancient": "Ancient",
    "anubis": "Anubis", "train": "Train", "overpass": "Overpass", "vertigo": "Vertigo",
    "cache": "Cache", "cobblestone": "Cobblestone",
}


def normalize_map(name: str) -> str:
    key = name.strip().lower().replace("de_", "")
    return _MAP_ALIASES.get(key, name.strip().title())


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно накладывает ``override`` на ``base`` (ручные данные важнее)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        elif v is not None:
            out[k] = v
    return out
