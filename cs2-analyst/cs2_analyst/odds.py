"""Разбор коэффициентов (--odds) и оценка ценности ставок.

Формат: JSON-объект ``{"событие": коэффициент}`` или CSV ``событие,коэффициент``
(строка, путь к файлу или ``-`` для stdin). Только десятичные коэффициенты.

Канонические ключи событий (team1/team2 можно заменить названием команды):
  ml:team1                          победа в матче
  score:2-0                         точный счёт серии
  handicap:team1:-1.5               фора по картам
  total_maps:over:2.5               тотал карт
  map1:team1                        победа на карте N
  map1:total_rounds:over:21.5       тотал раундов на карте N
  map1:handicap:team2:+3.5          фора по раундам на карте N
  map1:overtime:yes                 овертайм на карте N
  map1:pistol1:team1                первый пистолет на карте N
Синонимы: П1/П2, W1/W2, ТБ/ТМ, Ф1/Ф2, over/under, o/u.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OddsLine:
    raw_key: str
    key: str
    odds: float


@dataclass
class Assessment:
    key: str
    label: str
    odds: float
    p_model: float
    p_low: float
    p_high: float
    implied: float
    fair_implied: float | None   # без маржи, если найдена пара
    margin: float | None
    ev: float
    ev_low: float
    kelly: float
    verdict: str


def load_odds_text(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    p = Path(arg)
    if len(arg) < 4096 and p.exists():
        return p.read_text(encoding="utf-8")
    return arg


def _parse_raw(text: str) -> list[tuple[str, float]]:
    text = text.strip()
    pairs: list[tuple[str, float]] = []
    if text.startswith("{") or text.startswith("["):
        data = json.loads(text)
        if isinstance(data, dict):
            pairs = [(k, v) for k, v in data.items()]
        else:
            for row in data:
                pairs.append((row.get("event") or row.get("key"), row.get("odds")))
    else:
        reader = csv.reader(io.StringIO(text), delimiter=";" if ";" in text.splitlines()[0] else ",")
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Строка CSV без коэффициента: {row}")
            if row[0].strip().lower() in ("event", "событие") :
                continue
            pairs.append((row[0], row[1]))
    out = []
    for k, v in pairs:
        try:
            odd = float(str(v).replace(",", "."))
        except ValueError:
            raise ValueError(f"Некорректный коэффициент для {k!r}: {v!r}") from None
        if odd <= 1.0:
            raise ValueError(f"Коэффициент {k!r}={odd} должен быть десятичным и > 1.0")
        out.append((str(k), odd))
    return out


def _num(s: str) -> str:
    v = float(s.replace(",", "."))
    return f"{v:+}" if s.strip()[0] in "+-" else f"{v}"


def normalize_key(raw: str, team1: str, team2: str) -> str:
    k = raw.strip().lower()
    t1, t2 = team1.lower(), team2.lower()
    # Названия команд → team1/team2 (длинное имя заменяем первым).
    # Только целым словом и не трогая уже канонические team1/team2 (команда «A» не должна
    # портить «team1»).
    for name, tag in sorted(((t1, "team1"), (t2, "team2")), key=lambda x: -len(x[0])):
        if name and name not in ("team1", "team2"):
            k = re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", tag, k)
    k = re.sub(r"\s+", " ", k)
    simple = {
        "п1": "ml:team1", "п2": "ml:team2", "w1": "ml:team1", "w2": "ml:team2",
        "1": "ml:team1", "2": "ml:team2", "team1": "ml:team1", "team2": "ml:team2",
        "ml team1": "ml:team1", "ml team2": "ml:team2",
    }
    if k in simple:
        return simple[k]
    k = k.replace(" ", ":")
    k = k.replace("тб", "over").replace("тм", "under").replace("ф1", "handicap:team1").replace("ф2", "handicap:team2")
    k = re.sub(r"(?<![a-z])o:", "over:", k)
    k = re.sub(r"(?<![a-z])u:", "under:", k)
    k = re.sub(r"(?:^|:)(?:winner|win)(?=:|$)", "", k).strip(":")
    # фора без знака не допускается — оставим как есть, ниже валидация
    m = re.match(r"^(map\d):(team[12])$", k)
    if m:
        return k
    # "map1:over:21.5" → тотал раундов
    m = re.match(r"^(map\d):(over|under):([\d.,]+)$", k)
    if m:
        return f"{m.group(1)}:total_rounds:{m.group(2)}:{_num(m.group(3))}"
    m = re.match(r"^(map\d):total_rounds:(over|under):([\d.,]+)$", k)
    if m:
        return f"{m.group(1)}:total_rounds:{m.group(2)}:{_num(m.group(3))}"
    m = re.match(r"^(map\d):handicap:(team[12]):([+-][\d.,]+)$", k)
    if m:
        return f"{m.group(1)}:handicap:{m.group(2)}:{_num(m.group(3))}"
    m = re.match(r"^handicap:(team[12]):([+-][\d.,]+)$", k)
    if m:
        return f"handicap:{m.group(1)}:{_num(m.group(2))}"
    m = re.match(r"^(?:total_maps|maps):(over|under):([\d.,]+)$", k)
    if m:
        return f"total_maps:{m.group(1)}:{_num(m.group(2))}"
    m = re.match(r"^(?:over|under):([\d.,]+)$", k)
    if m:
        return f"total_maps:{k.split(':')[0]}:{_num(m.group(1))}"
    m = re.match(r"^score:(\d)[-:](\d)$", k)
    if m:
        return f"score:{m.group(1)}-{m.group(2)}"
    return k


def parse_odds(arg: str, team1: str, team2: str) -> list[OddsLine]:
    return [OddsLine(raw, normalize_key(raw, team1, team2), o) for raw, o in _parse_raw(load_odds_text(arg))]


def complement(key: str) -> list[str]:
    """Ключи, образующие с данным полную группу исходов (для снятия маржи)."""
    swap = {"team1": "team2", "team2": "team1", "over": "under", "under": "over", "yes": "no", "no": "yes"}
    parts = key.split(":")
    if parts[0] == "score":
        return []  # группа счёта обрабатывается отдельно
    if "handicap" in parts:
        i = parts.index("handicap")
        team, line = parts[i + 1], parts[i + 2]
        other = swap[team]
        flipped = line.replace("+", "#").replace("-", "+").replace("#", "-")
        return [":".join(parts[:i] + ["handicap", other, flipped])]
    for i, p in enumerate(parts):
        if p in swap:
            return [":".join(parts[:i] + [swap[p]] + parts[i + 1:])]
    return []


def assess(lines: list[OddsLine], events: dict[str, float], ranges: dict[str, tuple[float, float]],
           labels: dict[str, str], min_edge: float = 0.03, kelly_fraction: float = 0.25,
           kelly_cap: float = 0.02) -> tuple[list[Assessment], list[str]]:
    by_key = {l.key: l for l in lines}
    unknown = [f"{l.raw_key} → {l.key}" for l in lines if l.key not in events]
    score_keys = [k for k in by_key if k.startswith("score:") and k in events]
    score_total = sum(1 / by_key[k].odds for k in score_keys)
    all_scores = [k for k in events if k.startswith("score:")]
    out: list[Assessment] = []
    for l in lines:
        if l.key not in events:
            continue
        p = events[l.key]
        lo, hi = ranges.get(l.key, (p, p))
        implied = 1 / l.odds
        fair = margin = None
        comp = [c for c in complement(l.key) if c in by_key]
        if comp:
            tot = implied + sum(1 / by_key[c].odds for c in comp)
            fair, margin = implied / tot, tot - 1
        elif l.key.startswith("score:") and set(all_scores) <= set(score_keys):
            fair, margin = implied / score_total, score_total - 1
        ev = p * l.odds - 1
        ev_low = lo * l.odds - 1
        b = l.odds - 1
        kelly = max(0.0, (p * l.odds - 1) / b) * kelly_fraction if b > 0 else 0.0
        kelly = min(kelly, kelly_cap)
        if ev >= min_edge and ev_low > 0:
            verdict = "перевес (value)"
        elif ev >= min_edge:
            verdict = "слабый перевес, высокая неопределённость"
            kelly = min(kelly, kelly_cap / 2)
        elif ev > -0.02:
            verdict = "около справедливого"
            kelly = 0.0
        else:
            verdict = "нет перевеса"
            kelly = 0.0
        out.append(Assessment(l.key, labels.get(l.key, l.key), l.odds, p, lo, hi, implied, fair, margin,
                              ev, ev_low, kelly, verdict))
    out.sort(key=lambda a: a.ev, reverse=True)
    return out, unknown
