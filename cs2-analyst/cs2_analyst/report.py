"""Формирование отчёта (Markdown / текст) и JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from .model import Prediction
from .models import NO_DATA
from .odds import Assessment

DISCLAIMER = (
    "⚠️ Это оценка вероятностей, а не гарантия. В CS2 высокая дисперсия: даже исход с "
    "вероятностью 70 % не сбывается примерно в каждом третьем матче. Модель не откалибрована "
    "на исторических коэффициентах; ставьте только то, что готовы потерять."
)


def event_label(key: str, t1: str, t2: str, map_names: list[str]) -> str:
    def team(tag: str) -> str:
        return t1 if tag == "team1" else t2

    p = key.split(":")
    if p[0] == "ml":
        return f"Победа {team(p[1])} (матч)"
    if p[0] == "score":
        a, b = p[1].split("-")
        return f"Счёт {t1} {a}:{b} {t2}"
    if p[0] == "handicap":
        return f"Фора {team(p[1])} {p[2]} по картам"
    if p[0] == "total_maps":
        return f"Тотал карт {'больше' if p[1] == 'over' else 'меньше'} {p[2]}"
    m = re.match(r"map(\d)", p[0])
    if m:
        i = int(m.group(1))
        mn = map_names[i - 1] if i - 1 < len(map_names) else "?"
        pref = f"Карта {i} ({mn}):"
        if len(p) == 2:
            return f"{pref} победа {team(p[1])}"
        if p[1] == "total_rounds":
            return f"{pref} тотал раундов {'больше' if p[2] == 'over' else 'меньше'} {p[3]}"
        if p[1] == "handicap":
            return f"{pref} фора {team(p[2])} {p[3]} по раундам"
        if p[1] == "overtime":
            return f"{pref} овертайм — {'да' if p[2] == 'yes' else 'нет'}"
        if p[1] == "pistol1":
            return f"{pref} первый пистолет — {team(p[2])}"
    return key


def labels_for(pred: Prediction) -> dict[str, str]:
    t1, t2 = pred.match.team1.name, pred.match.team2.name
    names = [m.name for m in pred.maps]
    return {k: event_label(k, t1, t2, names) for k in pred.events}


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _fair(p: float) -> str:
    return f"{1 / p:.2f}" if p > 0.001 else "—"


_MAIN_MARKETS = re.compile(
    r"^(ml:|score:|handicap:|total_maps:|map\d:team[12]$|map\d:total_rounds:(over|under):21\.5$|"
    r"map\d:handicap:team[12]:-3\.5$|map\d:pistol1:)"
)


def _most_likely(pred: Prediction, n: int = 10) -> list[tuple[str, float]]:
    """Самые вероятные исходы среди основных рынков (без почти гарантированных форм > 97 %)."""
    labels = labels_for(pred)
    cand = [(k, p) for k, p in pred.events.items() if _MAIN_MARKETS.match(k) and p <= 0.97]
    cand.sort(key=lambda x: x[1], reverse=True)
    return [(labels[k], p) for k, p in cand[:n]]


def render_markdown(pred: Prediction, assessments: list[Assessment] | None, unknown_odds: list[str],
                    notes: list[str]) -> str:
    m = pred.match
    t1, t2 = m.team1.name, m.team2.name
    L: list[str] = []
    L.append(f"# {t1} vs {t2}")
    L.append("")
    fmt = f"BO{pred.best_of}" + ("" if pred.best_of_known else f" (предположение, {NO_DATA})")
    lan = NO_DATA if m.lan is None else ("LAN" if m.lan else "онлайн")
    L.append(f"- **Турнир:** {m.event or NO_DATA}")
    L.append(f"- **Дата:** {m.date.isoformat() if m.date else NO_DATA}")
    L.append(f"- **Формат:** {fmt}, {lan}")
    if m.url:
        L.append(f"- **Источник:** {m.url}")
    L.append(f"- **Уверенность модели:** {pred.confidence} (полнота данных {pred.completeness:.0%}, "
             f"сжатие к 50 %: ×{pred.shrink:.2f})")
    L.append("")
    L.append(f"> {DISCLAIMER}")
    L.append("")

    # Итог
    lo, hi = pred.p_match_range
    fav = t1 if pred.p_match >= 0.5 else t2
    pf = max(pred.p_match, 1 - pred.p_match)
    L.append("## Победа в матче")
    L.append("")
    L.append("| Команда | Вероятность | Диапазон | Справедливый кф |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| {t1} | {_pct(pred.p_match)} | {_pct(lo)}–{_pct(hi)} | {_fair(pred.p_match)} |")
    L.append(f"| {t2} | {_pct(1 - pred.p_match)} | {_pct(1 - hi)}–{_pct(1 - lo)} | {_fair(1 - pred.p_match)} |")
    L.append("")
    L.append(f"Фаворит модели: **{fav}** ({_pct(pf)}).")
    L.append("")

    # Компоненты
    L.append("## Из чего складывается оценка")
    L.append("")
    L.append(f"Вклад в логит в пользу {t1} (положительное — за {t1}, отрицательное — за {t2}):")
    L.append("")
    L.append("| Фактор | Вклад | Данные |")
    L.append("|---|---:|---|")
    for c in pred.base_components:
        v = f"{c.value:+.2f}" if c.available or c.value else "0 " + NO_DATA
        L.append(f"| {c.name} | {v} | {c.detail} |")
    L.append("")

    # Карты
    L.append("## Карты")
    L.append("")
    src = pred.maps[0].source if pred.maps else ""
    L.append(f"Источник списка карт: **{src}**.")
    L.append("")
    L.append(f"| # | Карта | Пик | Фактор карты | P({t1}) | P({t2}) | OT | P(раунд {t1} CT/T) |")
    L.append("|---|---|---|---|---:|---:|---:|---:|")
    for i, mp in enumerate(pred.maps, start=1):
        pick = {1: t1, 2: t2}.get(mp.picked_by, "десайдер" if i == len(pred.maps) else "—")
        mc = mp.components[-1]
        ot = pred.events.get(f"map{i}:overtime:yes", 0)
        L.append(f"| {i} | {mp.name} | {pick} | {mc.detail} | {_pct(mp.p_team1)} | {_pct(1 - mp.p_team1)} | "
                 f"{_pct(ot)} | {mp.r_ct:.3f} / {mp.r_t:.3f} |")
    L.append("")

    # Серия
    if pred.best_of > 1:
        L.append("## Счёт серии, форы и тотал карт")
        L.append("")
        L.append("| Исход | Вероятность | Справедливый кф |")
        L.append("|---|---:|---:|")
        labels = labels_for(pred)
        keys = sorted([k for k in pred.events if k.startswith("score:")], key=lambda k: -pred.events[k])
        keys += [k for k in pred.events if k.startswith("handicap:") or k.startswith("total_maps:")]
        for k in keys:
            p = pred.events[k]
            L.append(f"| {labels[k]} | {_pct(p)} | {_fair(p)} |")
        L.append("")

    # Раунды по картам
    L.append("## Раунды по картам")
    L.append("")
    for i, mp in enumerate(pred.maps, start=1):
        L.append(f"**Карта {i}: {mp.name}**")
        L.append("")
        L.append("| Линия | Больше | Меньше |")
        L.append("|---|---:|---:|")
        for line in (19.5, 20.5, 21.5, 22.5, 23.5):
            ov = pred.events[f"map{i}:total_rounds:over:{line}"]
            L.append(f"| Тотал раундов {line} | {_pct(ov)} | {_pct(1 - ov)} |")
        for line in (3.5, 4.5):
            a = pred.events[f"map{i}:handicap:team1:-{line}"]
            b = pred.events[f"map{i}:handicap:team2:-{line}"]
            L.append(f"| Фора −{line}: {t1} / {t2} | {_pct(a)} | {_pct(b)} |")
        pist = pred.events.get(f"map{i}:pistol1:team1")
        L.append(f"| Первый пистолет {t1} | {_pct(pist) if pist is not None else NO_DATA} | |")
        L.append("")

    # Наиболее вероятные
    L.append("## Наиболее вероятные исходы")
    L.append("")
    L.append("Высокая вероятность ≠ выгодная ставка: смотрите сравнение с коэффициентами ниже.")
    L.append("")
    L.append("| Исход | Вероятность | Справедливый кф |")
    L.append("|---|---:|---:|")
    for label, p in _most_likely(pred):
        L.append(f"| {label} | {_pct(p)} | {_fair(p)} |")
    L.append("")

    # Коэффициенты
    L.append("## Сравнение с коэффициентами")
    L.append("")
    if assessments is None:
        L.append("Коэффициенты не переданы (`--odds`). Сравните справедливые кф выше с линией букмекера: "
                 "ставка имеет смысл, только если предложенный кф заметно выше справедливого.")
    elif not assessments:
        L.append("Ни один из переданных коэффициентов не сопоставлен с событиями модели.")
    else:
        L.append("| Событие | Кф | P модели (диапазон) | P рынка без маржи | EV | EV (пессим.) | Ставка, % банка | Вывод |")
        L.append("|---|---:|---|---:|---:|---:|---:|---|")
        for a in assessments:
            fair = _pct(a.fair_implied) if a.fair_implied is not None else f"{_pct(a.implied)}*"
            L.append(f"| {a.label} | {a.odds:.2f} | {_pct(a.p_model)} ({_pct(a.p_low)}–{_pct(a.p_high)}) | {fair} | "
                     f"{a.ev:+.1%} | {a.ev_low:+.1%} | {a.kelly * 100:.1f} | {a.verdict} |")
        L.append("")
        L.append("\\* — пара исхода не передана, маржа не снята. EV = P·кф − 1. Размер ставки — ¼ Келли, "
                 "не больше 2 % банка (1 % при высокой неопределённости).")
        margins = {round(a.margin, 4) for a in assessments if a.margin is not None}
        if margins:
            L.append(f"Маржа букмекера по найденным парам: {', '.join(f'{x:.1%}' for x in sorted(margins))}.")
    if unknown_odds:
        L.append("")
        L.append("Не распознаны: " + "; ".join(unknown_odds))
    L.append("")

    # Рекомендация
    L.append("## Рекомендация")
    L.append("")
    L.extend(_recommendation(pred, assessments))
    L.append("")

    # Данные
    L.append("## Качество данных")
    L.append("")
    if pred.missing:
        L.append("Отсутствует (" + NO_DATA + "): " + ", ".join(pred.missing) + ".")
    else:
        L.append("Все основные группы данных присутствуют.")
    for w in pred.warnings:
        L.append(f"- {w}")
    for n in notes:
        L.append(f"- {n}")
    L.append("")
    return "\n".join(L)


def _recommendation(pred: Prediction, assessments: list[Assessment] | None) -> list[str]:
    out = []
    t1, t2 = pred.match.team1.name, pred.match.team2.name
    pf = max(pred.p_match, 1 - pred.p_match)
    fav = t1 if pred.p_match >= 0.5 else t2
    if pf < 0.55:
        out.append(f"- Матч близок к 50/50 ({_pct(pf)} у {fav}): исход ML практически непредсказуем.")
    else:
        out.append(f"- Модель склоняется к **{fav}** ({_pct(pf)}), справедливый кф ≈ {1 / pf:.2f}.")
    if pred.confidence == "низкая":
        out.append("- Уверенность **низкая**: данных мало, рекомендуется воздержаться или дополнить данные (`--data`).")
    if assessments:
        value = [a for a in assessments if a.verdict == "перевес (value)"]
        weak = [a for a in assessments if a.verdict.startswith("слабый")]
        if value:
            out.append("- Ставки с перевесом даже при пессимистичной оценке:")
            for a in value[:5]:
                out.append(f"  - {a.label} @ {a.odds:.2f}: P={_pct(a.p_model)}, EV {a.ev:+.1%}, "
                           f"до {a.kelly * 100:.1f}% банка")
        if weak:
            out.append("- Возможный перевес, но в пределах неопределённости модели (только малой суммой или пропустить):")
            for a in weak[:5]:
                out.append(f"  - {a.label} @ {a.odds:.2f}: P={_pct(a.p_model)}, EV {a.ev:+.1%}, пессим. {a.ev_low:+.1%}")
        if not value and not weak:
            out.append("- Ни один переданный коэффициент не даёт перевеса — лучшая ставка здесь: **никакой**.")
        likely = sorted(assessments, key=lambda a: -a.p_model)[:3]
        out.append("- Наибольший шанс сбыться среди переданных линий: " +
                   "; ".join(f"{a.label} @ {a.odds:.2f} ({_pct(a.p_model)})" for a in likely) + ".")
    out.append("- Гарантированных ставок не бывает; оценка не учитывает инсайды, мотивацию и форму в день матча.")
    return out


def to_json(pred: Prediction, assessments: list[Assessment] | None, unknown: list[str], notes: list[str]) -> str:
    labels = labels_for(pred)
    data: dict[str, Any] = {
        "match": pred.match.to_dict(),
        "best_of": pred.best_of,
        "confidence": pred.confidence,
        "completeness": round(pred.completeness, 3),
        "p_team1_match": round(pred.p_match, 4),
        "p_team1_match_range": [round(x, 4) for x in pred.p_match_range],
        "components": [c.__dict__ for c in pred.base_components],
        "maps": [{"name": m.name, "picked_by": m.picked_by, "source": m.source, "p_team1": round(m.p_team1, 4),
                  "r_ct": round(m.r_ct, 4), "r_t": round(m.r_t, 4)} for m in pred.maps],
        "events": {k: {"label": labels[k], "p": round(v, 4), "range": [round(x, 4) for x in pred.event_ranges[k]],
                       "fair_odds": round(1 / v, 3) if v > 0.001 else None} for k, v in pred.events.items()},
        "odds_assessment": [a.__dict__ for a in assessments] if assessments is not None else None,
        "unrecognized_odds": unknown,
        "missing": pred.missing,
        "warnings": pred.warnings + notes,
        "disclaimer": DISCLAIMER,
    }
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
