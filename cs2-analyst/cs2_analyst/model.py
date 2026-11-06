"""Вероятностная модель матча.

Сила команды1 относительно команды2 собирается в логит-шкале из компонент:

=====================  ======================================================
рейтинг                k·(ln rank2 − ln rank1), либо очки HLTV
форма (90 дней)        результат минус ожидание с учётом силы соперника,
                       экспоненциальное затухание (полураспад 30 дней)
игроки                 средний HLTV Rating пятёрки
H2H                    малый вес, сильное сжатие к 50 %
LAN/онлайн             winrate в данном формате, сжатый к 50 %
состав                 штраф за замену/стендина и недавнюю смену состава
карта                  winrate по карте со сжатием (Байес, prior n=5)
=====================  ======================================================

Логит карты = (0.75·Σ матчевых компонент + компонента карты) × коэффициент полноты.
Итоговый логит умножается на коэффициент «полноты данных» (0.6–1.0): чем
меньше данных, тем ближе прогноз к 50 % — это честная реакция на неопределённость.
Коэффициенты подобраны экспертно и не откалиброваны на истории — см. README.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from .models import NO_DATA, Match, MapStat, Team
from .probability import calibrate_rounds, log5, logistic, logit, map_round_distribution, series_distribution

# Пул карт по умолчанию (Active Duty может измениться — передавайте --map-pool).
DEFAULT_MAP_POOL = ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Train"]

W_RANK = 0.40          # на единицу разницы ln(rank)
W_POINTS = 0.70        # на единицу разницы ln(points)
W_FORM = 1.50
W_PLAYERS = 3.0        # на 1.00 разницы среднего рейтинга
W_H2H = 0.30
W_SETTING = 0.40
W_MAP = 0.80
STANDIN_PENALTY = 0.35
NEW_ROSTER_PENALTY = 0.15
FORM_WINDOW_DAYS = 90
FORM_HALF_LIFE = 30.0
H2H_WINDOW_DAYS = 365
UNRANKED_RANK = 45     # «вне рейтинга»: условная позиция, отмечается в отчёте
MAP_PRIOR_N = 5.0
# Компоненты выше описывают силу на уровне матча; на одну карту переносим с коэффициентом
# < 1, т.к. серия сама усиливает перевес фаворита (BO3 из карт по 63 % ≈ 70 %).
MAP_SCALE = 0.75


@dataclass
class Component:
    name: str
    value: float               # вклад в логит в пользу команды1
    detail: str
    available: bool = True


@dataclass
class MapPrediction:
    name: str
    p_team1: float
    picked_by: int | None
    source: str                 # "вето HLTV" / "прогноз вето"
    components: list[Component]
    r_ct: float
    r_t: float
    rounds: dict[tuple[int, int], float]
    pistol_team1: float | None


@dataclass
class Prediction:
    match: Match
    best_of: int
    best_of_known: bool
    base_components: list[Component]
    completeness: float
    shrink: float
    confidence: str
    maps: list[MapPrediction]
    series: dict[tuple[int, int], float]
    p_match: float
    p_match_range: tuple[float, float]
    events: dict[str, float] = field(default_factory=dict)          # канонический ключ → p
    event_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ helpers
def _shrunk(rate: float, n: float, prior_n: float) -> float:
    return (rate * n + 0.5 * prior_n) / (n + prior_n)


def _rank_of(team: Team) -> tuple[int, bool]:
    return (team.rank, True) if team.rank else (UNRANKED_RANK, False)


def _team_strength_logit(team: Team) -> float:
    """Абсолютная «сила» по рейтингу (для ожидания в форме)."""
    rank, _ = _rank_of(team)
    return -W_RANK * math.log(rank)


def _form(team: Team, today: date) -> tuple[float | None, float, str]:
    """(остаток формы −1..1, эффективное число матчей, пояснение)."""
    own = _team_strength_logit(team)
    num = den = 0.0
    n = 0
    unknown_opp = 0
    for m in team.recent_matches:
        if m.date is None:
            continue
        age = (today - m.date).days
        if age < 0 or age > FORM_WINDOW_DAYS:
            continue
        w = 0.5 ** (age / FORM_HALF_LIFE)
        opp_rank = m.opponent_rank or UNRANKED_RANK
        if m.opponent_rank is None:
            unknown_opp += 1
        expected = logistic(own - (-W_RANK * math.log(opp_rank)))
        num += w * ((1.0 if m.won else 0.0) - expected)
        den += w
        n += 1
    if n == 0:
        return None, 0.0, NO_DATA
    resid = num / den
    wins = sum(1 for m in team.recent_matches if m.date and 0 <= (today - m.date).days <= FORM_WINDOW_DAYS and m.won)
    note = f"{wins}-{n - wins} за {FORM_WINDOW_DAYS} дн., отклонение от ожидания {resid:+.2f}"
    if unknown_opp:
        note += f" (ранг соперника неизвестен в {unknown_opp} матч.)"
    return resid, float(n), note


def _avg_rating(team: Team) -> tuple[float | None, int]:
    vals = [p.rating for p in team.players if p.rating is not None]
    if len(vals) < 3:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def _setting_wr(team: Team, lan: bool) -> tuple[float | None, int]:
    if lan:
        return team.lan_winrate, team.lan_matches or 0
    return team.online_winrate, team.online_matches or 0


# --------------------------------------------------------------- components
def base_components(match: Match, today: date) -> list[Component]:
    t1, t2 = match.team1, match.team2
    comps: list[Component] = []

    # 1) Рейтинг / очки
    if t1.points and t2.points:
        v = W_POINTS * (math.log(t1.points) - math.log(t2.points))
        comps.append(Component("Рейтинг HLTV (очки)", v, f"#{t1.rank or '?'} {t1.points} очк. vs #{t2.rank or '?'} {t2.points} очк."))
    else:
        r1, k1 = _rank_of(t1)
        r2, k2 = _rank_of(t2)
        v = W_RANK * (math.log(r2) - math.log(r1))
        detail = f"#{t1.rank if k1 else 'вне рейтинга'} vs #{t2.rank if k2 else 'вне рейтинга'}"
        comps.append(Component("Рейтинг HLTV", v, detail, available=k1 and k2))

    # 2) Форма
    f1, n1, d1 = _form(t1, today)
    f2, n2, d2 = _form(t2, today)
    if f1 is not None and f2 is not None:
        s1, s2 = n1 / (n1 + 3), n2 / (n2 + 3)
        v = W_FORM * (f1 * s1 - f2 * s2)
        comps.append(Component("Форма (3 мес., с затуханием)", v, f"{t1.name}: {d1}; {t2.name}: {d2}"))
    else:
        comps.append(Component("Форма (3 мес., с затуханием)", 0.0, f"{t1.name}: {d1}; {t2.name}: {d2}", False))

    # 3) Игроки
    a1, c1 = _avg_rating(t1)
    a2, c2 = _avg_rating(t2)
    if a1 is not None and a2 is not None:
        comps.append(Component("Рейтинг игроков (avg)", W_PLAYERS * (a1 - a2), f"{a1:.2f} vs {a2:.2f}"))
    else:
        comps.append(Component("Рейтинг игроков (avg)", 0.0, NO_DATA, False))

    # 4) H2H
    recent = [h for h in match.h2h if h.date is None or 0 <= (today - h.date).days <= H2H_WINDOW_DAYS]
    if recent:
        w1 = sum(1 for h in recent if h.winner == 1)
        share = (w1 + 2) / (len(recent) + 4)
        comps.append(Component("H2H (12 мес., малый вес)", W_H2H * logit(share), f"{w1}-{len(recent) - w1}"))
    else:
        comps.append(Component("H2H (12 мес., малый вес)", 0.0, f"{NO_DATA} или не встречались", False))

    # 5) LAN / онлайн
    if match.lan is None:
        comps.append(Component("LAN/онлайн", 0.0, f"формат {NO_DATA}", False))
    else:
        label = "LAN" if match.lan else "онлайн"
        w1r, n1r = _setting_wr(t1, match.lan)
        w2r, n2r = _setting_wr(t2, match.lan)
        if w1r is not None and w2r is not None:
            v = W_SETTING * (logit(_shrunk(w1r, n1r, 8)) - logit(_shrunk(w2r, n2r, 8)))
            comps.append(Component(f"Winrate на {label}", v, f"{w1r:.0%} (n={n1r}) vs {w2r:.0%} (n={n2r})"))
        else:
            comps.append(Component(f"Winrate на {label}", 0.0, NO_DATA, False))

    # 6) Состав
    v = 0.0
    parts = []
    for sign, t in ((1, t1), (-1, t2)):
        st = sum(1 for p in t.players if p.is_standin)
        if st:
            v -= sign * STANDIN_PENALTY * st
            parts.append(f"{t.name}: стендин ×{st}")
        if t.days_since_roster_change is not None and t.days_since_roster_change < 30:
            v -= sign * NEW_ROSTER_PENALTY
            parts.append(f"{t.name}: смена состава {t.days_since_roster_change} дн. назад")
    known = bool(t1.players and t2.players)
    comps.append(Component("Состав / замены", v, "; ".join(parts) or ("без изменений" if known else NO_DATA), known))
    return comps


def map_component(match: Match, map_name: str) -> Component:
    m1 = match.team1.maps.get(map_name)
    m2 = match.team2.maps.get(map_name)

    def s(m: MapStat | None) -> tuple[float, str] | None:
        if m is None or m.winrate is None:
            return None
        n = m.played or 0
        return _shrunk(m.winrate, n, MAP_PRIOR_N), f"{m.winrate:.0%} (n={n})"

    a, b = s(m1), s(m2)
    if a is None or b is None:
        return Component(f"Карта {map_name}", 0.0, f"{a[1] if a else NO_DATA} vs {b[1] if b else NO_DATA}", False)
    return Component(f"Карта {map_name}", W_MAP * (logit(a[0]) - logit(b[0])), f"{a[1]} vs {b[1]}")


# --------------------------------------------------------------------- veto
def _veto_sequence(best_of: int, pool_size: int) -> list[tuple[str, int]]:
    """Порядок действий вето: ("ban"/"pick", команда). Команда1 начинает."""
    if pool_size != 7:
        # Обобщённый вариант: баны по очереди до нужного числа карт.
        seq = []
        picks = best_of - 1
        bans = pool_size - best_of
        for i in range(bans):
            seq.append(("ban", 1 + i % 2))
        for i in range(picks):
            seq.append(("pick", 1 + i % 2))
        return seq
    if best_of == 1:
        return [("ban", 1), ("ban", 2), ("ban", 1), ("ban", 2), ("ban", 1), ("ban", 2)]
    if best_of == 3:
        return [("ban", 1), ("ban", 2), ("pick", 1), ("pick", 2), ("ban", 1), ("ban", 2)]
    if best_of == 5:
        return [("ban", 1), ("ban", 2), ("pick", 1), ("pick", 2), ("pick", 1), ("pick", 2)]
    raise ValueError(f"Неподдерживаемый формат BO{best_of}")


def predict_veto(match: Match, pool: list[str], map_logit: dict[str, float], best_of: int) -> list[tuple[str, int | None]]:
    """Жадный прогноз вето: команда пикает карту, где она сильнее, и банит, где слабее.

    Если есть pick/ban-rate, привычки команды важнее модели (вес 1.0 против 0.3).
    Возвращает карты в порядке игры: пики, затем десайдер.
    """
    remaining = list(pool)
    picks: list[tuple[str, int | None]] = []
    for action, team_idx in _veto_sequence(best_of, len(pool)):
        team = match.team1 if team_idx == 1 else match.team2
        sign = 1 if team_idx == 1 else -1

        def score(m: str) -> float:
            pref = sign * map_logit.get(m, 0.0)        # >0 — карта выгодна этой команде
            ms = team.maps.get(m)
            if action == "pick":
                habit = ms.pick_rate if ms and ms.pick_rate is not None else None
                return (habit if habit is not None else 0.0) + 0.3 * pref
            habit = ms.ban_rate if ms and ms.ban_rate is not None else None
            return (habit if habit is not None else 0.0) - 0.3 * pref

        choice = max(remaining, key=score)
        remaining.remove(choice)
        if action == "pick":
            picks.append((choice, team_idx))
    if remaining:
        picks.append((remaining[0], None))
    return picks[:best_of]


# ---------------------------------------------------------------- analysis
def _completeness(match: Match, comps: list[Component]) -> tuple[float, list[str]]:
    missing: list[str] = []
    weights = {
        "Рейтинг": 0.25, "Форма": 0.2, "Рейтинг игроков": 0.15, "Карты": 0.2,
        "LAN/онлайн": 0.05, "Формат серии": 0.05, "Состав": 0.05, "Стороны CT/T": 0.05,
    }
    score = 0.0
    for c in comps:
        key = next((k for k in sorted(weights, key=len, reverse=True) if c.name.startswith(k)), None)
        if c.name.startswith("Winrate на"):
            key = "LAN/онлайн"
        if c.name.startswith("H2H"):
            if not c.available:
                missing.append("H2H")
            continue
        if key is None:
            continue
        if c.available:
            score += weights[key]
        else:
            missing.append(c.name)
    pool_maps = set(match.team1.maps) & set(match.team2.maps)
    if pool_maps and any(match.team1.maps[m].winrate is not None for m in pool_maps):
        score += weights["Карты"]
    else:
        missing.append("Статистика карт")
    if any(ms.ct_round_win is not None for ms in match.team1.maps.values()) and any(
        ms.t_round_win is not None for ms in match.team2.maps.values()
    ):
        score += weights["Стороны CT/T"]
    else:
        missing.append("Раунды CT/T по картам")
    if match.best_of:
        score += weights["Формат серии"]
    else:
        missing.append("Формат матча (BO1/BO3/BO5)")
    if not any(ms.pick_rate is not None or ms.ban_rate is not None for t in (match.team1, match.team2) for ms in t.maps.values()):
        missing.append("Pick/ban-статистика")
    if match.team1.pistol_winrate is None or match.team2.pistol_winrate is None:
        missing.append("Пистолетные раунды")
    return min(score, 1.0), missing


def _side_params(match: Match, map_name: str) -> tuple[float, float, bool]:
    """(общий CT-перекос карты, CT-преимущество команды1 относительно её T, есть ли данные)."""
    m1 = match.team1.maps.get(map_name)
    m2 = match.team2.maps.get(map_name)
    if not (m1 and m2 and None not in (m1.ct_round_win, m1.t_round_win, m2.ct_round_win, m2.t_round_win)):
        return 0.0, 0.0, False
    r_ct = log5(m1.ct_round_win, m2.t_round_win)   # команда1 за CT
    r_t = log5(m1.t_round_win, m2.ct_round_win)    # команда1 за T
    # Общий перекос стороны по карте: среднее CT-раундов обеих команд против их T.
    bias = (logit(m1.ct_round_win) + logit(m2.ct_round_win) - logit(m1.t_round_win) - logit(m2.t_round_win)) / 4
    edge = logit(r_ct) - logit(r_t) - 2 * bias
    return bias, edge, True


def analyze(match: Match, today: date | None = None, map_pool: list[str] | None = None,
            default_best_of: int = 3) -> Prediction:
    today = today or match.date or date.today()
    warnings: list[str] = []

    best_of_known = match.best_of is not None
    best_of = match.best_of or default_best_of
    if not best_of_known:
        warnings.append(f"Формат матча {NO_DATA}: принят BO{best_of} (укажите --bo).")

    comps = base_components(match, today)
    completeness, missing = _completeness(match, comps)
    shrink = 0.6 + 0.4 * completeness
    base = sum(c.value for c in comps)

    pool = map_pool or match.map_pool or sorted(set(match.team1.maps) | set(match.team2.maps)) or DEFAULT_MAP_POOL
    if not (map_pool or match.map_pool) and len(pool) != 7:
        warnings.append(f"Пул карт определён по статистике ({len(pool)} карт); при необходимости задайте --map-pool.")
        if len(pool) < best_of:
            pool = DEFAULT_MAP_POOL
    map_comps = {m: map_component(match, m) for m in pool}
    map_logit = {m: c.value for m, c in map_comps.items()}

    # Карты серии: известное вето или прогноз.
    if match.veto:
        order = [(v.name, v.picked_by) for v in match.veto][:best_of]
        src = "вето HLTV"
        if len(order) < best_of:
            known = {n for n, _ in order}
            rest = predict_veto(match, [m for m in pool if m not in known], map_logit, best_of)
            order += [x for x in rest if x[0] not in known][: best_of - len(order)]
            src = "вето HLTV (частично) + прогноз"
    else:
        order = predict_veto(match, pool, map_logit, best_of)
        src = "прогноз вето"
        warnings.append("Вето ещё не опубликовано: карты спрогнозированы, вероятности по картам менее надёжны.")

    maps: list[MapPrediction] = []
    for name, picked_by in order:
        mc = map_comps.get(name) or map_component(match, name)
        total = (MAP_SCALE * base + mc.value) * shrink
        p = logistic(total)
        bias, edge, has_sides = _side_params(match, name)
        r_ct, r_t = calibrate_rounds(p, bias, edge)
        rounds = map_round_distribution(r_ct, r_t)
        pist = None
        if match.team1.pistol_winrate is not None and match.team2.pistol_winrate is not None:
            pist = log5(match.team1.pistol_winrate, match.team2.pistol_winrate)
        maps.append(MapPrediction(name, p, picked_by, src, [*comps, mc], r_ct, r_t, rounds, pist))

    series = series_distribution([m.p_team1 for m in maps], best_of)
    need = best_of // 2 + 1
    p_match = sum(v for (w, l), v in series.items() if w == need)

    conf = "высокая" if completeness >= 0.8 else "средняя" if completeness >= 0.55 else "низкая"
    if not match.veto and conf == "высокая":
        conf = "средняя"

    pred = Prediction(match, best_of, best_of_known, comps, completeness, shrink, conf, maps, series,
                      p_match, (0, 0), missing=missing, warnings=warnings)
    _fill_events(pred)
    _fill_ranges(pred, base, map_logit, order)
    return pred


# ---------------------------------------------------------------- events
TOTAL_LINES = (19.5, 20.5, 21.5, 22.5, 23.5)
ROUND_HCP = (2.5, 3.5, 4.5, 5.5)


def _events_from(map_probs: list[float], maps: list[MapPrediction] | None, best_of: int) -> dict[str, float]:
    ev: dict[str, float] = {}
    series = series_distribution(map_probs, best_of)
    need = best_of // 2 + 1
    p1 = sum(v for (w, l), v in series.items() if w == need)
    ev["ml:team1"] = p1
    ev["ml:team2"] = 1 - p1
    for (w, l), v in series.items():
        ev[f"score:{w}-{l}"] = v
    if best_of > 1:
        # Форы по картам
        for k in range(1, need):
            line = k + 0.5
            # team1 -line: выигрывает с разницей > line
            ev[f"handicap:team1:-{line}"] = sum(v for (w, l), v in series.items() if w - l > line)
            ev[f"handicap:team2:+{line}"] = 1 - ev[f"handicap:team1:-{line}"]
            ev[f"handicap:team2:-{line}"] = sum(v for (w, l), v in series.items() if l - w > line)
            ev[f"handicap:team1:+{line}"] = 1 - ev[f"handicap:team2:-{line}"]
        # Тотал карт
        for line in [x + 0.5 for x in range(need, best_of)]:
            ev[f"total_maps:over:{line}"] = sum(v for (w, l), v in series.items() if w + l > line)
            ev[f"total_maps:under:{line}"] = 1 - ev[f"total_maps:over:{line}"]
    for i, p in enumerate(map_probs[:best_of], start=1):
        ev[f"map{i}:team1"] = p
        ev[f"map{i}:team2"] = 1 - p
    if maps:
        for i, m in enumerate(maps, start=1):
            r = m.rounds
            for line in TOTAL_LINES:
                ov = sum(v for (a, b), v in r.items() if a + b > line)
                ev[f"map{i}:total_rounds:over:{line}"] = ov
                ev[f"map{i}:total_rounds:under:{line}"] = 1 - ov
            for line in ROUND_HCP:
                ev[f"map{i}:handicap:team1:-{line}"] = sum(v for (a, b), v in r.items() if a - b > line)
                ev[f"map{i}:handicap:team2:+{line}"] = 1 - ev[f"map{i}:handicap:team1:-{line}"]
                ev[f"map{i}:handicap:team2:-{line}"] = sum(v for (a, b), v in r.items() if b - a > line)
                ev[f"map{i}:handicap:team1:+{line}"] = 1 - ev[f"map{i}:handicap:team2:-{line}"]
            ot = sum(v for (a, b), v in r.items() if a + b > 24)
            ev[f"map{i}:overtime:yes"] = ot
            ev[f"map{i}:overtime:no"] = 1 - ot
            if m.pistol_team1 is not None:
                ev[f"map{i}:pistol1:team1"] = m.pistol_team1
                ev[f"map{i}:pistol1:team2"] = 1 - m.pistol_team1
    return ev


def _fill_events(pred: Prediction) -> None:
    pred.events = _events_from([m.p_team1 for m in pred.maps], pred.maps, pred.best_of)


def _fill_ranges(pred: Prediction, base: float, map_logit: dict[str, float], order) -> None:
    """Диапазон неопределённости: сдвиг логита на ±σ, σ растёт при нехватке данных."""
    sigma = 0.25 + 0.6 * (1 - pred.completeness)
    if not pred.match.veto:
        sigma += 0.1
    lows, highs = [], []
    for m in pred.maps:
        L = logit(m.p_team1)
        lows.append(logistic(L - sigma))
        highs.append(logistic(L + sigma))
    ev_lo = _events_from(lows, None, pred.best_of)
    ev_hi = _events_from(highs, None, pred.best_of)
    for k, p in pred.events.items():
        if k in ev_lo:
            # Немонотонные исходы (например, счёт 2:1) не расширяются сдвигом силы,
            # поэтому берём ещё и симметричную полосу вокруг p.
            a, b = ev_lo[k], ev_hi[k]
            L = logit(p)
            pred.event_ranges[k] = (min(a, b, p, logistic(L - 0.5 * sigma)),
                                    max(a, b, p, logistic(L + 0.5 * sigma)))
        else:
            # Раундовые рынки (тоталы, форы, OT): тот же сдвиг логита с коэффициентом 0.8.
            L = logit(p)
            pred.event_ranges[k] = (logistic(L - 0.8 * sigma), logistic(L + 0.8 * sigma))
    pred.p_match_range = pred.event_ranges["ml:team1"]
