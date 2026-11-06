"""Точная математика: серии карт и распределение раундов внутри карты (MR12 + OT MR3)."""

from __future__ import annotations

import math
from functools import lru_cache

REG_HALF = 12          # раундов в половине (MR12)
WIN_REG = 13           # до 13 побед
OT_HALF = 3            # MR3 в овертайме
OT_WIN = 4             # до 4 побед в одном овертайме
MAX_OT = 8             # обрезаем хвост (вероятность > 8 OT пренебрежимо мала)


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def log5(a: float, b: float) -> float:
    """Вероятность, что A с базовой силой a обыграет B с силой b (формула log5)."""
    num = a * (1 - b)
    den = num + b * (1 - a)
    return 0.5 if den == 0 else num / den


# ------------------------------------------------------------------ series
def series_distribution(map_probs: list[float], best_of: int) -> dict[tuple[int, int], float]:
    """Распределение счёта серии при известных вероятностях каждой карты по порядку.

    Карты играются по порядку, серия останавливается, когда у кого-то
    ``best_of // 2 + 1`` побед. Карты считаются независимыми при заданных p.
    """
    need = best_of // 2 + 1
    if len(map_probs) < best_of:
        raise ValueError("Нужна вероятность для каждой возможной карты серии")
    return _series_exact(tuple(map_probs[:best_of]), need)


def _series_exact(probs: tuple[float, ...], need: int) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}

    def rec(i: int, w: int, l: int, pr: float) -> None:
        if w == need or l == need:
            out[(w, l)] = out.get((w, l), 0.0) + pr
            return
        p = probs[i]
        rec(i + 1, w + 1, l, pr * p)
        rec(i + 1, w, l + 1, pr * (1 - p))

    rec(0, 0, 0, 1.0)
    return out


# ------------------------------------------------------------------ rounds
def _half_dist(p: float, n: int) -> list[float]:
    """Биномиальное распределение числа выигранных раундов из n."""
    return [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


@lru_cache(maxsize=4096)
def _map_round_dist_cached(r_first: float, r_second: float, r_ot: float) -> tuple[tuple[tuple[int, int], float], ...]:
    """Распределение итогового счёта карты (раунды команды1, раунды команды2).

    r_first  — вероятность команды1 выиграть раунд в первой половине,
    r_second — во второй, r_ot — в овертайме (среднее по сторонам).
    Учитывается досрочное окончание при 13 победах.
    """
    dist: dict[tuple[int, int], float] = {}
    # Первая половина: 12 раундов, досрочного окончания быть не может.
    first = _half_dist(r_first, REG_HALF)
    for a1, pa in enumerate(first):
        b1 = REG_HALF - a1
        # Вторая половина: играем пораунд до 13 или до 12-12.
        state = {(a1, b1): pa}
        for _ in range(REG_HALF):
            nxt: dict[tuple[int, int], float] = {}
            for (a, b), pr in state.items():
                if a == WIN_REG or b == WIN_REG or (a == 12 and b == 12):
                    nxt[(a, b)] = nxt.get((a, b), 0.0) + pr
                    continue
                nxt[(a + 1, b)] = nxt.get((a + 1, b), 0.0) + pr * r_second
                nxt[(a, b + 1)] = nxt.get((a, b + 1), 0.0) + pr * (1 - r_second)
            state = nxt
        for k, pr in state.items():
            dist[k] = dist.get(k, 0.0) + pr

    tie = dist.pop((12, 12), 0.0)
    if tie > 0:
        # Один овертайм: 6 раундов, до 4 побед; 3-3 → следующий OT.
        ot_final: dict[tuple[int, int], float] = {}
        ot_tie = 0.0
        state = {(0, 0): 1.0}
        for _ in range(2 * OT_HALF):
            nxt = {}
            for (a, b), pr in state.items():
                if a == OT_WIN or b == OT_WIN:
                    nxt[(a, b)] = nxt.get((a, b), 0.0) + pr
                    continue
                nxt[(a + 1, b)] = nxt.get((a + 1, b), 0.0) + pr * r_ot
                nxt[(a, b + 1)] = nxt.get((a, b + 1), 0.0) + pr * (1 - r_ot)
            state = nxt
        for (a, b), pr in state.items():
            if a == 3 and b == 3:
                ot_tie += pr
            else:
                ot_final[(a, b)] = pr
        carry = tie
        for k in range(MAX_OT):
            base = 12 + 3 * k
            for (a, b), pr in ot_final.items():
                key = (base + a, base + b)
                dist[key] = dist.get(key, 0.0) + carry * pr
            carry *= ot_tie
        # Остаток хвоста делим пропорционально шансу выиграть OT.
        if carry > 0:
            p_ot = sum(pr for (a, b), pr in ot_final.items() if a > b) / max(1e-12, 1 - ot_tie)
            base = 12 + 3 * MAX_OT
            dist[(base + 4, base + 2)] = dist.get((base + 4, base + 2), 0.0) + carry * p_ot
            dist[(base + 2, base + 4)] = dist.get((base + 2, base + 4), 0.0) + carry * (1 - p_ot)
    return tuple(sorted(dist.items()))


def map_round_distribution(r_ct: float, r_t: float) -> dict[tuple[int, int], float]:
    """Счёт карты, усреднённый по тому, какой стороной команда1 начинает.

    r_ct — вероятность команды1 выиграть раунд, играя за CT; r_t — за T.
    """
    r_ot = (r_ct + r_t) / 2
    a = dict(_map_round_dist_cached(round(r_ct, 6), round(r_t, 6), round(r_ot, 6)))
    b = dict(_map_round_dist_cached(round(r_t, 6), round(r_ct, 6), round(r_ot, 6)))
    out: dict[tuple[int, int], float] = {}
    for d in (a, b):
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v / 2
    return out


def map_win_prob(r_ct: float, r_t: float) -> float:
    return sum(p for (a, b), p in map_round_distribution(r_ct, r_t).items() if a > b)


def calibrate_rounds(p_map: float, side_bias: float = 0.0, ct_edge: float = 0.0) -> tuple[float, float]:
    """Находит (r_ct, r_t) команды1, при которых P(победа на карте) = p_map.

    ``side_bias`` — общий для обеих команд перекос карты в сторону CT (в логитах),
    ``ct_edge`` — относительное преимущество команды1 за CT против её T (в логитах),
    положительное значение = команда1 сильнее за CT, чем за T.
    """
    lo, hi = -4.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        r_ct = logistic(mid + side_bias + ct_edge / 2)
        r_t = logistic(mid - side_bias - ct_edge / 2)
        if map_win_prob(r_ct, r_t) < p_map:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2
    return logistic(mid + side_bias + ct_edge / 2), logistic(mid - side_bias - ct_edge / 2)
