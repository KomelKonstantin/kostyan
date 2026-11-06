"""Командная строка cs2-analyst."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from . import __version__
from .fetch import BlockedError, ChainSource, DirectorySource, PoliteFetcher, SourceMissingError
from .hltv import Collector
from .model import analyze
from .models import Match, merge_dicts, normalize_map
from .odds import assess, parse_odds
from .report import labels_for, render_markdown, to_json

EXIT_BLOCKED = 3
EXIT_INPUT = 2

TEMPLATE = {
    "_comment": "Шаблон ручного ввода. Пустые/отсутствующие поля = [нет данных]. Не выдумывайте значения.",
    "url": "https://www.hltv.org/matches/<id>/<slug>",
    "event": None,
    "date": "YYYY-MM-DD",
    "best_of": 3,
    "lan": None,
    "map_pool": ["Ancient", "Anubis", "Dust2", "Inferno", "Mirage", "Nuke", "Train"],
    "veto": [
        {"name": "Mirage", "picked_by": 1},
        {"name": "Inferno", "picked_by": 2},
        {"name": "Nuke", "picked_by": None},
    ],
    "h2h": [{"date": "YYYY-MM-DD", "winner": 1}],
    "team1": {
        "name": "Team A",
        "rank": None,
        "points": None,
        "lan_winrate": None, "lan_matches": None,
        "online_winrate": None, "online_matches": None,
        "pistol_winrate": None,
        "days_since_roster_change": None,
        "players": [{"name": "player1", "rating": None, "is_standin": False}],
        "recent_matches": [
            {"date": "YYYY-MM-DD", "opponent": "Team X", "opponent_rank": None, "won": True,
             "maps_won": 2, "maps_lost": 1, "lan": True}
        ],
        "maps": {
            "Mirage": {"played": None, "winrate": None, "ct_round_win": None, "t_round_win": None,
                       "pick_rate": None, "ban_rate": None}
        },
    },
    "team2": {"name": "Team B", "...": "та же структура, что у team1"},
}

BLOCKED_HELP = """
HLTV закрыл доступ защитой Cloudflare ({reason}).
Обходить защиту инструмент не будет (никаких прокси, подмены IP или решения капчи).

Варианты:
  (2) Браузер — Claude in Chrome. Откройте страницы HLTV в своём браузере и сохраните HTML
      в один каталог под именами ролей, затем запустите с --html-dir:
        match.html          — страница матча
        ranking.html        — https://www.hltv.org/ranking/teams
        team1_results.html  — https://www.hltv.org/results?team=<id1>
        team2_results.html  — https://www.hltv.org/results?team=<id2>
        team1_maps.html     — https://www.hltv.org/stats/teams/maps/<id1>/<slug1>
        team2_maps.html     — https://www.hltv.org/stats/teams/maps/<id2>/<slug2>
        player_<slug>.html  — https://www.hltv.org/stats/players/<id>/<slug> (по желанию)
      Либо попросите Claude in Chrome заполнить JSON-шаблон (`cs2-analyst template`).
  (3) Вручную: `cs2-analyst template > match.json`, заполните известные поля и запустите
      `cs2-analyst analyze --data match.json [--odds odds.json]`.
Неизвестные значения оставляйте null — они будут помечены [нет данных].
""".strip()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cs2-analyst", description="Вероятностный анализ матчей CS2 по данным HLTV.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Проанализировать матч")
    a.add_argument("url", nargs="?", help="Ссылка на матч HLTV")
    a.add_argument("--teams", nargs=2, metavar=("TEAM1", "TEAM2"), help="Названия команд вместо ссылки")
    a.add_argument("--odds", help="Коэффициенты: JSON/CSV строкой, путь к файлу или '-' (stdin)")
    a.add_argument("--data", help="JSON с ручными данными (дополняет/перекрывает HLTV)")
    a.add_argument("--html-dir", help="Каталог с HTML-страницами HLTV, сохранёнными из браузера")
    a.add_argument("--no-fetch", action="store_true", help="Не обращаться к сети вообще")
    a.add_argument("--offline", action="store_true", help="Только дисковый кэш (без сети)")
    a.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "cs2-analyst"))
    a.add_argument("--delay", type=float, default=3.0, help="Пауза между запросами, с (минимум 3)")
    a.add_argument("--no-players", action="store_true", help="Не загружать статистику игроков")
    a.add_argument("--bo", type=int, choices=(1, 3, 5), help="Формат матча, если неизвестен")
    g = a.add_mutually_exclusive_group()
    g.add_argument("--lan", dest="lan", action="store_const", const=True, help="Матч на LAN")
    g.add_argument("--online", dest="lan", action="store_const", const=False, help="Онлайн-матч")
    a.add_argument("--map-pool", nargs="+", help="Актуальный пул карт (7 карт)")
    a.add_argument("--maps", nargs="+", help="Известные карты серии по порядку (перекрывает вето)")
    a.add_argument("--date", help="Дата анализа YYYY-MM-DD (для затухания формы)")
    a.add_argument("--min-edge", type=float, default=0.03, help="Минимальный EV для перевеса (0.03 = 3%%)")
    a.add_argument("--format", choices=("md", "json"), default="md")
    a.add_argument("--out", help="Записать отчёт в файл")
    a.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("template", help="Вывести JSON-шаблон для ручного ввода")
    sub.add_parser("events", help="Список ключей событий для --odds")
    return p


def _collect(args) -> tuple[dict, list[str]]:
    notes: list[str] = []
    manual: dict = {}
    if args.data:
        manual = json.loads(Path(args.data).read_text(encoding="utf-8"))
        manual.pop("_comment", None)
        notes.append(f"Ручные данные: {args.data}")

    if not args.url and not args.teams:
        if not manual:
            raise SystemExit("Укажите ссылку на матч, --teams или --data")
        return manual, notes

    sources = []
    if args.html_dir:
        sources.append(DirectorySource(Path(args.html_dir)))
        notes.append(f"HTML из каталога: {args.html_dir}")
    if not args.no_fetch:
        sources.append(PoliteFetcher(Path(args.cache_dir), min_delay=args.delay, offline=args.offline,
                                     verbose=args.verbose))
    if not sources:
        if manual:
            return manual, notes
        raise SystemExit("--no-fetch без --html-dir и --data: нет источника данных")

    collector = Collector(ChainSource(*sources), fetch_players=not args.no_players,
                          today=_today(args))
    try:
        if args.url:
            fetched = collector.from_match_url(args.url)
            notes.append("Данные HLTV получены")
        else:
            fetched = collector.from_team_names(*args.teams)
    except BlockedError as e:
        if manual:
            notes.append(f"HLTV недоступен ({e}); анализ только по ручным данным")
            notes += collector.notes
            return manual, notes
        print(BLOCKED_HELP.format(reason=e), file=sys.stderr)
        raise SystemExit(EXIT_BLOCKED)
    except SourceMissingError as e:
        if manual:
            notes.append(f"{e}; используются ручные данные")
            return manual, notes
        raise SystemExit(f"Нет обязательной страницы: {e}")
    notes += collector.notes
    return merge_dicts(fetched, manual), notes


def _today(args) -> date | None:
    return datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None


def run_analyze(args) -> int:
    raw, notes = _collect(args)
    if args.teams:
        raw.setdefault("team1", {}).setdefault("name", args.teams[0])
        raw.setdefault("team2", {}).setdefault("name", args.teams[1])
    if args.url:
        raw.setdefault("url", args.url)
    if args.bo:
        raw["best_of"] = args.bo
    if args.lan is not None:
        raw["lan"] = args.lan
    if args.map_pool:
        raw["map_pool"] = args.map_pool
    if args.maps:
        raw["veto"] = [{"name": m, "picked_by": None} for m in args.maps]
    for t in ("team1", "team2"):
        if not (raw.get(t) or {}).get("name"):
            print(f"Не удалось определить {t}. Передайте --teams или заполните --data.", file=sys.stderr)
            return EXIT_INPUT

    try:
        match = Match.from_dict(raw)
    except (KeyError, ValueError, TypeError) as e:
        print(f"Ошибка во входных данных: {e}", file=sys.stderr)
        return EXIT_INPUT

    pool = [normalize_map(m) for m in args.map_pool] if args.map_pool else None
    pred = analyze(match, today=_today(args), map_pool=pool)

    assessments = None
    unknown: list[str] = []
    if args.odds:
        try:
            lines = parse_odds(args.odds, match.team1.name, match.team2.name)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Ошибка в --odds: {e}", file=sys.stderr)
            return EXIT_INPUT
        assessments, unknown = assess(lines, pred.events, pred.event_ranges, labels_for(pred),
                                      min_edge=args.min_edge)

    text = to_json(pred, assessments, unknown, notes) if args.format == "json" else \
        render_markdown(pred, assessments, unknown, notes)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Отчёт записан: {args.out}")
    else:
        print(text)
    return 0


EVENTS_HELP = """Ключи событий для --odds (team1/team2 можно заменить названием команды):
  ml:team1 | ml:team2 (синонимы: П1, П2, W1, W2, «<команда>»)
  score:2-0, score:2-1, score:1-2, score:0-2 (BO3); score:3-1 и т.п. (BO5)
  handicap:team1:-1.5, handicap:team2:+1.5 (фора по картам; Ф1/Ф2)
  total_maps:over:2.5, total_maps:under:2.5 (ТБ/ТМ)
  map1:team1, map2:team2 ...
  map1:total_rounds:over:21.5 (или map1:over:21.5, map1 ТБ 21.5)
  map1:handicap:team1:-3.5, map1:handicap:team2:+4.5 (фора по раундам)
  map1:overtime:yes | map1:overtime:no
  map1:pistol1:team1 (первый пистолет; требуется pistol_winrate обеих команд)
Пример JSON: {"ml:team1": 1.62, "ml:team2": 2.30, "handicap:team1:-1.5": 2.75}
Пример CSV:  event,odds\\nП1,1.62\\nП2,2.30"""


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "template":
        print(json.dumps(TEMPLATE, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "events":
        print(EVENTS_HELP)
        return 0
    return run_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
