"""Сбор данных матча из открытого API bo3.gg (пауза 3 с между запросами)."""
import json, sys, time, urllib.parse
from collections import defaultdict
from datetime import date
import requests
UA = "cs2-analyst/0.1 (personal research)"
S = requests.Session(); S.headers["User-Agent"] = UA
def get(path, **params):
    time.sleep(3)
    r = S.get("https://api.bo3.gg/api/v1/" + path, params=params, timeout=20); r.raise_for_status(); return r.json()
t1, t2, slug, today = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], date.fromisoformat(sys.argv[4])
match = get(f"matches/{slug}", scope="show-match", **{"with": "games,teams,tournament_deep,stage"})
names = {match["team1"]["id"]: match["team1"]["name"], match["team2"]["id"]: match["team2"]["name"]}
ranks = {match["team1"]["id"]: match["team1"]["rank"], match["team2"]["id"]: match["team2"]["rank"]}
rec = {}
for t in (t1, t2):
    ms = []
    for side in ("team1_id", "team2_id"):
        d = get("matches", **{f"filter[matches.{side}][eq]": t, "filter[matches.status][eq]": "finished",
                              "sort": "-start_date", "page[limit]": 50})
        ms += [m for m in d["results"] if 0 <= (today - date.fromisoformat(m["start_date"][:10])).days <= 90]
    rec[t] = sorted(ms, key=lambda m: m["start_date"], reverse=True)
h2h = get("matches", **{"filter[matches.team1_id][in]": f"{t1},{t2}", "filter[matches.team2_id][in]": f"{t1},{t2}",
                        "filter[matches.status][eq]": "finished", "sort": "-start_date", "page[limit]": 20})["results"]
opp = {m["team2_id"] if m["team1_id"] == t else m["team1_id"] for t, ms in rec.items() for m in ms}
opps = {x["id"]: x for x in get("teams", **{"filter[teams.id][in]": ",".join(map(str, opp)), "page[limit]": 100})["results"]}
tids = {m["tournament_id"] for ms in rec.values() for m in ms}
tours = {x["id"]: x for x in get("tournaments", **{"filter[tournaments.id][in]": ",".join(map(str, tids)), "page[limit]": 100})["results"]}
mids = {m["id"] for ms in rec.values() for m in ms} | {m["id"] for m in h2h}
games, off = [], 0
while True:
    d = get("games", **{"filter[games.match_id][in]": ",".join(map(str, mids)), "page[limit]": 100, "page[offset]": off})
    games += d["results"]; off += 100
    if off >= d["total"]["count"]: break
def same(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    return bool(a and b) and (a.startswith(b.split()[0]) or b.startswith(a.split()[0]))
def team(t):
    rm, lan, mp, mid = [], [0, 0], defaultdict(lambda: [0, 0]), set()
    for m in rec[t]:
        if m["winner_team_id"] is None: continue
        o = m["team2_id"] if m["team1_id"] == t else m["team1_id"]
        mine, th = (m["team1_score"], m["team2_score"]) if m["team1_id"] == t else (m["team2_score"], m["team1_score"])
        won = m["winner_team_id"] == t
        is_lan = {"lan": True, "online": False}.get(tours.get(m["tournament_id"], {}).get("event_type"))
        if is_lan: lan[0] += won; lan[1] += 1
        mid.add(m["id"])
        rm.append({"date": m["start_date"][:10], "opponent": opps.get(o, {}).get("name", str(o)),
                   "opponent_rank": opps.get(o, {}).get("rank"), "won": won, "maps_won": mine, "maps_lost": th, "lan": is_lan})
    for g in games:
        if g["match_id"] in mid and g["status"] == "finished" and (same(names[t], g["winner_clan_name"]) or same(names[t], g["loser_clan_name"])):
            n = g["map_name"].replace("de_", "").capitalize()
            mp[n][1] += 1; mp[n][0] += same(names[t], g["winner_clan_name"])
    out = {"name": names[t], "rank": ranks[t], "recent_matches": rm,
           "maps": {k: {"played": v[1], "winrate": round(v[0] / v[1], 3)} for k, v in mp.items()}}
    if lan[1]: out["lan_winrate"], out["lan_matches"] = lan[0] / lan[1], lan[1]
    return out
h2h_maps = [(g["match_id"], g["number"], g["map_name"], g["winner_clan_name"], g["winner_clan_score"], g["loser_clan_score"])
            for g in games if g["match_id"] in {m["id"] for m in h2h} and g["status"] == "finished"]
res = {"_comment": f"Собрано из открытого API bo3.gg {today}. Ранги — рейтинг bo3.gg, НЕ HLTV. Игроки, CT/T, pick/ban, пистолеты — нет данных.",
       "url": f"https://bo3.gg/matches/{slug}", "event": match["tournament"]["name"] + " — " + (match.get("round") or {}).get("name", ""),
       "date": str(today), "best_of": match["bo_type"], "lan": match["tournament"].get("event_type") == "lan",
       "h2h": [{"date": m["start_date"][:10], "winner": 1 if m["winner_team_id"] == t1 else 2} for m in h2h if m["winner_team_id"]],
       "team1": team(t1), "team2": team(t2)}
json.dump(res, open(sys.argv[5], "w"), ensure_ascii=False, indent=1)
print("games in match:", [(g.get("number"), g.get("map_name"), g.get("status")) for g in match.get("games", [])])
print("bet:", {k: (v.get("coeff") if isinstance(v, dict) else None) for k, v in (match.get("bet_updates") or {}).items() if k.startswith("team")})
for t in ("team1", "team2"):
    x = res[t]; print(x["name"], "#", x["rank"], "form", sum(r["won"] for r in x["recent_matches"]), "-", sum(not r["won"] for r in x["recent_matches"]),
                      "LAN", x.get("lan_winrate"), {k: f"{int(round(v['winrate']*v['played']))}/{v['played']}" for k, v in sorted(x["maps"].items())})
print("h2h:", res["h2h"])
for h in sorted(h2h_maps): print("  h2h map", h)
