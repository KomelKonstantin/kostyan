import json
from pathlib import Path

import pytest

from cs2_analyst import cli
from cs2_analyst.fetch import BlockedError, DirectorySource, looks_like_challenge
from cs2_analyst.hltv import (Collector, parse_match_page, parse_match_url, parse_ranking, parse_team_maps,
                              parse_team_results)

EX = Path(__file__).parent.parent / "examples"

MATCH_HTML = """
<div class="team1-gradient"><a href="/team/111/alpha"><div class="teamName">Alpha</div></a></div>
<div class="team2-gradient"><a href="/team/222/bravo"><div class="teamName">Bravo</div></a></div>
<div class="date" data-unix="1793000000000"></div>
<div class="event text-ellipsis"><a href="/events/1/x">Demo Cup</a></div>
<div class="padding preformatted-text">Best of 3 (LAN)
* Playoffs</div>
<div class="veto-box"><div class="padding">
<div>1. Alpha removed Anubis</div><div>2. Bravo removed Dust2</div>
<div>3. Alpha picked Mirage</div><div>4. Bravo picked Inferno</div>
<div>5. Alpha removed Train</div><div>6. Bravo removed Ancient</div>
<div>7. Nuke was left over</div></div></div>
<div class="lineups">
 <div class="lineup"><div class="teamRanking"><a>World rank: #4</a></div>
   <a href="/player/1/p-one"><div class="text-ellipsis">pOne</div></a></div>
 <div class="lineup"><div class="teamRanking"><a>World rank: #9</a></div>
   <a href="/player/2/p-two"><div class="text-ellipsis">pTwo</div></a></div>
</div>
"""

RANKING_HTML = """
<div class="ranked-team"><span class="position">#4</span><div class="teamLine"><span class="name">Alpha</span>
<span class="points">(640 points)</span></div></div>
<div class="ranked-team"><span class="position">#9</span><div class="teamLine"><span class="name">Bravo</span>
<span class="points">(380 points)</span></div></div>
"""

RESULTS_HTML = """
<div class="results-sublist"><div class="standard-headline">Results for November 1st 2026</div>
<div class="result-con"><div class="team1"><div class="team">Alpha</div></div>
<div class="result-score"><span class="score-won">2</span><span class="score-lost">1</span></div>
<div class="team2"><div class="team">Bravo</div></div></div></div>
"""

MAPS_HTML = """
<div class="two-grid"><div class="col"><div class="map-pool-map-name">Mirage - 66.7%</div>
<div class="stats-row"><span>Times played</span><span>18</span></div>
<div class="stats-row"><span>Win rate</span><span>66.7%</span></div></div></div>
"""


def test_parse_match_url():
    assert parse_match_url("https://www.hltv.org/matches/2380000/alpha-vs-bravo") == (2380000, "alpha-vs-bravo")
    with pytest.raises(ValueError):
        parse_match_url("https://example.com/foo")


def test_parse_match_page():
    d = parse_match_page(MATCH_HTML)
    assert d["team1"]["name"] == "Alpha" and d["team1"]["hltv_id"] == 111
    assert d["best_of"] == 3 and d["lan"] is True and d["event"] == "Demo Cup"
    assert d["team1"]["rank"] == 4 and d["team2"]["rank"] == 9
    assert d["veto"] == [{"name": "Mirage", "picked_by": 1}, {"name": "Inferno", "picked_by": 2},
                         {"name": "Nuke", "picked_by": None}]
    assert d["team1"]["players"][0]["name"] == "pOne"


def test_parse_ranking_results_maps():
    r = parse_ranking(RANKING_HTML)
    assert r["alpha"] == {"rank": 4, "points": 640}
    res = parse_team_results(RESULTS_HTML, "Bravo", r)
    assert res == [{"date": "2026-11-01", "opponent": "Alpha", "won": False, "opponent_rank": 4,
                    "maps_won": 1, "maps_lost": 2}]
    assert parse_team_maps(MAPS_HTML) == {"Mirage": {"played": 18, "winrate": pytest.approx(0.667)}}


def test_challenge_detection():
    assert looks_like_challenge(403, "")
    assert looks_like_challenge(200, "<title>Just a moment...</title>")
    assert not looks_like_challenge(200, "<html>ok</html>")


def test_collector_from_html_dir(tmp_path):
    (tmp_path / "match.html").write_text(MATCH_HTML)
    (tmp_path / "ranking.html").write_text(RANKING_HTML)
    (tmp_path / "team1_results.html").write_text(RESULTS_HTML)
    c = Collector(DirectorySource(tmp_path), fetch_players=False)
    d = c.from_match_url("https://www.hltv.org/matches/2380000/alpha-vs-bravo")
    assert d["team1"]["points"] == 640
    assert any("team2_results" in n for n in c.notes)  # отсутствующие страницы отмечены


class _Blocked:
    def get(self, role, url):
        raise BlockedError("403")


def test_cli_blocked_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(cli, "PoliteFetcher", lambda *a, **k: _Blocked())
    with pytest.raises(SystemExit) as e:
        cli.main(["analyze", "https://www.hltv.org/matches/1/a-vs-b"])
    assert e.value.code == cli.EXIT_BLOCKED
    assert "Claude in Chrome" in capsys.readouterr().err


def test_cli_blocked_falls_back_to_manual(monkeypatch, capsys):
    monkeypatch.setattr(cli, "PoliteFetcher", lambda *a, **k: _Blocked())
    rc = cli.main(["analyze", "https://www.hltv.org/matches/1/a-vs-b", "--data", str(EX / "demo_match.json")])
    assert rc == 0
    assert "HLTV недоступен" in capsys.readouterr().out


def test_cli_json_with_odds(capsys):
    rc = cli.main(["analyze", "--data", str(EX / "demo_match.json"), "--odds", str(EX / "demo_odds.json"),
                   "--format", "json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["odds_assessment"] and data["unrecognized_odds"] == []
    assert 0 < data["p_team1_match"] < 1


def test_cli_template(capsys):
    assert cli.main(["template"]) == 0
    assert "team1" in json.loads(capsys.readouterr().out)
