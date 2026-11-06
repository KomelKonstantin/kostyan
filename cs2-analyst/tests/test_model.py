import json
from datetime import date
from pathlib import Path

import pytest

from cs2_analyst.model import analyze
from cs2_analyst.models import Match

DEMO = json.loads((Path(__file__).parent.parent / "examples" / "demo_match.json").read_text(encoding="utf-8"))


def test_demo_consistency():
    pred = analyze(Match.from_dict(DEMO))
    ev = pred.events
    assert 0.5 < pred.p_match < 0.9
    assert ev["ml:team1"] + ev["ml:team2"] == pytest.approx(1)
    assert sum(v for k, v in ev.items() if k.startswith("score:")) == pytest.approx(1)
    assert ev["handicap:team1:-1.5"] == pytest.approx(ev["score:2-0"])
    assert ev["total_maps:over:2.5"] == pytest.approx(ev["score:2-1"] + ev["score:1-2"])
    for k, (lo, hi) in pred.event_ranges.items():
        assert lo <= ev[k] + 1e-9 and ev[k] <= hi + 1e-9
    assert len(pred.maps) == 3 and len({m.name for m in pred.maps}) == 3


def test_symmetry_swap_teams():
    swapped = dict(DEMO, team1=DEMO["team2"], team2=DEMO["team1"],
                   h2h=[dict(h, winner=3 - h["winner"]) for h in DEMO["h2h"]])
    a = analyze(Match.from_dict(DEMO))
    b = analyze(Match.from_dict(swapped))
    # Вето зависит от того, кто начинает, поэтому допускаем небольшое расхождение.
    assert a.p_match == pytest.approx(1 - b.p_match, abs=0.06)


def test_minimal_data_low_confidence_and_flags():
    m = Match.from_dict({"team1": {"name": "A"}, "team2": {"name": "B"}})
    pred = analyze(m, today=date(2026, 11, 6))
    assert pred.confidence == "низкая"
    assert pred.p_match == pytest.approx(0.5, abs=1e-6)
    assert "Формат матча (BO1/BO3/BO5)" in pred.missing
    assert any("BO3" in w for w in pred.warnings)


def test_known_veto_used_and_bo1():
    d = dict(DEMO, best_of=1, veto=[{"name": "Nuke", "picked_by": None}])
    pred = analyze(Match.from_dict(d))
    assert [m.name for m in pred.maps] == ["Nuke"]
    assert pred.maps[0].source == "вето HLTV"
    assert pred.events["ml:team1"] == pytest.approx(pred.events["map1:team1"])


def test_less_data_pulls_to_fifty():
    full = analyze(Match.from_dict(DEMO))
    stripped = json.loads(json.dumps(DEMO))
    for t in ("team1", "team2"):
        stripped[t].pop("recent_matches"); stripped[t].pop("players")
    weak = analyze(Match.from_dict(stripped))
    assert weak.completeness < full.completeness
    assert abs(weak.p_match - 0.5) < abs(full.p_match - 0.5)
