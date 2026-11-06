import pytest

from cs2_analyst.odds import assess, complement, normalize_key, parse_odds


@pytest.mark.parametrize("raw,key", [
    ("П1", "ml:team1"), ("w2", "ml:team2"), ("Alpha", "ml:team1"),
    ("Ф1 -1.5", "handicap:team1:-1.5"), ("handicap:team2:+1.5", "handicap:team2:+1.5"),
    ("ТБ 2.5", "total_maps:over:2.5"), ("total_maps:under:2.5", "total_maps:under:2.5"),
    ("map1 ТБ 21.5", "map1:total_rounds:over:21.5"), ("map2:Bravo", "map2:team2"),
    ("score 2:1", "score:2-1"), ("map1:handicap:team1:-3.5", "map1:handicap:team1:-3.5"),
])
def test_normalize(raw, key):
    assert normalize_key(raw, "Alpha", "Bravo") == key


def test_parse_json_and_csv():
    j = parse_odds('{"П1": 1.8, "П2": "2,05"}', "A", "B")
    assert [(l.key, l.odds) for l in j] == [("ml:team1", 1.8), ("ml:team2", 2.05)]
    c = parse_odds("event,odds\nП1,1.8\nП2,2.05\n", "A", "B")
    assert [(l.key, l.odds) for l in c] == [("ml:team1", 1.8), ("ml:team2", 2.05)]


def test_rejects_bad_odds():
    with pytest.raises(ValueError):
        parse_odds('{"П1": 0.9}', "A", "B")


def test_complement():
    assert complement("handicap:team1:-1.5") == ["handicap:team2:+1.5"]
    assert complement("map1:total_rounds:over:21.5") == ["map1:total_rounds:under:21.5"]
    assert complement("ml:team2") == ["ml:team1"]


def test_assess_margin_and_ev():
    lines = parse_odds('{"ml:team1": 1.9, "ml:team2": 1.9, "zzz": 2.0}', "A", "B")
    ev = {"ml:team1": 0.6, "ml:team2": 0.4}
    rng = {"ml:team1": (0.55, 0.65), "ml:team2": (0.35, 0.45)}
    res, unknown = assess(lines, ev, rng, {})
    a = next(x for x in res if x.key == "ml:team1")
    assert a.fair_implied == pytest.approx(0.5)
    assert a.margin == pytest.approx(2 / 1.9 - 1)
    assert a.ev == pytest.approx(0.6 * 1.9 - 1)
    assert a.verdict == "перевес (value)"
    assert 0 < a.kelly <= 0.02
    b = next(x for x in res if x.key == "ml:team2")
    assert b.kelly == 0 and b.verdict == "нет перевеса"
    assert unknown and "zzz" in unknown[0]
