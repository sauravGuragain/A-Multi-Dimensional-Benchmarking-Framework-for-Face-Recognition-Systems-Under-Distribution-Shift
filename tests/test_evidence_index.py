import json

import pytest

from faceeval.analysis import evidence_index as EI


def _fairness_dir(d):
    d.mkdir()
    (d / "fairness_by_group.csv").write_text("a,b\n1,2\n3,4\n")
    (d / "fairness_disparities.csv").write_text("a\n1\n")
    (d / "fairness_coverage.json").write_text("{}")
    return d


def test_present_incomplete_and_not_provided(tmp_path):
    f = _fairness_dir(tmp_path / "f")
    bad = tmp_path / "e"; bad.mkdir()
    idx = EI.build({"fairness": f, "efficiency": bad, "calibration": tmp_path / "nope"}, tmp_path / "out")
    st = {p["kind"]: p for p in idx["products"]}
    assert st["fairness"]["status"] == "PRESENT" and st["fairness"]["files"]["fairness_by_group.csv"]["rows"] == 2
    assert st["efficiency"]["status"] == "INCOMPLETE" and "missing efficiency_results.csv" in st["efficiency"]["problems"]
    assert st["calibration"]["status"] == "INCOMPLETE" and st["valid_pairs"]["status"] == "NOT_PROVIDED"
    assert idx["summary"] == {"PRESENT": 1, "INCOMPLETE": 2, "INVALID": 0, "NOT_PROVIDED": 3}
    assert (tmp_path / "out" / "evidence_index.md").read_text().startswith("# Evidence index")


def test_unparseable_json_is_incomplete_and_inputs_untouched(tmp_path):
    f = _fairness_dir(tmp_path / "f")
    (f / "fairness_coverage.json").write_text("{not json")
    before = {p.name: p.read_bytes() for p in f.iterdir()}
    idx = EI.build({"fairness": f}, tmp_path / "out")
    assert [p for p in idx["products"] if p["kind"] == "fairness"][0]["status"] == "INCOMPLETE"
    assert before == {p.name: p.read_bytes() for p in f.iterdir()}


def test_refuses_output_inside_input_and_nonempty_out(tmp_path):
    f = _fairness_dir(tmp_path / "f")
    with pytest.raises(EI.IndexError_, match="inside an input"):
        EI.build({"fairness": f}, f / "idx")
    (tmp_path / "o").mkdir(); (tmp_path / "o" / "x").write_text("1")
    assert EI.main(["--fairness", str(f), "--out", str(tmp_path / "o")]) == 2
    assert EI.main(["--fairness", str(f), "--out", str(tmp_path / "ok")]) == 0
    assert json.loads((tmp_path / "ok" / "evidence_index.json").read_text())["summary"]["PRESENT"] == 1
