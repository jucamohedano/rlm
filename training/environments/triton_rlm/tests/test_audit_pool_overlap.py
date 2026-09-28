import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_pool_overlap.py"
spec = importlib.util.spec_from_file_location("audit_pool_overlap", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

CONV = (
    "import torch\nimport torch.nn as nn\n\n\nclass Model(nn.Module):\n"
    "    def __init__(self, c_in, c_out):\n        super().__init__()\n"
    "        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)\n"
    "        self.bn = nn.BatchNorm2d(c_out)\n        self.act = nn.ReLU()\n\n"
    "    def forward(self, x):\n        x = self.conv(x)\n        x = self.bn(x)\n"
    "        return self.act(x) + x.mean(dim=(2, 3), keepdim=True)\n"
)
OTHER = (
    "import torch\nimport torch.nn as nn\n\n\nclass Model(nn.Module):\n"
    "    def forward(self, a, b):\n        return torch.softmax(a @ b.transpose(-1, -2), dim=-1) * 0.5\n"
)


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_hits_leave_pool_and_control_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ppb = _write(
        tmp_path / "manifest.jsonl",
        [{"_task_id": "kernelbook_1", "code": CONV, "cluster": 0}],
    )
    pool = _write(
        tmp_path / "lvl1.jsonl",
        [
            {"_task_id": "ops6k_dup", "code": "# comment only\n" + CONV},
            {"_task_id": "ops6k_other", "code": OTHER},
        ],
    )
    out = tmp_path / "overlap.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--ppbhatt-manifest", str(ppb), "--pool", str(pool), "--out", str(out)],
    )
    with pytest.raises(SystemExit) as exc:
        audit.main()
    assert exc.value.code == 1
    report = json.loads(out.read_text())["pools"][str(pool)]
    assert report["hits_at_threshold"] == ["ops6k_dup"]
    assert [n["pool_task"] for n in report["near_matches"]] == ["ops6k_dup"]
    assert report["near_matches"][0]["est_jaccard"] == 1.0
    survivors = audit.load_jsonl(tmp_path / "lvl1_disjoint.jsonl")
    assert [r["_task_id"] for r in survivors] == ["ops6k_other"]


def test_disjoint_pool_exits_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ppb = _write(
        tmp_path / "manifest.jsonl",
        [{"_task_id": "kernelbook_1", "code": CONV, "cluster": 0}],
    )
    pool = _write(tmp_path / "lvl1.jsonl", [{"_task_id": "ops6k_other", "code": OTHER}])
    out = tmp_path / "overlap.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--ppbhatt-manifest", str(ppb), "--pool", str(pool), "--out", str(out)],
    )
    audit.main()
    report = json.loads(out.read_text())
    assert report["ppbhatt_clusters"] == 1
    assert report["pools"][str(pool)]["hits_at_threshold"] == []
    assert report["pools"][str(pool)]["near_matches"] == []
    assert report["pools"][str(pool)]["disjoint_written"] is None
    assert not (tmp_path / "lvl1_disjoint.jsonl").exists()
