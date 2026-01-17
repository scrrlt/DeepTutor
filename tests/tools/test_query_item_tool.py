import pytest
from pathlib import Path
from src.tools.query_item_tool import query_numbered_item

def write_kb(tmp_path: Path, kb_name: str, items: dict):
    kb_dir = tmp_path / kb_name
    kb_dir.mkdir()
    items_file = kb_dir / "numbered_items.json"
    import json

    with open(items_file, "w", encoding="utf-8") as f:
        json.dump(items, f)
    return kb_dir

class TestQueryNumberedItem:

    def test_exact_match(self, tmp_path: Path):
        write_kb(tmp_path, "kb1", {"Definition 1.1": "A definition text"})
        res = query_numbered_item("Definition 1.1", kb_name="kb1", kb_base_dir=str(tmp_path))
        assert res["status"] == "success"
        assert res["count"] == 1
        assert "Definition 1.1" in res["items"][0]["identifier"]

    def test_not_found_with_suggestions(self, tmp_path: Path):
        write_kb(tmp_path, "kb1", {"Figure 1.1": "Figure text", "Figure 1.2": "Another figure"})
        res = query_numbered_item("Figure 2.1", kb_name="kb1", kb_base_dir=str(tmp_path))
        assert res["status"] == "failed"
        assert "Similar items" in res["error"] or "Similar items" in res["error"]

    def test_missing_kb(self, tmp_path: Path):
        res = query_numbered_item("Definition 1.1", kb_name="nonexistent", kb_base_dir=str(tmp_path))
        assert res["status"] == "failed"
        assert "does not exist" in res["error"]
