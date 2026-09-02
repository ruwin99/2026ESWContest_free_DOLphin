from __future__ import annotations

from common import cross_split_issues


def test_cross_split_checks_all_physical_and_content_identities() -> None:
    train = {
        "split": "train", "rows": 1, "group_ids": ["g1"], "image_hashes": ["i1"], "mask_hashes": ["m1"],
        "identities": {"source_id": ["s1"], "print_id": ["p1"], "placement_id": ["l1"], "session_id": ["x1"]},
    }
    validation = {
        "split": "validation", "rows": 1, "group_ids": ["g1"], "image_hashes": ["i1"], "mask_hashes": ["m1"],
        "identities": {"source_id": ["s1"], "print_id": ["p1"], "placement_id": ["l1"], "session_id": ["x1"]},
    }
    issues = cross_split_issues({"train": train, "validation": validation})
    for identity in ("group_ids", "image_hashes", "mask_hashes", "source_id", "print_id", "placement_id", "session_id"):
        assert any(identity in issue for issue in issues)


def test_independent_splits_have_no_leakage_issue() -> None:
    def split(name: str, suffix: str):
        return {"split": name, "rows": 1, "group_ids": ["g"+suffix], "image_hashes": ["i"+suffix],
                "mask_hashes": ["m"+suffix], "identities": {key: [key+suffix] for key in ("source_id", "print_id", "placement_id", "session_id")}}
    assert cross_split_issues({"train": split("train", "1"), "validation": split("validation", "2")}) == []
