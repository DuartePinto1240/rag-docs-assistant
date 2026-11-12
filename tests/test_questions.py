import json
from pathlib import Path

import pytest

DATA_DIR = Path("data")


@pytest.fixture(scope="module")
def questions():
    text = Path("eval/questions.json").read_text(encoding="utf-8")
    return json.loads(text)


def test_ids_are_unique(questions):
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_required_fields(questions):
    required = {"id", "question", "reference_answer", "expected_source", "answerable"}
    for q in questions:
        assert required <= q.keys(), f"{q.get('id')} is missing fields"
        assert q["question"].strip() and q["reference_answer"].strip()


def test_answerable_questions_point_to_existing_files(questions):
    for q in questions:
        if q["answerable"]:
            for filename in [q["expected_source"], *q.get("acceptable_sources", [])]:
                assert (DATA_DIR / filename).exists(), f"{q['id']}: {filename} not found"


def test_unanswerable_questions_have_no_source(questions):
    for q in questions:
        if not q["answerable"]:
            assert q["expected_source"] is None


def test_set_has_both_kinds(questions):
    assert any(q["answerable"] for q in questions)
    assert any(not q["answerable"] for q in questions)