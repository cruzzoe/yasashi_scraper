import pytest
from pydantic import ValidationError

from models import Enrichment, Translation, VocabularyCandidate


def test_enrichment_requires_unique_distractors():
    with pytest.raises(ValidationError, match="distractors must be unique"):
        VocabularyCandidate(
            surface="雨", reading="あめ", meaning="rain", paragraph_ids=["p1"],
            difficulty_rank=1, distractors=["snow", "snow", "wind"],
        )


def test_enrichment_accepts_valid_minimum_payload():
    result = Enrichment.model_validate({
        "translations": [{"paragraph_id": "p1", "english": "It rained."}],
        "vocabulary": [
            {"surface": f"語{i}", "reading": f"ご{i}", "meaning": f"word{i}", "paragraph_ids": ["p1"], "difficulty_rank": i, "distractors": ["a", "b", "c"]}
            for i in range(1, 5)
        ],
    })
    assert len(result.vocabulary) == 4


@pytest.mark.parametrize("english", [
    pytest.param(["It rained."], id="list"),
    pytest.param({"text": "It rained."}, id="dictionary"),
    pytest.param(123, id="number"),
    pytest.param(None, id="null"),
])
def test_translation_rejects_non_string_english(english):
    with pytest.raises(ValidationError) as error:
        Translation(paragraph_id="p1", english=english)

    failure = error.value.errors()[0]
    assert failure["loc"] == ("english",)
    assert failure["type"] == "string_type"


@pytest.mark.parametrize("missing_field", ["paragraph_id", "english"])
def test_translation_requires_both_fields(missing_field):
    payload = {"paragraph_id": "p1", "english": "It rained."}
    del payload[missing_field]

    with pytest.raises(ValidationError) as error:
        Translation.model_validate(payload)

    failure = error.value.errors()[0]
    assert failure["loc"] == (missing_field,)
    assert failure["type"] == "missing"


def test_translation_rejects_empty_english():
    with pytest.raises(ValidationError) as error:
        Translation(paragraph_id="p1", english="")

    assert error.value.errors()[0]["type"] == "string_too_short"


def test_translation_rejects_unexpected_fields():
    with pytest.raises(ValidationError) as error:
        Translation.model_validate({
            "paragraph_id": "p1",
            "english": "It rained.",
            "confidence": 0.9,
        })

    failure = error.value.errors()[0]
    assert failure["loc"] == ("confidence",)
    assert failure["type"] == "extra_forbidden"


@pytest.mark.parametrize("rank", [0, 101])
def test_vocabulary_rejects_out_of_range_difficulty(rank):
    with pytest.raises(ValidationError) as error:
        VocabularyCandidate(
            surface="雨", reading="あめ", meaning="rain", paragraph_ids=["p1"],
            difficulty_rank=rank, distractors=["snow", "wind", "sun"],
        )

    assert error.value.errors()[0]["loc"] == ("difficulty_rank",)


@pytest.mark.parametrize("distractors", [
    ["snow", "wind"],
    ["snow", "wind", "sun", "cloud"],
])
def test_vocabulary_requires_exactly_three_distractors(distractors):
    with pytest.raises(ValidationError) as error:
        VocabularyCandidate(
            surface="雨", reading="あめ", meaning="rain", paragraph_ids=["p1"],
            difficulty_rank=1, distractors=distractors,
        )

    assert error.value.errors()[0]["loc"] == ("distractors",)

