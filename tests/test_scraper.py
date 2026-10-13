import asyncio
from types import SimpleNamespace

import pytest

from models import Enrichment
from nhk_easy_scraper import (
    ARTICLE_LINK,
    capture_article_images,
    collect_images,
    enrich,
    grammar_candidates,
    grammar_key,
    safe_id,
    run,
    validate_grammar_selections,
)


ARTICLE_URL = "https://news.web.nhk/news/easy/20260818de44739/20260818de44739.html"
LEAD_IMG = {
    "src": "https://news.web.nhk/news/nd/kiji/20260817de44659/20260817de44659_E0000011297_1_l.jpg",
    "alt": "lead",
    "caption": "Caption",
}


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_enrichment_requires_api_key_even_with_old_dummy_setting(monkeypatch, api_key):
    if api_key is None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    monkeypatch.setenv("SCRAPER_DUMMY_TRANSLATION", "true")

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is required"):
        enrich([], [], [], "test-model")


def test_missing_api_key_stops_run_before_browser_or_output_creation(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("SCRAPER_OUTPUT", str(tmp_path / "output"))

    def unexpected_browser_start():
        pytest.fail("Browser must not start without a DeepSeek API key")

    monkeypatch.setattr("nhk_easy_scraper.async_playwright", unexpected_browser_start)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY is required"):
        asyncio.run(run(SimpleNamespace(article_url=None, headed=False, no_publish=True)))

    assert not (tmp_path / "output").exists()


def test_enrichment_calls_deepseek_even_with_old_dummy_setting(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("SCRAPER_DUMMY_TRANSLATION", "true")
    payload = Enrichment.model_validate({
        "translations": [{"paragraph_id": "p1", "english": "It rained."}],
        "vocabulary": [
            {"surface": f"語{i}", "reading": f"ご{i}", "meaning": f"word{i}", "paragraph_ids": ["p1"], "difficulty_rank": i, "distractors": ["a", "b", "c"]}
            for i in range(1, 5)
        ],
    })
    requests = []

    def fake_create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=payload.model_dump_json()),
        )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr("nhk_easy_scraper.OpenAI", lambda **kwargs: client)
    paragraphs = [SimpleNamespace(id="p1", japanese_plain="雨が降りました。")]

    result = enrich(paragraphs, [], [], "test-model")

    assert result == payload
    assert len(requests) == 1
    assert requests[0]["model"] == "test-model"


class FakeLocator:
    def __init__(self, selector, result):
        self.selector = selector
        self.result = result

    async def evaluate_all(self, _expression):
        return self.result


class FakePage:
    url = ARTICLE_URL

    def __init__(self, figure_images):
        self.figure_locator = FakeLocator("figure#js-article-figure img", figure_images)

    def locator(self, selector):
        assert selector == "figure#js-article-figure img"
        return self.figure_locator


class FakeContainer:
    def __init__(self, images):
        self.img_locator = FakeLocator("img", images)

    def locator(self, selector):
        assert selector == "img"
        return self.img_locator


def capture(container_images=(), protected_images=(), figure_images=(), posters=()):
    page = FakePage(list(figure_images))
    container = FakeContainer(list(container_images))
    return asyncio.run(
        capture_article_images(page, container, list(protected_images), list(posters))
    )


def test_photo_only_article_lead_image_is_captured_from_legacy_figure():
    images = capture(figure_images=[LEAD_IMG])
    assert images == [LEAD_IMG]


def test_video_article_poster_is_still_captured():
    poster = "/news/u/news/nd/kiji/20260817de44662/20260817de44662_E0000010830_1_l.jpg"
    images = capture(figure_images=[], posters=[poster])
    assert images == [{"src": "https://news.web.nhk" + poster, "alt": "", "caption": ""}]


def test_new_template_protected_lead_image_is_still_captured():
    images = capture(protected_images=[LEAD_IMG])
    assert images == [LEAD_IMG]


def test_in_body_image_is_still_captured():
    images = capture(container_images=[LEAD_IMG])
    assert images == [LEAD_IMG]


def test_same_source_from_multiple_paths_is_deduplicated():
    relative = "/news/nd/kiji/20260817de44659/20260817de44659_E0000011297_1_l.jpg"
    images = capture(
        container_images=[{"src": relative, "alt": "", "caption": ""}],
        protected_images=[LEAD_IMG],
        figure_images=[LEAD_IMG],
        posters=[relative],
    )
    assert images == [{"src": LEAD_IMG["src"], "alt": "", "caption": ""}]


def test_capture_priority_order_matches_existing_paths():
    distinct = [
        {"src": "https://news.web.nhk/a.jpg", "alt": "", "caption": ""},
        {"src": "https://news.web.nhk/b.jpg", "alt": "", "caption": ""},
        {"src": "https://news.web.nhk/c.jpg", "alt": "", "caption": ""},
        {"src": "https://news.web.nhk/d.jpg", "alt": "", "caption": ""},
    ]
    images = capture(
        container_images=[distinct[0]],
        protected_images=[distinct[1]],
        figure_images=[distinct[2]],
        posters=[distinct[3]["src"]],
    )
    assert [image["src"] for image in images] == [item["src"] for item in distinct]


def test_relative_poster_url_is_resolved_against_article_page():
    poster = "/news/u/news/nd/kiji/x.jpg"
    images = capture(posters=[poster])
    assert images[0]["src"] == "https://news.web.nhk" + poster


def test_collect_images_with_no_candidates_returns_empty_list():
    assert collect_images([], [], [], [], ARTICLE_URL) == []


def test_article_pattern_requires_matching_article_ids():
    assert ARTICLE_LINK.search(
        "/news/easy/20260814de44051/20260814de44051.html"
    )
    assert not ARTICLE_LINK.search("/news/easy/article/disaster_typhoon.html")
    assert not ARTICLE_LINK.search("/news/easy/first/second.html")


def test_safe_id_uses_the_article_filename():
    assert safe_id(
        "https://news.web.nhk/news/easy/20260814de44051/20260814de44051.html"
    ) == "20260814de44051"


def test_grammar_candidates_and_validation_are_grounded_in_source_text():
    sentences = [{"id": "p1-s1", "japanese": "無料で見ることができます。"}]
    index = {"むりょうで": {"slug": "むりょうで", "term": "無料で", "meaning": "for free", "level": "N5", "aliases": ["無料で"]}}
    candidates = grammar_candidates(index, sentences)
    result = Enrichment.model_validate({
        "translations": [{"paragraph_id": "p1", "english": "It can be watched for free."}],
        "vocabulary": [{"surface": f"語{i}", "reading": f"ご{i}", "meaning": f"word{i}", "paragraph_ids": ["p1"], "difficulty_rank": i, "distractors": ["a", "b", "c"]} for i in range(1, 5)],
        "grammar": [{"evidence_id": candidates[0]["evidence"][0]["id"]}],
    })
    selected = validate_grammar_selections(result, candidates)
    assert selected == [{
        "id": candidates[0]["evidence"][0]["id"], "sentenceId": "p1-s1", "matchedText": "無料で",
        "start": 0, "end": 3, "source": "alias", "slug": "むりょうで",
    }]


def test_grammar_candidates_do_not_match_across_sentence_boundaries():
    index = {"ab": {"slug": "ab", "term": "ab", "meaning": "test", "level": "N5", "aliases": ["ab"]}}
    sentences = [{"id": "p1-s1", "japanese": "a。"}, {"id": "p1-s2", "japanese": "b。"}]

    assert grammar_candidates(index, sentences) == []


def test_grammar_validation_rejects_unknown_or_duplicate_evidence():
    index = {"むりょうで": {"slug": "むりょうで", "term": "無料で", "meaning": "for free", "level": "N5", "aliases": ["無料で"]}}
    candidates = grammar_candidates(index, [{"id": "p1-s1", "japanese": "無料で見ます。"}])
    evidence_id = candidates[0]["evidence"][0]["id"]
    base = {
        "translations": [{"paragraph_id": "p1", "english": "It is free."}],
        "vocabulary": [{"surface": f"語{i}", "reading": f"ご{i}", "meaning": f"word{i}", "paragraph_ids": ["p1"], "difficulty_rank": i, "distractors": ["a", "b", "c"]} for i in range(1, 5)],
    }
    unknown = Enrichment.model_validate({**base, "grammar": [{"evidence_id": "invented"}]})
    duplicate = Enrichment.model_validate({**base, "grammar": [{"evidence_id": evidence_id}, {"evidence_id": evidence_id}]})

    import pytest
    with pytest.raises(RuntimeError, match="unknown sentence-scoped evidence"):
        validate_grammar_selections(unknown, candidates)
    with pytest.raises(RuntimeError, match="duplicated sentence-scoped evidence"):
        validate_grammar_selections(duplicate, candidates)


def test_grammar_key_strips_tilde_variants():
    assert grammar_key("たり～たりする") == "たりたりする"
    assert grammar_key("〜てこそ") == "てこそ"
    assert grammar_key("～ばと思います") == "ばと思います"


def test_surface_pattern_adds_structural_candidates():
    sentences = [{"id": "p1-s1", "japanese": "水が入ったり、壊れたりしました。"}]
    index = {"たり-たりする": {"slug": "たり-たりする", "term": "たり～たりする", "meaning": "things like", "level": "N5", "aliases": ["たり～たりする"]}}
    candidates = grammar_candidates(index, sentences)
    assert [item["slug"] for item in candidates] == ["たり-たりする"]
    assert candidates[0]["evidence"][0]["matchedText"] == "たり、壊れたり"
    assert candidates[0]["evidence"][0]["start"] == 4
    assert candidates[0]["evidence"][0]["end"] == 11
