import asyncio
from types import SimpleNamespace

from models import Enrichment
from nhk_easy_scraper import (
    AlignmentTarget,
    ARTICLE_LINK,
    align_sentence_spans,
    alignment_targets,
    capture_article_images,
    collect_images,
    detected_speech_end,
    grammar_candidates,
    grammar_key,
    safe_id,
    sentence_timings,
    split_html_sentences,
    spoken_reading_from_html,
    validate_final_cue_end,
    validate_grammar_selections,
)


ARTICLE_URL = "https://news.web.nhk/news/easy/20260818de44739/20260818de44739.html"
LEAD_IMG = {
    "src": "https://news.web.nhk/news/nd/kiji/20260817de44659/20260817de44659_E0000011297_1_l.jpg",
    "alt": "lead",
    "caption": "Caption",
}


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


def test_html_sentence_split_preserves_ruby():
    fragments = split_html_sentences("<ruby>日本<rt>にほん</rt></ruby>です。次です。")
    assert fragments == ["<ruby>日本<rt>にほん</rt></ruby>です。", "次です。"]


def test_spoken_reading_uses_ruby_and_retains_plain_text():
    assert spoken_reading_from_html("<ruby>八代市<rt>やつしろし</rt></ruby>では") == "やつしろしでは"


def test_alignment_targets_include_surface_and_ruby_reading_keys():
    paragraphs = [SimpleNamespace(
        id="p1",
        japanese_html="<ruby>八代市<rt>やつしろし</rt></ruby>です。",
    )]

    assert alignment_targets(paragraphs) == [AlignmentTarget(
        paragraph_id="p1",
        sentence_id="p1-s1",
        japanese="八代市です。",
        surface_key="八代市です",
        reading_key="やつしろしです",
    )]


def test_global_alignment_does_not_give_next_sentence_prefix_to_previous_sentence():
    words = [
        SimpleNamespace(word="aaa", start=0.0, end=0.5),
        SimpleNamespace(word="bbb", start=0.5, end=1.0),
        SimpleNamespace(word="x", start=1.2, end=1.4),
        SimpleNamespace(word="yz", start=1.4, end=1.8),
    ]
    targets = [
        AlignmentTarget("p1", "p1-s1", "aaabbbx", "aaabbbx", "aaabbbx"),
        AlignmentTarget("p1", "p1-s2", "xyz", "xyz", "xyz"),
    ]

    spans = align_sentence_spans(words, targets)

    assert (spans[0].start_word, spans[0].end_word) == (0, 2)
    assert (spans[1].start_word, spans[1].end_word) == (2, 4)


def test_confident_sentence_cues_get_bounded_300ms_preroll():
    words = [
        SimpleNamespace(word="first", start=0.5, end=1.0),
        SimpleNamespace(word="second", start=2.0, end=2.8),
    ]
    targets = [
        AlignmentTarget("p1", "p1-s1", "first", "first", "first"),
        AlignmentTarget("p1", "p1-s2", "second", "second", "second"),
    ]

    result = sentence_timings(words, targets)["p1"]

    assert [(sentence.start, sentence.end) for sentence in result] == [(0.2, 1.7), (1.7, 2.8)]


def test_missing_prefix_starts_at_previous_spoken_end_and_warns(capsys):
    words = [
        SimpleNamespace(word="first", start=0.2, end=1.0),
        SimpleNamespace(word="cd", start=2.0, end=2.5),
    ]
    targets = [
        AlignmentTarget("p1", "p1-s1", "first", "first", "first"),
        AlignmentTarget("p1", "p1-s2", "abcd", "abcd", "abcd"),
    ]

    result = sentence_timings(words, targets)["p1"]

    assert result[1].start == 1.0
    assert "uncertain prefix" in capsys.readouterr().out


def test_final_cue_extends_to_detected_trailing_speech(monkeypatch, tmp_path, capsys):
    timings = {"p1": [
        SimpleNamespace(id="p1-s1", start=0.0, end=2.0),
        SimpleNamespace(id="p1-s2", start=2.0, end=58.43),
    ]}
    monkeypatch.setattr("nhk_easy_scraper.detected_speech_end", lambda _: 60.629)

    validate_final_cue_end(tmp_path / "audio.mp3", timings)

    assert timings["p1"][-1].end == 60.629
    assert "extended final cue p1-s2" in capsys.readouterr().out


def test_detected_speech_end_uses_terminal_silence_before_mp3_duration(monkeypatch, tmp_path):
    def fake_run(command, **_kwargs):
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout="60.672\n", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="silence_start: 59.3819\nsilence_end: 60.6293 | silence_duration: 1.2474\n",
        )

    monkeypatch.setattr("nhk_easy_scraper.shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr("nhk_easy_scraper.subprocess.run", fake_run)

    assert detected_speech_end(tmp_path / "audio.mp3") == 59.3819


def test_final_cue_is_unchanged_when_the_remaining_audio_is_short(monkeypatch, tmp_path):
    final = SimpleNamespace(id="p1-s1", start=0.0, end=10.0)
    timings = {"p1": [final]}
    monkeypatch.setattr("nhk_easy_scraper.detected_speech_end", lambda _: 10.2)

    validate_final_cue_end(tmp_path / "audio.mp3", timings)

    assert final.end == 10.0


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
