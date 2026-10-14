from types import SimpleNamespace

from audio_alignment import (
    AlignmentTarget,
    align_sentence_spans,
    alignment_targets,
    detected_speech_end,
    sentence_timings,
    split_html_sentences,
    spoken_reading_from_html,
    validate_final_cue_end,
)


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
    monkeypatch.setattr("audio_alignment.detected_speech_end", lambda _: 60.629)

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

    monkeypatch.setattr("audio_alignment.shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr("audio_alignment.subprocess.run", fake_run)

    assert detected_speech_end(tmp_path / "audio.mp3") == 59.3819


def test_final_cue_is_unchanged_when_the_remaining_audio_is_short(monkeypatch, tmp_path):
    final = SimpleNamespace(id="p1-s1", start=0.0, end=10.0)
    timings = {"p1": [final]}
    monkeypatch.setattr("audio_alignment.detected_speech_end", lambda _: 10.2)

    validate_final_cue_end(tmp_path / "audio.mp3", timings)

    assert final.end == 10.0
