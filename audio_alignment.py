"""Align NHK article sentences to narration using local Whisper timestamps.

align_audio() is the entry point. Helpers prepare ruby-aware sentence text,
choose ordered transcript spans, and produce playback cues without clipping.
This module depends on article models, not on the browser or DeepSeek client.
"""
from __future__ import annotations

import difflib
import html as html_lib
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import bleach

from models import ExtractedParagraph, SentenceManifest

SENTENCE_ENDINGS = "。！？!?"
ALIGNMENT_SCORE_WARNING = 0.65
ALIGNMENT_PREFIX_WARNING = 0.50
ALIGNMENT_PREFIX_CHARS = 8
ALIGNMENT_PRE_ROLL_SECONDS = 0.300
ALIGNMENT_SKIP_PENALTY = 0.01
FINAL_CUE_EXTENSION_MIN_SECONDS = 0.350
TAIL_SILENCE_NOISE_DB = "-45dB"
TAIL_SILENCE_MIN_SECONDS = "0.20"
TAIL_SILENCE_END_TOLERANCE_SECONDS = 0.100


def split_sentences(text: str) -> list[str]:
    """Split Japanese prose while retaining sentence-ending punctuation."""
    parts = re.findall(r"[^。！？!?]+[。！？!?]?", text.replace("\n", " "))
    return [part.strip() for part in parts if part.strip()]


def split_html_sentences(html: str) -> list[str]:
    """Split sanitized paragraph HTML at Japanese sentence ends without losing ruby."""
    tokens = re.split(r"(<[^>]+>)", html)
    stack: list[tuple[str, str]] = []
    output: list[str] = []
    current: list[str] = []
    in_rt = 0

    def close_current() -> None:
        nonlocal current
        value = "".join(current).strip()
        if value and spoken_text_from_html(value):
            output.append(value + "".join(f"</{name}>" for name, _ in reversed(stack)))
        current = [opening for _, opening in stack]

    for token in tokens:
        if not token:
            continue
        if token.startswith("<"):
            close = re.match(r"</\s*([a-z0-9]+)", token, re.I)
            opening = re.match(r"<\s*([a-z0-9]+)(?:\s[^>]*)?/?>", token, re.I)
            if close:
                name = close.group(1).lower()
                current.append(token)
                if name == "rt":
                    in_rt = max(0, in_rt - 1)
                for index in range(len(stack) - 1, -1, -1):
                    if stack[index][0] == name:
                        stack.pop(index)
                        break
            elif opening:
                name = opening.group(1).lower()
                current.append(token)
                if not token.rstrip().endswith("/>") and name != "br":
                    stack.append((name, token))
                    if name == "rt":
                        in_rt += 1
            else:
                current.append(token)
            continue
        for char in token:
            current.append(char)
            if not in_rt and char in SENTENCE_ENDINGS:
                close_current()
    value = "".join(current).strip()
    if value and spoken_text_from_html(value):
        output.append(value)
    return output


def spoken_text_from_html(html: str) -> str:
    """Remove ruby readings because the narrator speaks the base text only."""
    text = re.sub(r"<rt\b[^>]*>.*?</rt\s*>", "", html, flags=re.I | re.S)
    text = re.sub(r"<rp\b[^>]*>.*?</rp\s*>", "", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", "", text)
    return bleach.clean(text, tags=[], strip=True).replace("&nbsp;", " ").strip()


def spoken_reading_from_html(html: str) -> str:
    """Use NHK's ruby readings where available, retaining unannotated text."""
    def replace_ruby(match: re.Match[str]) -> str:
        ruby = match.group(0)
        readings = re.findall(r"<rt\b[^>]*>(.*?)</rt\s*>", ruby, flags=re.I | re.S)
        if readings:
            return "".join(re.sub(r"<[^>]+>", "", reading) for reading in readings)
        return spoken_text_from_html(ruby)

    text = re.sub(r"<ruby\b[^>]*>.*?</ruby\s*>", replace_ruby, html, flags=re.I | re.S)
    text = re.sub(r"<rt\b[^>]*>.*?</rt\s*>", "", text, flags=re.I | re.S)
    text = re.sub(r"<rp\b[^>]*>.*?</rp\s*>", "", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", "", text)
    return html_lib.unescape(text).replace("\xa0", " ").strip()


def alignment_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("マイナス", "-")
    normalized = "".join(
        chr(ord(character) - 0x60) if "ァ" <= character <= "ヶ" else character
        for character in normalized
    )
    return re.sub(r"[\s、。，．・「」『』（）()［］【】,:：;；！？!?…。、]", "", normalized)


@dataclass(frozen=True)
class AlignmentTarget:
    paragraph_id: str
    sentence_id: str
    japanese: str
    surface_key: str
    reading_key: str

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for value in (self.surface_key, self.reading_key) if value))


@dataclass(frozen=True)
class AlignmentSpan:
    start_word: int
    end_word: int
    score: float
    prefix_score: float
    transcript: str


def alignment_targets(paragraphs: list[ExtractedParagraph]) -> list[AlignmentTarget]:
    """Create surface and ruby-reading targets in article order."""
    targets: list[AlignmentTarget] = []
    for paragraph in paragraphs:
        spoken_sentences = split_sentences(spoken_text_from_html(paragraph.japanese_html))
        fragments = split_html_sentences(paragraph.japanese_html)
        readings: list[str] = []
        if len(fragments) == len(spoken_sentences) and all(
            alignment_key(spoken_text_from_html(fragment)) == alignment_key(sentence)
            for fragment, sentence in zip(fragments, spoken_sentences)
        ):
            readings = [spoken_reading_from_html(fragment) for fragment in fragments]
        else:
            readings = spoken_sentences
        for index, (sentence, reading) in enumerate(zip(spoken_sentences, readings), start=1):
            targets.append(AlignmentTarget(
                paragraph_id=paragraph.id,
                sentence_id=f"{paragraph.id}-s{index}",
                japanese=sentence,
                surface_key=alignment_key(sentence),
                reading_key=alignment_key(reading),
            ))
    return targets


def _span_similarity(target: AlignmentTarget, transcript: str) -> float:
    scores = []
    for variant in target.variants:
        similarity = difflib.SequenceMatcher(None, variant, transcript, autojunk=False).ratio()
        length_penalty = abs(len(transcript) - len(variant)) / max(1, len(variant)) * 0.04
        scores.append(similarity - length_penalty)
    return max(scores, default=-1.0)


def _prefix_similarity(target: AlignmentTarget, transcript: str) -> float:
    scores = []
    for variant in target.variants:
        width = min(ALIGNMENT_PREFIX_CHARS, len(variant))
        if width:
            transcript_prefix = transcript[:width]
            matcher = difflib.SequenceMatcher(
                None, variant[:width], transcript_prefix, autojunk=False,
            )
            first_match = next((block for block in matcher.get_matching_blocks() if block.size), None)
            if first_match is None:
                scores.append(0.0)
                continue
            # A high whole-prefix ratio can still hide omitted opening text
            # (for example 4月から6月... transcribed as 6月...). Penalize how
            # far into the expected prefix its first acoustic match appears.
            # If the first match begins after the expected prefix begins, the
            # ASR span may have dropped spoken opening words. Treat that as
            # uncertain rather than allowing a strong remainder to mask it.
            leading_coverage = 1.0 if first_match.a == 0 else 0.0
            length_coverage = min(1.0, len(transcript_prefix) / width)
            scores.append(matcher.ratio() * leading_coverage * length_coverage)
    return max(scores, default=0.0)


def _candidate_spans(word_keys: list[str], target: AlignmentTarget) -> list[AlignmentSpan]:
    lengths = [len(variant) for variant in target.variants]
    if not lengths:
        return []
    minimum = max(1, int(min(lengths) * 0.35))
    maximum = max(minimum, int(max(lengths) * 2.0))
    candidates: list[AlignmentSpan] = []
    for start in range(len(word_keys)):
        transcript = ""
        for end in range(start + 1, len(word_keys) + 1):
            transcript += word_keys[end - 1]
            if len(transcript) > maximum:
                break
            if len(transcript) < minimum:
                continue
            candidates.append(AlignmentSpan(
                start_word=start,
                end_word=end,
                score=_span_similarity(target, transcript),
                prefix_score=_prefix_similarity(target, transcript),
                transcript=transcript,
            ))
    return candidates


def align_sentence_spans(words: list, targets: list[AlignmentTarget]) -> list[AlignmentSpan]:
    """Globally align ordered sentences to non-overlapping ASR word spans."""
    if not targets:
        return []
    if len(words) < len(targets):
        raise RuntimeError("Whisper produced fewer timestamped words than article sentences")
    word_keys = [alignment_key(word.word) for word in words]
    word_lengths = [max(1, len(value)) for value in word_keys]
    # State key is the exclusive ending word index. Values retain the best total
    # score and chosen spans for all sentences processed so far.
    states: dict[int, tuple[float, tuple[AlignmentSpan, ...]]] = {0: (0.0, ())}
    for target in targets:
        next_states: dict[int, tuple[float, tuple[AlignmentSpan, ...]]] = {}
        candidates = _candidate_spans(word_keys, target)
        if not candidates:
            raise RuntimeError(f"No ASR span candidates were available for {target.sentence_id}")
        for candidate in candidates:
            best: tuple[float, tuple[AlignmentSpan, ...]] | None = None
            for previous_end, (previous_score, previous_spans) in states.items():
                if previous_end > candidate.start_word:
                    continue
                skipped = sum(word_lengths[previous_end:candidate.start_word])
                total = previous_score + candidate.score - skipped * ALIGNMENT_SKIP_PENALTY
                if best is None or total > best[0]:
                    best = (total, previous_spans + (candidate,))
            if best is None:
                continue
            existing = next_states.get(candidate.end_word)
            if existing is None or best[0] > existing[0]:
                next_states[candidate.end_word] = best
        if not next_states:
            raise RuntimeError(f"Could not monotonically align {target.sentence_id} to the narration")
        states = next_states
    best_result: tuple[float, tuple[AlignmentSpan, ...]] | None = None
    for ending, (score, spans) in states.items():
        trailing = sum(word_lengths[ending:])
        adjusted = score - trailing * ALIGNMENT_SKIP_PENALTY
        if best_result is None or adjusted > best_result[0]:
            best_result = (adjusted, spans)
    if best_result is None:
        raise RuntimeError("Could not align article sentences to the narration")
    return list(best_result[1])


def sentence_timings(words: list, targets: list[AlignmentTarget]) -> dict[str, list[SentenceManifest]]:
    """Turn aligned ASR spans into pause-safe, non-overlapping UI cues."""
    spans = align_sentence_spans(words, targets)
    spoken_ends: list[float] = []
    cue_starts: list[float] = []
    for index, (target, span) in enumerate(zip(targets, spans)):
        first_start = float(words[span.start_word].start)
        spoken_end = float(words[span.end_word - 1].end)
        if first_start < 0 or spoken_end <= first_start:
            raise RuntimeError(f"Whisper produced invalid timestamps for {target.sentence_id}")
        previous_spoken_end = spoken_ends[-1] if spoken_ends else 0.0
        prefix_uncertain = span.prefix_score <= ALIGNMENT_PREFIX_WARNING
        if prefix_uncertain:
            cue_start = previous_spoken_end if index else 0.0
        else:
            cue_start = max(previous_spoken_end, first_start - ALIGNMENT_PRE_ROLL_SECONDS)
        cue_starts.append(round(cue_start, 3))
        spoken_ends.append(spoken_end)
        warnings = []
        if span.score < ALIGNMENT_SCORE_WARNING:
            warnings.append(f"low score {span.score:.2f}")
        if prefix_uncertain:
            warnings.append(f"uncertain prefix {span.prefix_score:.2f}")
        if warnings:
            print(
                f"warning: alignment {'; '.join(warnings)} for {target.sentence_id}; "
                f"target={target.japanese!r}; transcript={span.transcript!r}; cue_start={cue_start:.3f}"
            )
    aligned: dict[str, list[SentenceManifest]] = {}
    for index, target in enumerate(targets):
        end = cue_starts[index + 1] if index + 1 < len(targets) else round(spoken_ends[index], 3)
        if end <= cue_starts[index]:
            raise RuntimeError(f"Alignment produced a non-positive cue for {target.sentence_id}")
        aligned.setdefault(target.paragraph_id, []).append(SentenceManifest(
            id=target.sentence_id,
            japanese=target.japanese,
            start=cue_starts[index],
            end=end,
        ))
    return aligned


def detected_speech_end(audio: Path) -> float | None:
    """Return the final non-silent point in an audio file, when ffmpeg can inspect it.

    This is deliberately best-effort: alignment remains available for direct MP3
    imports on machines without ffmpeg, and a failed media probe must not block a
    complete article bundle.
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("warning: ffmpeg/ffprobe unavailable; skipping final cue validation")
        return None
    duration_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        duration = float(duration_probe.stdout.strip())
    except ValueError:
        duration = 0.0
    if duration_probe.returncode != 0 or duration <= 0:
        print(f"warning: could not determine audio duration for {audio}; skipping final cue validation")
        return None
    detection = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(audio), "-af", f"silencedetect=noise={TAIL_SILENCE_NOISE_DB}:d={TAIL_SILENCE_MIN_SECONDS}", "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if detection.returncode != 0:
        print(f"warning: could not detect trailing silence for {audio}; skipping final cue validation")
        return None
    starts = list(re.finditer(r"silence_start: ([0-9.]+)", detection.stderr))
    ends = list(re.finditer(r"silence_end: ([0-9.]+)", detection.stderr))
    # ffmpeg may close a terminal silence at EOF, a few MP3 frames before the
    # container duration. Treat that as terminal too. If speech continues to
    # EOF, use the file duration as the speech boundary instead.
    terminal_silence = starts and (
        not ends
        or starts[-1].start() > ends[-1].start()
        or duration - float(ends[-1].group(1)) <= TAIL_SILENCE_END_TOLERANCE_SECONDS
    )
    if terminal_silence:
        return min(duration, float(starts[-1].group(1)))
    return duration


def validate_final_cue_end(audio: Path, timings: dict[str, list[SentenceManifest]]) -> None:
    """Extend a clearly premature final cue to the measured end of speech."""
    sentences = [sentence for paragraph in timings.values() for sentence in paragraph]
    if not sentences:
        return
    speech_end = detected_speech_end(audio)
    if speech_end is None:
        return
    final = sentences[-1]
    if speech_end - final.end < FINAL_CUE_EXTENSION_MIN_SECONDS:
        return
    original_end = final.end
    final.end = round(speech_end, 3)
    print(
        f"warning: extended final cue {final.id} from {original_end:.3f}s to {final.end:.3f}s "
        f"after detecting speech beyond its Whisper timestamp"
    )


def align_audio(audio: Path, paragraphs: list[ExtractedParagraph]) -> dict[str, list[SentenceManifest]]:
    """Align article sentences using local Whisper word timestamps."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("faster-whisper is required for sentence alignment; install it in the scraper environment") from error
    model = WhisperModel(
        os.environ.get("SCRAPER_WHISPER_MODEL", "small"),
        device=os.environ.get("SCRAPER_WHISPER_DEVICE", "cpu"),
        compute_type=os.environ.get("SCRAPER_WHISPER_COMPUTE", "int8"),
    )
    segments, _ = model.transcribe(str(audio), language="ja", beam_size=5, word_timestamps=True, vad_filter=True)
    words = [
        word for segment in segments for word in (segment.words or [])
        if word.start is not None and word.end is not None and alignment_key(word.word)
    ]
    if not words:
        raise RuntimeError("Whisper produced no word timestamps for the narration audio")
    timings = sentence_timings(words, alignment_targets(paragraphs))
    validate_final_cue_end(audio, timings)
    return timings
