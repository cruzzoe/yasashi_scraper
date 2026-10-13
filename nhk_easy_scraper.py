#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import html as html_lib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from zoneinfo import ZoneInfo

import bleach
from dotenv import load_dotenv
from openai import OpenAI
from playwright.async_api import BrowserContext, Page, async_playwright

from models import (
    ArticleManifest,
    Enrichment,
    ExtractedParagraph,
    GrammarManifest,
    GrammarOccurrenceManifest,
    ImageManifest,
    MediaManifest,
    ParagraphManifest,
    SentenceManifest,
    Translation,
    VocabularyManifest,
    VocabularyCandidate,
)

ALLOWED_TAGS = ["ruby", "rt", "rp", "br", "span", "strong", "em"]
ARTICLE_LINK = re.compile(r"/news/easy/([A-Za-z0-9_-]+)/\1\.html(?:$|\?)")
GRAMMAR_INDEX_DEFAULT = Path(__file__).with_name("data") / "nihongopro-grammar-index.json"
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
GRAMMAR_IGNORED_CHARACTERS = re.compile(r"[\s~、。，．・「」『』（）()［］【】,:：;；！？!?…。]")

# Curated surface patterns for grammar whose canonical aliases are Bunpro
# pattern placeholders (e.g. "Verbたり") or fixed dictionary forms that never
# appear verbatim in real sentences (e.g. たり…たり, ば…ほど). A match adds the
# entry to the candidate pool even though pure substring alias matching fails.
SURFACE_PATTERNS: dict[str, tuple[str, ...]] = {
    "たり-たりする": (r"たり.*たり",),
    "ば-ほど": (r"ば.*ほど",),
    "ことにする": (r"ことに(する|した|しました|している)",),
    "そうだ": (r"そう(だ|で|です|な|に)",),
    "ながら": (r"ながら",),
    "てほしい": (r"てほしい",),
    "までに": (r"までに",),
}


def grammar_key(value: str) -> str:
    """Normalize Japanese text enough for safe, conservative substring matching."""
    normalized = unicodedata.normalize("NFKC", value).lower()
    # NFKC turns full-width ～ into ASCII ~, which would otherwise survive the
    # punctuation strip and make literal aliases like たり～たりする unmatchable.
    normalized = normalized.replace("〜", "～").replace("～", "~")
    return GRAMMAR_IGNORED_CHARACTERS.sub("", normalized)


def grammar_key_with_offsets(value: str) -> tuple[str, list[int]]:
    """Normalize text like grammar_key while retaining source character offsets."""
    key: list[str] = []
    offsets: list[int] = []
    for offset, char in enumerate(value):
        normalized = unicodedata.normalize("NFKC", char).lower().replace("〜", "～").replace("～", "~")
        for normalized_char in normalized:
            if GRAMMAR_IGNORED_CHARACTERS.fullmatch(normalized_char):
                continue
            key.append(normalized_char)
            offsets.append(offset)
    return "".join(key), offsets


def load_grammar_index(path: Path | None = None) -> dict[str, dict]:
    """Load the portable NihongoPro grammar index produced by its export script."""
    source = path or Path(os.environ.get("NIHONGOPRO_GRAMMAR_INDEX", GRAMMAR_INDEX_DEFAULT))
    if not source.is_file():
        print(f"warning: NihongoPro grammar index not found at {source}; skipping grammar enrichment")
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"NihongoPro grammar index is invalid JSON: {source}") from error
    if payload.get("version") != 1 or not isinstance(payload.get("items"), list):
        raise RuntimeError(f"NihongoPro grammar index has an unsupported format: {source}")
    items: dict[str, dict] = {}
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise RuntimeError("NihongoPro grammar index contains an invalid item")
        slug, term, meaning, level = (item.get(key) for key in ("slug", "term", "meaning", "level"))
        aliases = item.get("aliases", [])
        if not all(isinstance(value, str) and value for value in (slug, term, meaning)) or level not in {"N5", "N4", "N3", "N2", "N1"} or not isinstance(aliases, list):
            raise RuntimeError("NihongoPro grammar index item is missing required fields")
        if slug in items:
            raise RuntimeError(f"NihongoPro grammar index has a duplicate slug: {slug}")
        items[slug] = {"slug": slug, "term": term, "meaning": meaning, "level": level,
                       "aliases": [alias for alias in aliases
                                   if isinstance(alias, str)
                                   and not re.search(r"[A-Za-z0-9]", alias)
                                   and len(grammar_key(alias)) >= 2]}
    return items


def grammar_candidates(index: dict[str, dict], sentences: list[dict[str, str]], limit: int = 40) -> list[dict]:
    """Build sentence-scoped, position-aware grammar evidence for DeepSeek."""
    matches: list[tuple[int, dict]] = []
    for item in index.values():
        evidence: list[dict] = []
        matched_aliases: list[str] = []
        for sentence in sentences:
            sentence_key, offsets = grammar_key_with_offsets(sentence["japanese"])
            for alias in item["aliases"]:
                alias_key = grammar_key(alias)
                start = sentence_key.find(alias_key)
                while start != -1:
                    end = start + len(alias_key)
                    raw_start, raw_end = offsets[start], offsets[end - 1] + 1
                    evidence.append({
                        "sentenceId": sentence["id"], "matchedText": sentence["japanese"][raw_start:raw_end],
                        "start": raw_start, "end": raw_end, "source": "alias",
                    })
                    matched_aliases.append(alias)
                    start = sentence_key.find(alias_key, start + 1)
        if evidence:
            unique = {(entry["sentenceId"], entry["start"], entry["end"]): entry for entry in evidence}
            matches.append((max(len(grammar_key(alias)) for alias in matched_aliases), {
                **item, "aliases": list(dict.fromkeys(matched_aliases))[:8], "evidence": list(unique.values()),
            }))
    matched_slugs = {item["slug"] for _, item in matches}
    for slug, patterns in SURFACE_PATTERNS.items():
        item = index.get(slug)
        if item is None or slug in matched_slugs:
            continue
        evidence = []
        for sentence in sentences:
            for pattern in patterns:
                for match in re.finditer(pattern, sentence["japanese"]):
                    evidence.append({
                        "sentenceId": sentence["id"], "matchedText": match.group(),
                        "start": match.start(), "end": match.end(), "source": "pattern",
                    })
        if evidence:
            # Surface-pattern hits are real structural matches; rank them above
            # substring noise so the candidate cap cannot crowd them out.
            matches.append((10_000, {**item, "aliases": item["aliases"][:8], "evidence": evidence}))
    matches.sort(key=lambda entry: (-entry[0], entry[1]["level"], entry[1]["term"]))
    candidates = [item for _, item in matches[:limit]]
    for candidate_index, candidate in enumerate(candidates, start=1):
        for evidence_index, evidence in enumerate(candidate["evidence"], start=1):
            evidence["id"] = f"g{candidate_index}e{evidence_index}"
    return candidates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(url: str) -> str:
    name = Path(urlparse(url).path).stem
    if not re.fullmatch(r"[A-Za-z0-9_-]{5,80}", name):
        raise RuntimeError(f"Could not derive a safe article ID from {url}")
    return name


async def accept_usage_gate(page: Page) -> None:
    start_buttons = page.get_by_role("button", name="サービスの利用を開始する", exact=True)
    start_button = None
    for index in range(await start_buttons.count()):
        candidate = start_buttons.nth(index)
        if await candidate.is_visible():
            start_button = candidate
            break
    if start_button is None:
        return

    area = page.locator('select[aria-label="地域選択"]')
    if await area.count():
        checkbox = page.locator('input[type="checkbox"]').first
        try:
            await checkbox.wait_for(state="visible", timeout=10_000)
            if not await checkbox.is_checked():
                await checkbox.check()
            await area.first.evaluate(
                "element => new Promise((resolve, reject) => { const timeout = setTimeout(() => reject(new Error('region options did not load')), 10000); const ready = () => { if (!element.disabled && element.options.length) { clearTimeout(timeout); resolve(true); return true; } return false; }; if (ready()) return; new MutationObserver((_, observer) => { if (ready()) observer.disconnect(); }).observe(element, {attributes: true, childList: true}); })"
            )
            await area.first.select_option(os.environ.get("NHK_AREA_INDEX", "18"), timeout=10_000)
        except Exception as error:
            raise RuntimeError("Could not complete NHK's usage gate; check NHK_AREA_INDEX") from error
    await start_button.click(timeout=10_000)
    await page.wait_for_timeout(1800)


async def wait_for_nhk_auth(page: Page, timeout: float = 25.0) -> None:
    """Wait for NHK's accountless authentication to finish after the usage gate.

    Article images are served behind a JWT-protected endpoint. NHK's page JS only
    completes the auth round-trip (setting the z_at/authz_type cookies) after the
    usage gate, so media requests must wait for those cookies to appear.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        names = {cookie["name"] for cookie in await page.context.cookies()}
        if "z_at" in names and "authz_type" in names:
            return
        await page.wait_for_timeout(500)
    raise RuntimeError("NHK accountless authentication did not complete in time")


async def discover_top_article(page: Page, homepage: str) -> str:
    # The consent form is React-controlled; wait for hydration before changing it.
    await page.goto(homepage, wait_until="networkidle", timeout=60_000)
    await accept_usage_gate(page)
    handle = await page.wait_for_function(
        "config => [...document.querySelectorAll('a[href]')].find(a => a.offsetParent !== null && new RegExp(config.source, config.flags).test(new URL(a.href).pathname))?.href || null",
        arg={"source": ARTICLE_LINK.pattern, "flags": ""},
        timeout=60_000,
    )
    candidate = await handle.json_value()
    if isinstance(candidate, str) and ARTICLE_LINK.search(urlparse(candidate).path):
        return candidate
    raise RuntimeError("No primary NHK Easy article link was found; run with --headed to inspect the page")


async def article_container(page: Page):
    selectors = [
        "#js-article-body",
        ".article-main__body",
        "main article",
        "article",
        "main",
    ]
    for selector in selectors:
        locator = page.locator(selector).first
        if await locator.count() and await locator.locator("p").count():
            return locator
    raise RuntimeError("Could not identify the article body")


async def discover_easy_audio(page: Page) -> str | None:
    links = page.locator('a[href="#"]').filter(has_text=re.compile(r"ニュース.*聞"))
    for index in range(await links.count()):
        link = links.nth(index)
        if not await link.is_visible():
            continue
        # NHK's click handler is attached to the anchor and can be bypassed by
        # an overlapping video iframe in headless Chromium; dispatch it in the
        # page so the dedicated audio-v6 iframe is always opened.
        await link.evaluate("(element) => element.click()")
        # The audio iframe loads its player markup asynchronously; poll long
        # enough for the frame and its data-hlsurl attribute to appear.
        hls = None
        for _ in range(20):
            for frame in page.frames:
                if "/player/audio-v6.html" not in frame.url:
                    continue
                player = frame.locator(".nol_audio_player[data-hlsurl]").first
                if await player.count():
                    candidate = await player.get_attribute("data-hlsurl")
                    if candidate and "/news/easy_audio/" in candidate:
                        hls = candidate
                        break
            if hls:
                break
            await page.wait_for_timeout(500)
        if not hls:
            continue
        if hls:
            token_response = await page.context.request.get(
                "https://mediatoken.web.nhk/v1/token",
                headers={"Referer": page.url},
                timeout=30_000,
            )
            if not token_response.ok:
                raise RuntimeError(f"NHK media token request failed ({token_response.status})")
            token = (await token_response.json()).get("token")
            if not token:
                raise RuntimeError("NHK media token response did not contain a token")
            return f"{hls}{'&' if '?' in hls else '?'}hdnts={token}"
    return None


def collect_images(
    container_images: list[dict[str, str]],
    protected_images: list[dict[str, str]],
    figure_images: list[dict[str, str]],
    posters: list[str],
    base_url: str,
) -> list[dict[str, str]]:
    """Merge image candidates from all discovery paths into a deduplicated list.

    Priority order is preserved from the original extraction logic: in-body
    images first, then the new-template protected lead image, then the legacy
    template's photo-only lead image, then video posters. Duplicate sources are
    collapsed so an article that matches several paths still yields one entry
    per image.
    """
    images: list[dict[str, str]] = []
    seen_sources: set[str] = set()

    def add_image(src: str, alt: str = "", caption: str = "") -> None:
        absolute = urljoin(base_url, src)
        if not absolute or absolute in seen_sources:
            return
        seen_sources.add(absolute)
        images.append({"src": absolute, "alt": alt, "caption": caption})

    for item in container_images:
        add_image(item["src"], item["alt"], item["caption"])
    for item in protected_images:
        add_image(item["src"], item["alt"], item["caption"])
    for item in figure_images:
        add_image(item["src"], item["alt"], item["caption"])
    for poster in posters:
        add_image(poster)
    return images


async def capture_article_images(
    page: Page,
    container,
    protected_images: list[dict[str, str]],
    posters: list[str],
) -> list[dict[str, str]]:
    """Collect every image the rendered article exposes to the scraper.

    The legacy NHK Easy template keeps a photo-only article's lead image in
    ``figure#js-article-figure`` — outside the article body — as a plain
    ``<img>`` with no ``data-image-protection`` attribute and no video poster.
    Video articles use the same figure but with an ``<iframe>``, so this scan
    only ever adds the plain-image lead for photo-only articles.
    """
    container_images = await container.locator("img").evaluate_all(
        "els => els.filter(e => e.offsetParent !== null && e.naturalWidth >= 300).map(e => ({src: e.currentSrc || e.src, alt: e.alt || '', caption: e.closest('figure')?.querySelector('figcaption')?.innerText || ''}))"
    )
    figure_images = await page.locator("figure#js-article-figure img").evaluate_all(
        "els => els.filter(e => e.offsetParent !== null).map(e => ({src: e.currentSrc || e.src, alt: e.alt || '', caption: e.closest('figure')?.querySelector('figcaption')?.innerText || ''}))"
    )
    return collect_images(container_images, protected_images, figure_images, posters, page.url)


async def extract_article(page: Page, url: str):
    media_urls: list[str] = []
    posters: list[str] = []

    def capture(response):
        clean = response.url.split("#", 1)[0]
        if re.search(r"\.(?:mp3|m3u8)(?:\?|$)", clean, re.I):
            media_urls.append(clean)
        match = re.search(r"mrposter=([^&]+)", response.url)
        if match:
            posters.append(unquote(match.group(1)))

    page.on("response", capture)
    await page.goto(url, wait_until="networkidle", timeout=60_000)
    # The article's lead image exists as a protected <img> before NHK's
    # accountless auth rewrites the page into its video-player form.
    protected_images = await page.locator("img[data-image-protection]").evaluate_all(
        "els => els.map(e => ({src: e.currentSrc || e.src, alt: e.alt || '', caption: e.closest('figure')?.querySelector('figcaption')?.innerText || ''}))"
    )
    await accept_usage_gate(page)
    await page.wait_for_load_state("networkidle", timeout=30_000)
    await wait_for_nhk_auth(page)
    title = (await page.locator("h1").first.inner_text()).replace(" | NHKやさしいことばニュース", "").strip()
    container = await article_container(page)
    raw_paragraphs = await container.locator("p").evaluate_all(
        """els => els.filter(e => e.offsetParent !== null && e.innerText.trim().length > 1)
          .map(e => ({html: e.innerHTML, plain: e.innerText.trim()}))"""
    )
    ignored = ("ご利用にあたって", "受信契約", "サービスの利用", "シェア", "Copyright")
    paragraphs: list[ExtractedParagraph] = []
    for item in raw_paragraphs:
        if any(term in item["plain"] for term in ignored):
            continue
        html = bleach.clean(item["html"], tags=ALLOWED_TAGS, attributes={}, strip=True).strip()
        if item["plain"] and html:
            paragraphs.append(ExtractedParagraph(id=f"p{len(paragraphs) + 1}", japanese_html=html, japanese_plain=item["plain"]))
    if not paragraphs:
        raise RuntimeError("No article paragraphs were extracted")

    time_value = None
    time_locator = page.locator("time[datetime]").first
    if await time_locator.count():
        time_value = await time_locator.get_attribute("datetime")
    if not time_value:
        match = re.search(r"(20\d{2})(\d{2})(\d{2})", safe_id(url))
        time_value = f"{match[1]}-{match[2]}-{match[3]}T00:00:00+09:00" if match else datetime.now(UTC).isoformat()

    images = await capture_article_images(page, container, protected_images, posters)
    audio = await page.locator("audio source, audio").evaluate_all(
        "els => els.map(e => e.src || e.currentSrc).filter(Boolean)"
    )
    media_urls.extend(audio)
    easy_audio = await discover_easy_audio(page)
    if easy_audio:
        media_urls = [easy_audio]
    elif not media_urls:
        for pattern in (r"ニュースを聞く", r"音声", r"再生"):
            button = page.get_by_role("button", name=re.compile(pattern)).first
            if await button.count() and await button.is_visible():
                await button.click(force=True)
                await page.wait_for_timeout(2500)
                break
    if not easy_audio:
        raise RuntimeError("NHK Easy narration audio was not discovered; refusing to use embedded video audio")
    published = datetime.fromisoformat(time_value.replace("Z", "+00:00"))
    if published.tzinfo is None:
        published = published.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    return title, published, paragraphs, images, list(dict.fromkeys(media_urls))


def enrich(paragraphs: list[ExtractedParagraph], sentences: list[dict[str, str]], candidates: list[dict], model: str) -> Enrichment:
    if os.environ.get("SCRAPER_DUMMY_TRANSLATION", "false").lower() == "true":
        surfaces: list[str] = []
        for paragraph in paragraphs:
            surfaces.extend(re.findall(r"[\u4e00-\u9fff]{2,}", paragraph.japanese_plain))
        unique = list(dict.fromkeys(surfaces))
        while len(unique) < 4:
            unique.append(f"学習語{len(unique) + 1}")
        vocabulary = [
            VocabularyCandidate(
                surface=surface,
                reading="(reading unavailable)",
                meaning="Dummy translation — add DEEPSEEK_API_KEY later",
                paragraph_ids=[paragraphs[index % len(paragraphs)].id],
                difficulty_rank=index + 1,
                distractors=["Dummy option A", "Dummy option B", "Dummy option C"],
            )
            for index, surface in enumerate(unique[:12])
        ]
        return Enrichment(
            translations=[
                Translation(
                    paragraph_id=paragraph.id,
                    english="[Dummy translation — configure DEEPSEEK_API_KEY for English.]",
                )
                for paragraph in paragraphs
            ],
            vocabulary=vocabulary,
            grammar=[],
        )
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required unless SCRAPER_DUMMY_TRANSLATION=true")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    payload = {
        "paragraphs": [{"id": p.id, "japanese": p.japanese_plain} for p in paragraphs],
        "sentences": sentences,
        "grammarCandidates": [{
            **{key: item[key] for key in ("slug", "term", "meaning", "level")},
            "evidence": [{key: evidence[key] for key in ("id", "sentenceId", "matchedText")} for evidence in item["evidence"]],
        } for item in candidates],
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You create accurate study material for intermediate learners of Japanese. "
                "Translate every paragraph naturally into English, preserving its paragraph ID. "
                "Select 15-25 useful difficult words or fixed expressions actually present in the text. "
                "Give kana readings, concise contextual meanings, paragraph IDs, a unique difficulty rank, "
                "and exactly three plausible but clearly incorrect English quiz distractors. Do not add facts. "
                "For grammar, select at most five entries ONLY by their grammarCandidates evidence ID. Each evidence "
                "record is already grounded in one exact sentence span. It is fine to return no grammar selections. "
                "Do not invent, rename, or guess evidence IDs. "
                "Return only one valid JSON object matching this schema: " + json.dumps(Enrichment.model_json_schema(), ensure_ascii=False)
            )},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("DeepSeek returned no structured study material")
    result = Enrichment.model_validate_json(content)
    expected = {p.id for p in paragraphs}
    actual = {item.paragraph_id for item in result.translations}
    if actual != expected or len(result.translations) != len(paragraphs):
        raise RuntimeError("OpenAI paragraph translations did not align with the article")
    for word in result.vocabulary:
        if not set(word.paragraph_ids).issubset(expected):
            raise RuntimeError(f"Vocabulary item {word.surface} references an unknown paragraph")
        if word.meaning.casefold() in {item.casefold() for item in word.distractors}:
            raise RuntimeError(f"Vocabulary item {word.surface} has its answer among distractors")
    validate_grammar_selections(result, candidates)
    return result


def grammar_evidence_by_id(candidates: list[dict]) -> dict[str, dict]:
    """Flatten candidate evidence and fail closed if a scraper bug creates duplicate IDs."""
    evidence: dict[str, dict] = {}
    for candidate in candidates:
        for entry in candidate["evidence"]:
            if entry["id"] in evidence:
                raise RuntimeError(f"Duplicate grammar evidence ID: {entry['id']}")
            evidence[entry["id"]] = {**entry, "slug": candidate["slug"]}
    return evidence


def validate_grammar_selections(result: Enrichment, candidates: list[dict]) -> list[dict]:
    """Resolve model selections only from locally generated sentence evidence."""
    available = grammar_evidence_by_id(candidates)
    seen: set[str] = set()
    resolved: list[dict] = []
    for selection in result.grammar:
        if selection.evidence_id not in available:
            raise RuntimeError("DeepSeek grammar selection references unknown sentence-scoped evidence")
        if selection.evidence_id in seen:
            raise RuntimeError("DeepSeek grammar selection duplicated sentence-scoped evidence")
        seen.add(selection.evidence_id)
        resolved.append(available[selection.evidence_id])
    return resolved


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


def attach_sentence_html(paragraph: ExtractedParagraph, sentences: list[SentenceManifest]) -> None:
    """Attach ruby-preserving sentence fragments when their text split is unambiguous."""
    fragments = split_html_sentences(paragraph.japanese_html)
    if len(fragments) != len(sentences):
        print(f"warning: could not map ruby HTML to sentences for {paragraph.id}; using text fallback")
        return
    for sentence, fragment in zip(sentences, fragments):
        if grammar_key(spoken_text_from_html(fragment)) == grammar_key(sentence.japanese):
            sentence.japaneseHtml = fragment
        else:
            print(f"warning: ruby HTML did not match sentence text for {sentence.id}; using text fallback")


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


async def download_with_context(context: BrowserContext, url: str, target: Path) -> str:
    response = await context.request.get(url, headers={"Referer": "https://news.web.nhk/"}, timeout=60_000)
    if not response.ok:
        raise RuntimeError(f"Download failed ({response.status}) for {url}")
    target.write_bytes(await response.body())
    return response.headers.get("content-type", "")


async def download_audio(context: BrowserContext, media_urls: list[str], target: Path) -> None:
    mp3 = next((url for url in media_urls if re.search(r"\.mp3(?:\?|$)", url, re.I)), None)
    if mp3:
        await download_with_context(context, mp3, target)
        return
    hls = next((url for url in media_urls if re.search(r"\.m3u8(?:\?|$)", url, re.I)), None)
    if not hls:
        raise RuntimeError("No MP3 or HLS audio source was discovered")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to convert NHK HLS audio to MP3")
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-analyzeduration", "10M", "-probesize", "10M",
        "-y", "-i", hls, "-map", "0:a:0", "-vn", "-codec:a", "libmp3lame", "-q:a", "2", str(target),
    ], check=True)


async def build_bundle(context: BrowserContext, url: str, title: str, published: datetime, paragraphs: list[ExtractedParagraph], images: list[dict], media_urls: list[str], output: Path, model: str) -> Path:
    article_id = safe_id(url)
    bundle = output / article_id
    if bundle.exists():
        raise RuntimeError(f"Output already exists: {bundle}")
    (bundle / "images").mkdir(parents=True)
    await download_audio(context, media_urls, bundle / "audio.mp3")
    sentence_timings = align_audio(bundle / "audio.mp3", paragraphs)
    for paragraph in paragraphs:
        attach_sentence_html(paragraph, sentence_timings[paragraph.id])
    image_manifests: list[ImageManifest] = []
    for index, image in enumerate(images[:20], start=1):
        parsed = urlparse(image["src"])
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        relative = f"images/{index:02d}{suffix}"
        await download_with_context(context, image["src"], bundle / relative)
        image_manifests.append(ImageManifest(file=relative, alt=image["alt"], caption=image["caption"], sourceUrl=image["src"]))
    sentence_payload = [
        {"id": sentence.id, "japanese": sentence.japanese}
        for paragraph in paragraphs for sentence in sentence_timings[paragraph.id]
    ]
    grammar_index = load_grammar_index()
    candidates = grammar_candidates(grammar_index, sentence_payload)
    study = enrich(paragraphs, sentence_payload, candidates, model)
    selected_grammar = validate_grammar_selections(study, candidates)
    translations = {item.paragraph_id: item.english for item in study.translations}
    paragraph_manifests = [ParagraphManifest(id=p.id, japaneseHtml=p.japanese_html, japanesePlain=p.japanese_plain, english=translations[p.id], sentences=sentence_timings[p.id]) for p in paragraphs]
    vocab_manifests = [VocabularyManifest(id=f"v{index}", surface=w.surface, reading=w.reading, meaning=w.meaning, paragraphIds=w.paragraph_ids, difficultyRank=w.difficulty_rank, distractors=w.distractors) for index, w in enumerate(sorted(study.vocabulary, key=lambda item: item.difficulty_rank), start=1)]
    grammar_by_slug: dict[str, GrammarManifest] = {}
    for selection in selected_grammar:
        item = grammar_index[selection["slug"]]
        existing = grammar_by_slug.get(selection["slug"])
        occurrence = GrammarOccurrenceManifest(
            sentenceId=selection["sentenceId"], matchedText=selection["matchedText"],
            start=selection["start"], end=selection["end"],
        )
        if existing:
            existing.occurrences.append(occurrence)
        else:
            grammar_by_slug[selection["slug"]] = GrammarManifest(
                slug=item["slug"], term=item["term"], meaning=item["meaning"], level=item["level"], occurrences=[occurrence],
            )
    files = [bundle / "audio.mp3", *(bundle / image.file for image in image_manifests)]
    checksums = {str(path.relative_to(bundle)): sha256(path) for path in files}
    manifest = ArticleManifest(id=article_id, sourceUrl=url, title=title, publishedAt=published, generatedAt=datetime.now(UTC), paragraphs=paragraph_manifests, vocabulary=vocab_manifests, grammar=list(grammar_by_slug.values()), media=MediaManifest(images=image_manifests), checksums=checksums)
    (bundle / "article.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    archive = output / f"{article_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(bundle)), recursive=False)
    return archive


def publish(archive: Path) -> None:
    target = os.environ.get("PUBLISH_SSH_TARGET")
    incoming = os.environ.get("PUBLISH_INCOMING_PATH")
    command = os.environ.get("PUBLISH_COMMAND", "sudo -u nhkreader /usr/local/bin/nhk-reader-publish")
    if not target or not incoming:
        print(f"Bundle ready at {archive}; publishing is not configured")
        return
    key = str(Path(os.environ.get("PUBLISH_SSH_KEY", "~/.ssh/nhk-reader-publisher")).expanduser())
    subprocess.run(["scp", "-i", key, str(archive), f"{target}:{incoming}/{archive.name}"], check=True)
    subprocess.run(["ssh", "-i", key, target, *command.split(), f"{incoming}/{archive.name}"], check=True)


async def run(args: argparse.Namespace) -> None:
    output = Path(os.environ.get("SCRAPER_OUTPUT", "./output")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = Path(os.environ.get("SCRAPER_BROWSER_STATE", "./browser-state")).resolve()
    async with async_playwright() as playwright:
        executable = os.environ.get("SCRAPER_CHROMIUM_EXECUTABLE") or None
        context = await playwright.chromium.launch_persistent_context(
            str(state), headless=not args.headed, locale="ja-JP", executable_path=executable,
        )
        page = context.pages[0] if context.pages else await context.new_page()
        url = args.article_url or await discover_top_article(page, os.environ.get("NHK_EASY_URL", "https://news.web.nhk/news/easy/"))
        title, published, paragraphs, images, media_urls = await extract_article(page, url)
        archive = await build_bundle(context, url, title, published, paragraphs, images, media_urls, output, os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
        await context.close()
    print(f"Created {archive}")
    if not args.no_publish:
        publish(archive)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Import the top NHK Easy article")
    parser.add_argument("--article-url", help="Import a specific article instead of the homepage's first article")
    parser.add_argument("--headed", action="store_true", help="Show Chromium for selector diagnostics")
    parser.add_argument("--no-publish", action="store_true", help="Create the bundle without SCP publishing")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
