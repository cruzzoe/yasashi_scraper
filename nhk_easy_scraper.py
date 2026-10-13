#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from zoneinfo import ZoneInfo

import bleach
from dotenv import load_dotenv
from openai import OpenAI
from playwright.async_api import BrowserContext, Page, async_playwright

from audio_alignment import align_audio, split_html_sentences, spoken_text_from_html
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
    VocabularyManifest,
)

ALLOWED_TAGS = ["ruby", "rt", "rp", "br", "span", "strong", "em"]
ARTICLE_LINK = re.compile(r"/news/easy/([A-Za-z0-9_-]+)/\1\.html(?:$|\?)")
GRAMMAR_INDEX_DEFAULT = Path(__file__).with_name("data") / "nihongopro-grammar-index.json"
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
    posters: list[str] = []

    def capture(response):
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
    easy_audio = await discover_easy_audio(page)
    if not easy_audio:
        raise RuntimeError("NHK Easy narration audio was not discovered; refusing to use embedded video audio")
    published = datetime.fromisoformat(time_value.replace("Z", "+00:00"))
    if published.tzinfo is None:
        published = published.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    return title, published, paragraphs, images, [easy_audio]


def require_deepseek_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    return api_key


def enrich(paragraphs: list[ExtractedParagraph], sentences: list[dict[str, str]], candidates: list[dict], model: str) -> Enrichment:
    api_key = require_deepseek_api_key()
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
        raise RuntimeError("DeepSeek paragraph translations did not align with the article")
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
    require_deepseek_api_key()
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
