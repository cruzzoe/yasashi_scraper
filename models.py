from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedParagraph(StrictModel):
    id: str
    japanese_html: str
    japanese_plain: str


class Translation(StrictModel):
    paragraph_id: str
    english: str = Field(min_length=1)


class VocabularyCandidate(StrictModel):
    surface: str = Field(min_length=1)
    reading: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    paragraph_ids: list[str] = Field(min_length=1)
    difficulty_rank: int = Field(ge=1, le=100)
    distractors: list[str] = Field(min_length=3, max_length=3)

    @field_validator("distractors")
    @classmethod
    def unique_distractors(cls, value: list[str]) -> list[str]:
        if len(set(item.casefold() for item in value)) != 3:
            raise ValueError("distractors must be unique")
        return value


class GrammarSelection(StrictModel):
    """A model-selected, sentence-scoped grammar evidence record."""

    evidence_id: str = Field(min_length=1)


class Enrichment(StrictModel):
    translations: list[Translation] = Field(min_length=1)
    vocabulary: list[VocabularyCandidate] = Field(min_length=4, max_length=40)
    grammar: list[GrammarSelection] = Field(default_factory=list, max_length=5)


class ImageManifest(StrictModel):
    file: str
    alt: str
    caption: str = ""
    sourceUrl: HttpUrl


class SentenceManifest(StrictModel):
    id: str
    japanese: str
    # The original inline ruby markup for this sentence. Older bundles omit it.
    japaneseHtml: str = ""
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class ParagraphManifest(StrictModel):
    id: str
    japaneseHtml: str
    japanesePlain: str
    english: str
    sentences: list[SentenceManifest] = Field(default_factory=list)


class VocabularyManifest(StrictModel):
    id: str
    surface: str
    reading: str
    meaning: str
    paragraphIds: list[str]
    difficultyRank: int
    distractors: list[str]


class GrammarOccurrenceManifest(StrictModel):
    sentenceId: str
    matchedText: str
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def valid_range(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("grammar occurrence start and end must be provided together")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("grammar occurrence end must be after start")
        return self


class GrammarManifest(StrictModel):
    slug: str
    term: str
    meaning: str
    level: Literal["N5", "N4", "N3", "N2", "N1"]
    occurrences: list[GrammarOccurrenceManifest] = Field(min_length=1)


class MediaManifest(StrictModel):
    audioFile: Literal["audio.mp3"] = "audio.mp3"
    images: list[ImageManifest]


class ArticleManifest(StrictModel):
    manifestVersion: Literal[1] = 1
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{5,80}$")
    sourceUrl: HttpUrl
    title: str
    publishedAt: datetime
    generatedAt: datetime
    paragraphs: list[ParagraphManifest] = Field(min_length=1)
    vocabulary: list[VocabularyManifest] = Field(min_length=4, max_length=40)
    grammar: list[GrammarManifest] = Field(default_factory=list, max_length=5)
    media: MediaManifest
    checksums: dict[str, str]
