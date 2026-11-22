"""Human-annotated Golden Dataset contracts and completeness gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..domain.entities.screening_decision import STANDARD_EXCLUSION_REASONS


class HumanAnnotation(BaseModel):
    annotator_id: str = Field(min_length=1)
    annotated_at: datetime
    adjudicated: bool = False


class GoldenPublication(BaseModel):
    publication_id: str = Field(min_length=1)
    citation: str = Field(min_length=5)
    doi: str | None = None
    pdf_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScreeningLabel(BaseModel):
    publication_id: str
    decision: Literal["include", "exclude"]
    exclusion_reason: str | None = None
    annotation: HumanAnnotation

    @model_validator(mode="after")
    def validate_reason(self) -> ScreeningLabel:
        if self.decision == "exclude" and not self.exclusion_reason:
            raise ValueError("Excluded screening labels require a standardized reason")
        if self.decision == "exclude" and self.exclusion_reason not in STANDARD_EXCLUSION_REASONS:
            raise ValueError("Screening exclusion reason must use a standardized domain code")
        return self


class ValidatedExtraction(BaseModel):
    publication_id: str
    field_name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    annotation: HumanAnnotation


class GoldenQuestion(BaseModel):
    question_id: str = Field(pattern=r"^q[0-9]{2,}$")
    question: str = Field(min_length=10)
    source_slr_citation: str = Field(min_length=5)
    source_slr_url: str = Field(min_length=8)
    reference_answer: str = Field(min_length=10)
    publications: list[GoldenPublication] = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    screening_labels: list[ScreeningLabel] = Field(min_length=1)
    validated_extractions: list[ValidatedExtraction] = Field(min_length=1)
    relevance_annotation: HumanAnnotation

    @model_validator(mode="after")
    def validate_publication_references(self) -> GoldenQuestion:
        publication_ids = {publication.publication_id for publication in self.publications}
        if any(label.publication_id not in publication_ids for label in self.screening_labels):
            raise ValueError("Every screening annotation must reference a listed publication")
        if any(
            extraction.publication_id not in publication_ids
            for extraction in self.validated_extractions
        ):
            raise ValueError("Every validated extraction must reference a listed publication")
        annotated_chunk_ids = set(self.relevant_chunk_ids)
        if any(
            extraction.source_chunk_id not in annotated_chunk_ids
            for extraction in self.validated_extractions
        ):
            raise ValueError(
                "Validated extraction source chunks must be present in the annotated gold passages"
            )
        return self


class GoldenDataset(BaseModel):
    dataset_name: str = Field(min_length=3)
    schema_version: Literal["1.0"] = "1.0"
    source_slr_citation: str = Field(min_length=5)
    questions: list[GoldenQuestion] = Field(min_length=20, max_length=30)

    @model_validator(mode="after")
    def validate_dataset(self) -> GoldenDataset:
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question_id values must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> GoldenDataset:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def validate_pdf_assets(self, base_dir: Path) -> None:
        root = base_dir.resolve()
        for question in self.questions:
            for publication in question.publications:
                pdf_path = (root / publication.pdf_path).resolve()
                try:
                    pdf_path.relative_to(root)
                except ValueError as error:
                    raise ValueError(
                        "Golden PDF paths must remain inside the dataset directory"
                    ) from error
                if not pdf_path.is_file():
                    raise ValueError(f"Golden PDF is missing: {publication.pdf_path}")
                digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
                if digest != publication.sha256:
                    raise ValueError(f"Golden PDF checksum mismatch: {publication.pdf_path}")


def create_annotation_template(question_count: int = 25) -> dict:
    """Create empty annotation slots; this intentionally cannot pass the gold-data gate."""
    if not 20 <= question_count <= 30:
        raise ValueError("Golden Dataset size must be between 20 and 30 questions")
    return {
        "dataset_name": "",
        "schema_version": "1.0",
        "source_slr_citation": "",
        "questions": [
            {
                "question_id": f"q{index:02}",
                "question": "",
                "source_slr_citation": "",
                "source_slr_url": "",
                "reference_answer": "",
                "publications": [],
                "relevant_chunk_ids": [],
                "screening_labels": [],
                "validated_extractions": [],
                "relevance_annotation": None,
            }
            for index in range(1, question_count + 1)
        ],
    }


def write_annotation_template(path: Path, question_count: int = 25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(create_annotation_template(question_count), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
