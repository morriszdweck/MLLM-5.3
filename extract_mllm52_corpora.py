# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "typer>=0.16"]
# ///
"""Extract clean training text from the embedded MLLM-5.2 corpora.

The Golden 5.2 release contains a serialized copy of an earlier model source
around its own corpus.  This extractor parses literals with ``ast`` and, when
it finds that nested source, selects the inner corpus instead of training on
Python implementation text.

Usage:
    uv run extract_mllm52_corpora.py /private/tmp/mllm52-source
    uv run extract_mllm52_corpora.py /private/tmp/mllm52-source \
        --output-dir work/mllm52_corpora
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
CORPUS_NAME: Final[str] = "BUILT_IN_CORPUS"
NESTED_CORPUS_MARKER: Final[str] = 'BUILT_IN_CORPUS = r"""'


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    filename: str


@dataclass(frozen=True, slots=True)
class TextStats:
    chars: int
    utf8_bytes: int
    lines: int
    units: int
    unique_units: int
    sha256: str


SOURCE_SPECS: Final[tuple[SourceSpec, ...]] = (
    SourceSpec("muir", "MLLM-5.2-Muir-26P.py"),
    SourceSpec("monterey", "MLLM-5.2-Monterey-71P.py"),
    SourceSpec("tahoe", "MLLM-5.2-Tahoe-309P.py"),
    SourceSpec("whitney", "MLLM-5.2-Whitney-1042P.py"),
    SourceSpec("golden", "MLLM-5.2-Golden-3981P.py"),
)


class CorpusExtractionError(ValueError):
    """Raised when a source does not contain a readable corpus literal."""


def _corpus_literal(tree: ast.Module, source_name: str) -> str:
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == CORPUS_NAME
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        if not isinstance(value, str):
            raise CorpusExtractionError(f"{source_name} corpus is not a string literal")
        return value
    raise CorpusExtractionError(f"{source_name} has no {CORPUS_NAME} assignment")


def _read_corpus(path: Path, *, nested: bool = True) -> str:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        value = _corpus_literal(ast.parse(source), path.name)
        if nested and NESTED_CORPUS_MARKER in value:
            return _read_literal_source(value, f"{path.name}:nested")
        return value
    except (OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, CorpusExtractionError):
            raise
        raise CorpusExtractionError(f"cannot parse {path}: {exc}") from exc


def _read_literal_source(source: str, source_name: str) -> str:
    try:
        return _corpus_literal(ast.parse(source), source_name)
    except (SyntaxError, ValueError) as exc:
        if isinstance(exc, CorpusExtractionError):
            raise
        raise CorpusExtractionError(
            f"cannot parse nested {source_name}: {exc}"
        ) from exc


def _normalise_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _units(text: str) -> list[str]:
    candidates: list[str] = []
    for block in re.split(r"\n[ \t]*\n+", _normalise_text(text)):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) > 1:
            candidates.extend(lines)
        elif lines:
            candidates.append(lines[0])
    return [item for item in candidates if len(item) >= 3]


def _deduplicate(units: list[str], seen: set[str]) -> list[str]:
    unique: list[str] = []
    for item in units:
        key = re.sub(r"\s+", " ", item).strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _stats(text: str, units: list[str]) -> TextStats:
    encoded_lengths = np.asarray(
        [len(item.encode("utf-8")) for item in units], dtype=np.int64
    )
    return TextStats(
        chars=len(text),
        utf8_bytes=int(encoded_lengths.sum()),
        lines=text.count("\n") + (1 if text else 0),
        units=len(units),
        unique_units=len(
            {re.sub(r"\s+", " ", item).strip().casefold() for item in units}
        ),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _write_text(path: Path, units: list[str]) -> str:
    text = "\n\n".join(units).strip() + "\n"
    path.write_text(text, encoding="utf-8")
    return text


@app.command()
def main(
    source_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("work/mllm52_corpora"), "--output-dir"),
) -> None:
    """Extract per-model and combined corpus files plus a provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_units: list[str] = []
    combined_seen: set[str] = set()
    manifest_sources: dict[str, dict[str, object]] = {}

    for spec in SOURCE_SPECS:
        source_path = source_dir / spec.filename
        raw = _read_corpus(source_path)
        source_units = _deduplicate(_units(raw), set())
        source_text = _write_text(output_dir / f"{spec.name}.txt", source_units)
        combined_unique = _deduplicate(source_units, combined_seen)
        combined_units.extend(combined_unique)
        manifest_sources[spec.name] = {
            "source_file": spec.filename,
            "nested_literal_selected": spec.name == "golden",
            "stats": asdict(_stats(source_text, source_units)),
            "contributed_units": len(combined_unique),
        }

    combined_text = _write_text(output_dir / "combined.txt", combined_units)
    manifest = {
        "release": "MLLM-5.3",
        "source_repository": "https://github.com/morriszdweck/MLLM-5.2",
        "source_ref": "4089ca8",
        "selection": "Embedded BUILT_IN_CORPUS literals from Muir, Monterey, Tahoe, Whitney, and Golden; Golden uses its nested corpus literal.",
        "deduplication": "NFC normalization, blank-block/line units, casefolded whitespace-normalized first-seen deduplication.",
        "sources": manifest_sources,
        "combined": asdict(_stats(combined_text, combined_units)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    typer.echo(
        f"Wrote {len(manifest_sources)} source corpora and {len(combined_units):,} combined units to {output_dir}"
    )
    typer.echo(
        f"Combined text: {len(combined_text):,} chars, sha256={manifest['combined']['sha256']}"
    )


if __name__ == "__main__":
    app()
