# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "torch>=2.3", "typer>=0.16"]
# ///
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer

from mllm53_model import (
    MODEL_CONFIGS,
    Checkpoint,
    GenerationOptions,
    NumpyAutocompleteModel,
    PrefixMemory,
    config_from_name,
)
from mllm53_training import TrainingSettings, train_checkpoint

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _settings(
    model_name: str,
    steps: int,
    batch_size: int | None,
    sequence_length: int,
    seed: int,
    device: str,
) -> TrainingSettings:
    default_batch = 64 if model_name == "578K" else 32 if model_name == "2.4M" else 16
    return TrainingSettings(
        steps=steps,
        batch_size=batch_size or default_batch,
        sequence_length=sequence_length,
        evaluation_interval=max(50, steps // 8),
        seed=seed,
        device=device,
    )


@app.command("train")
def train(
    model_name: str = typer.Argument(..., help="Size tier: 578K, 2.4M, or 6.0M."),
    corpus_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(Path("578K.npz"), "--output", "-o"),
    steps: int = typer.Option(2_000, min=1),
    batch_size: int | None = typer.Option(None, min=1),
    sequence_length: int = typer.Option(96, min=8),
    seed: int = typer.Option(7),
    device: str = typer.Option("auto", help="auto, cpu, or mps."),
) -> None:
    """Train one MLLM-5.3 size tier from a cleaned corpus."""
    config = config_from_name(model_name)
    settings = _settings(model_name, steps, batch_size, sequence_length, seed, device)
    typer.echo(f"Training {model_name}: {config.parameter_count:,} parameters")
    checkpoint, metadata = train_checkpoint(corpus_path, config, settings)
    checkpoint.save(output_path)
    typer.echo(
        json.dumps(
            {
                "output": str(output_path),
                "best_validation_loss": metadata["best_validation_loss"],
            },
            indent=2,
        )
    )


@app.command("train-all")
def train_all(
    corpus_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(Path("."), "--output-dir", "-o"),
    steps: int = typer.Option(2_000, min=1),
    sequence_length: int = typer.Option(96, min=8),
    seed: int = typer.Option(7),
    device: str = typer.Option("auto"),
) -> None:
    """Train every release size tier with the same corpus split and seed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in MODEL_CONFIGS:
        config = config_from_name(model_name)
        settings = _settings(model_name, steps, None, sequence_length, seed, device)
        typer.echo(f"Training {model_name}: {config.parameter_count:,} parameters")
        checkpoint, metadata = train_checkpoint(corpus_path, config, settings)
        output_path = output_dir / f"{model_name}.npz"
        checkpoint.save(output_path)
        typer.echo(
            f"Saved {output_path} with validation loss {metadata['best_validation_loss']:.4f}"
        )


@app.command("attach-memory")
def attach_memory(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True
    ),
    corpus_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(Path("578K.npz"), "--output", "-o"),
    validation_fraction: float = typer.Option(0.1, min=0.0, max=0.9),
) -> None:
    """Attach a train-split prefix memory without changing neural weights."""
    checkpoint = Checkpoint.load(checkpoint_path)
    units = [
        unit.strip()
        for unit in corpus_path.read_text(encoding="utf-8").split("\n\n")
        if unit.strip()
    ]
    split = min(max(int(len(units) * (1.0 - validation_fraction)), 1), len(units) - 1)
    memory = PrefixMemory.from_text("\n\n".join(units[:split]))
    metadata = json.loads(checkpoint.metadata)
    metadata["architecture"] = (
        "single-layer GRU with tied embedding/output weights, byte fallback tokenizer, "
        "and train-split prefix memory"
    )
    replace(
        checkpoint,
        memory=memory.to_json(),
        metadata=json.dumps(metadata, separators=(",", ":")),
    ).save(output_path)
    typer.echo(f"Attached {len(memory.snippets):,} prefix snippets to {output_path}")


@app.command("generate")
def generate(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True
    ),
    prompt: str = typer.Argument(...),
    max_tokens: int = typer.Option(24, min=1),
    temperature: float = typer.Option(0.25, min=0.0),
    seed: int | None = typer.Option(7),
    top_k: int = typer.Option(32, min=1),
) -> None:
    """Generate an autocomplete continuation from a trained NPZ checkpoint."""
    checkpoint = Checkpoint.load(checkpoint_path)
    model = NumpyAutocompleteModel(checkpoint)
    completion = model.generate(
        prompt,
        GenerationOptions(
            max_tokens=max_tokens, temperature=temperature, seed=seed, top_k=top_k
        ),
    )
    typer.echo(completion)


@app.command("inspect")
def inspect(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True
    ),
) -> None:
    """Print the architecture and training metadata for a checkpoint."""
    checkpoint = Checkpoint.load(checkpoint_path)
    typer.echo(
        json.dumps(
            {
                "name": checkpoint.config.name,
                "parameters": checkpoint.config.parameter_count,
                "vocab_size": checkpoint.config.vocab_size,
                "embedding_size": checkpoint.config.embedding_size,
                "hidden_size": checkpoint.config.hidden_size,
                "tokenizer_vocab": checkpoint.tokenizer.vocab_size,
                "prefix_memory_snippets": len(PrefixMemory.from_json(checkpoint.memory).snippets),
                "metadata": json.loads(checkpoint.metadata),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
