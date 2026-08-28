#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0",
#     "typer>=0.16",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run train_mllm53.py [ARGS]
# 3. Or make executable and run:
#      chmod +x train_mllm53.py && ./train_mllm53.py
# ──────────────────

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from mllm53_model import (
    ModelConfig,
    TinyAutocompleteModel,
    detokenize,
    load_checkpoint,
    make_examples,
    make_vocabulary,
    save_checkpoint,
    tokenize_corpus,
    tokenize_prompt,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _batches(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(size)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


@app.command()
def train(
    corpus: Path,
    checkpoint: Path = Path("mllm53_tiny.npz"),
    epochs: int = 18,
    batch_size: int = 512,
    learning_rate: float = 0.003,
    stride: int = 1,
    seed: int = 7,
) -> None:
    text = corpus.read_text(encoding="utf-8")
    sequences = tokenize_corpus(text)
    split = max(1, int(len(sequences) * 0.9))
    train_sequences = sequences[:split]
    validation_sequences = sequences[split:] or sequences[-1:]
    vocabulary = make_vocabulary(train_sequences)
    config = ModelConfig()
    train_contexts, train_targets = make_examples(train_sequences, vocabulary, config.context_size)
    validation_contexts, validation_targets = make_examples(
        validation_sequences, vocabulary, config.context_size
    )
    train_contexts = train_contexts[::stride]
    train_targets = train_targets[::stride]
    model = TinyAutocompleteModel(len(vocabulary), config, seed=seed)
    rng = np.random.default_rng(seed)
    typer.echo(
        f"training {len(train_targets):,} examples, {len(vocabulary)} symbols, "
        f"{config.context_size}-token context"
    )
    for epoch in range(1, epochs + 1):
        losses = [
            model.train_batch(train_contexts[indexes], train_targets[indexes], learning_rate)
            for indexes in _batches(len(train_targets), batch_size, rng)
        ]
        validation_loss = model.loss(validation_contexts, validation_targets)
        typer.echo(
            f"epoch {epoch:02d}/{epochs} train_loss={np.mean(losses):.4f} "
            f"validation_loss={validation_loss:.4f}"
        )
    save_checkpoint(checkpoint, model, vocabulary)
    typer.echo(f"saved checkpoint: {checkpoint}")


def _completion(
    model: TinyAutocompleteModel,
    vocabulary: list[str],
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    unknown_id = token_to_id["<unk>"]
    ids = [token_to_id.get(token, unknown_id) for token in tokenize_prompt(prompt)]
    generated: list[str] = []
    rng = np.random.default_rng(seed)
    for _ in range(max_tokens):
        history = ids[-model.config.context_size :]
        context = [token_to_id["<bos>"]] * (model.config.context_size - len(history)) + history
        logits = model.logits(np.asarray([context], dtype=np.int64))[0]
        if temperature <= 0:
            next_id = int(np.argmax(logits))
        else:
            scaled = logits / max(temperature, 1e-6)
            scaled -= np.max(scaled)
            probabilities = np.exp(scaled)
            probabilities /= np.sum(probabilities)
            next_id = int(rng.choice(len(vocabulary), p=probabilities))
        token = vocabulary[next_id]
        if token == "<eos>":
            break
        if token != "<unk>":
            generated.append(token)
        ids.append(next_id)
    return detokenize(generated)


@app.command()
def generate(
    checkpoint: Path,
    prompt: str,
    max_tokens: int = 24,
    temperature: float = 0.2,
    seed: int = 7,
) -> None:
    model, vocabulary = load_checkpoint(checkpoint)
    completion = _completion(model, vocabulary, prompt, max_tokens, temperature, seed)
    separator = "" if not completion or prompt.endswith((" ", "\n")) else " "
    typer.echo(prompt + separator + completion)


if __name__ == "__main__":
    app()
