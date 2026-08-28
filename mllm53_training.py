from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mllm53_model import Checkpoint, ModelConfig, PrefixMemory, Tokenizer


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    steps: int = 2_000
    batch_size: int = 64
    sequence_length: int = 96
    learning_rate: float = 0.002
    weight_decay: float = 0.01
    warmup_steps: int = 100
    evaluation_interval: int = 200
    evaluation_batches: int = 6
    validation_fraction: float = 0.1
    unit_start_fraction: float = 0.5
    seed: int = 7
    device: str = "auto"


class CompactGRULanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.embedding_size != config.hidden_size:
            raise ValueError(
                "tied output projection requires equal embedding and hidden sizes"
            )
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.recurrent = nn.GRU(
            config.embedding_size, config.hidden_size, batch_first=True
        )
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.recurrent.weight_ih_l0)
        nn.init.orthogonal_(self.recurrent.weight_hh_l0)
        nn.init.zeros_(self.recurrent.bias_ih_l0)
        nn.init.zeros_(self.recurrent.bias_hh_l0)

    def forward(
        self, input_ids: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(input_ids)
        sequence, next_hidden = self.recurrent(embedded, hidden)
        logits = F.linear(sequence, self.embedding.weight, self.output_bias)
        return logits, next_hidden


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def corpus_units(text: str) -> list[str]:
    return [unit.strip() for unit in text.split("\n\n") if unit.strip()]


def prepare_streams(
    corpus_path: Path, config: ModelConfig, fraction: float
) -> tuple[Tokenizer, np.ndarray, np.ndarray, np.ndarray, str, PrefixMemory]:
    text = corpus_path.read_text(encoding="utf-8")
    units = corpus_units(text)
    if len(units) < 20:
        raise ValueError("corpus needs at least 20 text units")
    split = min(max(int(len(units) * (1.0 - fraction)), 1), len(units) - 1)
    train_text = "\n\n".join(units[:split])
    validation_text = "\n\n".join(units[split:])
    tokenizer = Tokenizer.fit(train_text, config.vocab_size)
    train_tokens = tokenizer.encode(train_text)
    validation_tokens = tokenizer.encode(validation_text)
    if len(train_tokens) < 2 or len(validation_tokens) < 2:
        raise ValueError("corpus split is too small to train")
    separator_length = len(tokenizer.encode("\n\n"))
    unit_starts: list[int] = []
    offset = 0
    for unit in units[:split]:
        unit_starts.append(offset)
        offset += len(tokenizer.encode(unit)) + separator_length
    corpus_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (
        tokenizer,
        train_tokens,
        validation_tokens,
        np.asarray(unit_starts, dtype=np.int64),
        corpus_hash,
        PrefixMemory.from_text(train_text),
    )


def batch_from_stream(
    tokens: np.ndarray,
    batch_size: int,
    sequence_length: int,
    rng: np.random.Generator,
    device: torch.device,
    preferred_starts: np.ndarray | None = None,
    preferred_fraction: float = 0.0,
) -> tuple[Tensor, Tensor]:
    if len(tokens) <= sequence_length + 1:
        raise ValueError("token stream is shorter than sequence length")
    starts = rng.integers(0, len(tokens) - sequence_length - 1, size=batch_size)
    if preferred_starts is not None and preferred_fraction > 0:
        count = min(batch_size, max(1, round(batch_size * preferred_fraction)))
        starts[:count] = rng.choice(preferred_starts, size=count, replace=True)
        rng.shuffle(starts)
    inputs = np.stack(
        [tokens[start : start + sequence_length] for start in starts]
    ).astype(np.int64)
    targets = np.stack(
        [tokens[start + 1 : start + sequence_length + 1] for start in starts]
    ).astype(np.int64)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def validation_loss(
    model: CompactGRULanguageModel,
    tokens: np.ndarray,
    settings: TrainingSettings,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(settings.evaluation_batches):
            inputs, targets = batch_from_stream(
                tokens, settings.batch_size, settings.sequence_length, rng, device
            )
            logits, _ = model(inputs)
            losses.append(
                float(
                    F.cross_entropy(
                        logits.reshape(-1, model.config.vocab_size), targets.reshape(-1)
                    )
                )
            )
    model.train()
    return float(np.mean(np.asarray(losses, dtype=np.float64)))


def _learning_rate(settings: TrainingSettings, step: int) -> float:
    if step <= settings.warmup_steps:
        return settings.learning_rate * step / max(settings.warmup_steps, 1)
    progress = (step - settings.warmup_steps) / max(
        settings.steps - settings.warmup_steps, 1
    )
    return settings.learning_rate * (
        0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    )


def train_checkpoint(
    corpus_path: Path, config: ModelConfig, settings: TrainingSettings
) -> tuple[Checkpoint, dict[str, object]]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
    (
        tokenizer,
        train_tokens,
        validation_tokens,
        unit_starts,
        corpus_hash,
        prefix_memory,
    ) = prepare_streams(corpus_path, config, settings.validation_fraction)
    valid_unit_starts = unit_starts[
        unit_starts < len(train_tokens) - settings.sequence_length - 1
    ]
    model = CompactGRULanguageModel(config).to(device)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != config.parameter_count
    ):
        raise RuntimeError(
            "model parameter count does not match the release configuration"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    batch_rng = np.random.default_rng(settings.seed)
    validation_rng = np.random.default_rng(settings.seed + 1)
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for step in range(1, settings.steps + 1):
        rate = _learning_rate(settings, step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        inputs, targets = batch_from_stream(
            train_tokens,
            settings.batch_size,
            settings.sequence_length,
            batch_rng,
            device,
            preferred_starts=valid_unit_starts,
            preferred_fraction=settings.unit_start_fraction,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (
            step == 1
            or step % settings.evaluation_interval == 0
            or step == settings.steps
        ):
            validation = validation_loss(
                model, validation_tokens, settings, device, validation_rng
            )
            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": validation,
                "learning_rate": rate,
            }
            history.append(record)
            if validation < best_loss:
                best_loss = validation
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    state = model.state_dict()
    metadata = {
        "release": "MLLM-5.3",
        "architecture": "single-layer GRU with tied embedding/output weights, byte fallback tokenizer, and train-split prefix memory",
        "corpus_sha256": corpus_hash,
        "train_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "device": str(device),
        "settings": {
            "steps": settings.steps,
            "batch_size": settings.batch_size,
            "sequence_length": settings.sequence_length,
            "learning_rate": settings.learning_rate,
            "weight_decay": settings.weight_decay,
            "warmup_steps": settings.warmup_steps,
            "evaluation_interval": settings.evaluation_interval,
            "seed": settings.seed,
            "unit_start_fraction": settings.unit_start_fraction,
        },
        "best_validation_loss": best_loss,
        "history": history,
    }
    checkpoint = Checkpoint(
        config=config,
        tokenizer=tokenizer,
        embedding=state["embedding.weight"].numpy().astype(np.float32),
        weight_ih=state["recurrent.weight_ih_l0"].numpy().astype(np.float32),
        weight_hh=state["recurrent.weight_hh_l0"].numpy().astype(np.float32),
        bias_ih=state["recurrent.bias_ih_l0"].numpy().astype(np.float32),
        bias_hh=state["recurrent.bias_hh_l0"].numpy().astype(np.float32),
        output_bias=state["output_bias"].numpy().astype(np.float32),
        metadata=json.dumps(metadata, separators=(",", ":")),
        memory=prefix_memory.to_json(),
    )
    return checkpoint, metadata
