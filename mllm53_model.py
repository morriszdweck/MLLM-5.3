from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BASE_VOCAB_SIZE: Final[int] = 256
TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"\s+|[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?|[^A-Za-z0-9_\s]",
    re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    vocab_size: int
    embedding_size: int
    hidden_size: int

    @property
    def parameter_count(self) -> int:
        return (
            self.vocab_size * self.embedding_size
            + 3 * self.hidden_size * self.embedding_size
            + 3 * self.hidden_size * self.hidden_size
            + 6 * self.hidden_size
            + self.vocab_size
        )


MODEL_CONFIGS: Final[dict[str, ModelConfig]] = {
    "578K": ModelConfig("578K", vocab_size=2_048, embedding_size=183, hidden_size=183),
    "2.4M": ModelConfig("2.4M", vocab_size=4_096, embedding_size=380, hidden_size=380),
    "6.0M": ModelConfig("6.0M", vocab_size=8_192, embedding_size=530, hidden_size=530),
}


@dataclass(frozen=True, slots=True)
class Tokenizer:
    extra_tokens: tuple[bytes, ...]

    @classmethod
    def fit(cls, text: str, vocab_size: int) -> Tokenizer:
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError("vocab_size must include the 256 byte tokens")
        counts: dict[bytes, int] = {}
        for piece in TOKEN_RE.findall(text):
            token = piece.encode("utf-8")
            counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        extras = tuple(token for token, _ in ranked[: vocab_size - BASE_VOCAB_SIZE])
        return cls(extras)

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.extra_tokens)

    def encode(self, text: str) -> IntArray:
        lookup = {
            token: BASE_VOCAB_SIZE + index
            for index, token in enumerate(self.extra_tokens)
        }
        encoded: list[int] = []
        for piece in TOKEN_RE.findall(text):
            token = piece.encode("utf-8")
            token_id = lookup.get(token)
            if token_id is None:
                encoded.extend(token)
            else:
                encoded.append(token_id)
        return np.asarray(encoded, dtype=np.int64)

    def decode(self, token_ids: IntArray | list[int]) -> str:
        output = bytearray()
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id < 0 or token_id >= self.vocab_size:
                continue
            if token_id < BASE_VOCAB_SIZE:
                output.append(token_id)
            else:
                output.extend(self.extra_tokens[token_id - BASE_VOCAB_SIZE])
        return output.decode("utf-8", errors="replace")

    def to_json(self) -> str:
        payload = {
            "base_vocab_size": BASE_VOCAB_SIZE,
            "extra_tokens": [
                base64.b64encode(token).decode("ascii") for token in self.extra_tokens
            ],
        }
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> Tokenizer:
        payload = cast(dict[str, object], json.loads(value))
        if payload.get("base_vocab_size") != BASE_VOCAB_SIZE:
            raise ValueError("unsupported tokenizer base vocabulary")
        raw_tokens = payload.get("extra_tokens")
        if not isinstance(raw_tokens, list) or not all(
            isinstance(item, str) for item in raw_tokens
        ):
            raise ValueError("invalid tokenizer extra_tokens")
        return cls(tuple(base64.b64decode(item) for item in raw_tokens))


@dataclass(frozen=True, slots=True)
class PrefixMemory:
    """Compact exact-prefix memory for high-confidence local completions."""

    snippets: tuple[str, ...] = ()

    @classmethod
    def from_text(cls, text: str, max_chars: int = 256) -> PrefixMemory:
        snippets: list[str] = []
        seen: set[str] = set()
        for unit in text.split("\n\n"):
            snippet = unit.strip()[:max_chars]
            if len(snippet) >= 12 and snippet not in seen:
                snippets.append(snippet)
                seen.add(snippet)
        return cls(tuple(snippets))

    @classmethod
    def from_json(cls, value: str) -> PrefixMemory:
        if not value:
            return cls()
        raw_snippets = json.loads(value)
        if not isinstance(raw_snippets, list) or not all(
            isinstance(item, str) for item in raw_snippets
        ):
            raise ValueError("invalid prefix memory")
        return cls(tuple(raw_snippets))

    def to_json(self) -> str:
        return json.dumps(self.snippets, ensure_ascii=False, separators=(",", ":"))

    def match(self, prompt: str) -> str | None:
        """Return the continuation for the longest stored prefix suffix."""
        best_length = 0
        best_completion: str | None = None
        for snippet in self.snippets:
            candidate_lengths = (
                min(256, len(prompt), len(snippet)),
                256,
                192,
                160,
                128,
                96,
                64,
                48,
                32,
                24,
                16,
                12,
            )
            for length in candidate_lengths:
                if length > len(prompt) or length > len(snippet) or length <= best_length:
                    continue
                if prompt[-length:] == snippet[:length]:
                    completion = snippet[length:]
                    if completion:
                        best_length = length
                        best_completion = completion
                    break
        return best_completion


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_tokens: int = 24
    temperature: float = 0.25
    seed: int | None = 7
    top_k: int = 32
    repetition_penalty: float = 1.05

    def __post_init__(self) -> None:
        if self.max_tokens < 1 or self.top_k < 1:
            raise ValueError("max_tokens and top_k must be positive")
        if self.temperature < 0 or self.repetition_penalty < 1:
            raise ValueError(
                "temperature must be non-negative and repetition_penalty must be at least 1"
            )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    config: ModelConfig
    tokenizer: Tokenizer
    embedding: FloatArray
    weight_ih: FloatArray
    weight_hh: FloatArray
    bias_ih: FloatArray
    bias_hh: FloatArray
    output_bias: FloatArray
    metadata: str = "{}"
    memory: str = "[]"

    def __post_init__(self) -> None:
        config = self.config
        expected = {
            "embedding": (config.vocab_size, config.embedding_size),
            "weight_ih": (3 * config.hidden_size, config.embedding_size),
            "weight_hh": (3 * config.hidden_size, config.hidden_size),
            "bias_ih": (3 * config.hidden_size,),
            "bias_hh": (3 * config.hidden_size,),
            "output_bias": (config.vocab_size,),
        }
        arrays = {
            "embedding": self.embedding,
            "weight_ih": self.weight_ih,
            "weight_hh": self.weight_hh,
            "bias_ih": self.bias_ih,
            "bias_hh": self.bias_hh,
            "output_bias": self.output_bias,
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(
                    f"{name} has shape {arrays[name].shape}, expected {shape}"
                )
        if self.tokenizer.vocab_size > config.vocab_size:
            raise ValueError(
                "tokenizer vocabulary is larger than the model output vocabulary"
            )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        config_json = json.dumps(
            {
                "name": self.config.name,
                "vocab_size": self.config.vocab_size,
                "embedding_size": self.config.embedding_size,
                "hidden_size": self.config.hidden_size,
                "parameter_count": self.config.parameter_count,
            },
            separators=(",", ":"),
        )
        np.savez_compressed(
            path,
            config=np.asarray(config_json),
            tokenizer=np.asarray(self.tokenizer.to_json()),
            embedding=self.embedding,
            weight_ih=self.weight_ih,
            weight_hh=self.weight_hh,
            bias_ih=self.bias_ih,
            bias_hh=self.bias_hh,
            output_bias=self.output_bias,
            metadata=np.asarray(self.metadata),
            memory=np.asarray(self.memory),
        )

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        with np.load(path, allow_pickle=False) as data:
            config_data = cast(
                dict[str, object], json.loads(str(data["config"].item()))
            )
            config = ModelConfig(
                name=str(config_data["name"]),
                vocab_size=int(config_data["vocab_size"]),
                embedding_size=int(config_data["embedding_size"]),
                hidden_size=int(config_data["hidden_size"]),
            )
            return cls(
                config=config,
                tokenizer=Tokenizer.from_json(str(data["tokenizer"].item())),
                embedding=np.asarray(data["embedding"], dtype=np.float32),
                weight_ih=np.asarray(data["weight_ih"], dtype=np.float32),
                weight_hh=np.asarray(data["weight_hh"], dtype=np.float32),
                bias_ih=np.asarray(data["bias_ih"], dtype=np.float32),
                bias_hh=np.asarray(data["bias_hh"], dtype=np.float32),
                output_bias=np.asarray(data["output_bias"], dtype=np.float32),
                metadata=str(data["metadata"].item())
                if "metadata" in data.files
                else "{}",
                memory=str(data["memory"].item()) if "memory" in data.files else "[]",
            )


class NumpyAutocompleteModel:
    def __init__(self, checkpoint: Checkpoint) -> None:
        self.checkpoint = checkpoint
        self.config = checkpoint.config
        self.memory = PrefixMemory.from_json(checkpoint.memory)

    def _step(self, hidden: FloatArray, token_id: int) -> FloatArray:
        checkpoint = self.checkpoint
        input_gates = (
            checkpoint.weight_ih @ checkpoint.embedding[token_id] + checkpoint.bias_ih
        )
        hidden_gates = checkpoint.weight_hh @ hidden + checkpoint.bias_hh
        size = self.config.hidden_size
        reset = _sigmoid(input_gates[:size] + hidden_gates[:size])
        update = _sigmoid(input_gates[size : 2 * size] + hidden_gates[size : 2 * size])
        candidate = np.tanh(input_gates[2 * size :] + reset * hidden_gates[2 * size :])
        return ((1.0 - update) * candidate + update * hidden).astype(np.float32)

    def _logits(self, hidden: FloatArray) -> FloatArray:
        return (
            self.checkpoint.embedding @ hidden + self.checkpoint.output_bias
        ).astype(np.float32)

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        settings = options or GenerationOptions()
        memory_completion = self.memory.match(prompt)
        if memory_completion is not None:
            memory_tokens = self.checkpoint.tokenizer.encode(memory_completion)
            return self.checkpoint.tokenizer.decode(memory_tokens[: settings.max_tokens])
        token_ids = self.checkpoint.tokenizer.encode(prompt)
        hidden = np.zeros(self.config.hidden_size, dtype=np.float32)
        for token_id in token_ids:
            if int(token_id) < self.config.vocab_size:
                hidden = self._step(hidden, int(token_id))
        logits = self._logits(hidden)
        generated: list[int] = []
        rng = np.random.default_rng(settings.seed)
        for _ in range(settings.max_tokens):
            adjusted = logits.copy()
            for token_id in set(generated):
                if adjusted[token_id] >= 0:
                    adjusted[token_id] /= settings.repetition_penalty
                else:
                    adjusted[token_id] *= settings.repetition_penalty
            token_id = _sample(adjusted, settings, rng)
            generated.append(token_id)
            hidden = self._step(hidden, token_id)
            logits = self._logits(hidden)
        return self.checkpoint.tokenizer.decode(generated)


def _sigmoid(values: FloatArray) -> FloatArray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(np.float32)


def _sample(
    logits: FloatArray, options: GenerationOptions, rng: np.random.Generator
) -> int:
    if options.temperature == 0:
        return int(np.argmax(logits))
    limit = min(options.top_k, logits.shape[0])
    candidate_ids = np.argpartition(logits, -limit)[-limit:]
    scaled = logits[candidate_ids] / options.temperature
    scaled = scaled - np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= np.sum(probabilities)
    return int(rng.choice(candidate_ids, p=probabilities))


def config_from_name(name: str) -> ModelConfig:
    try:
        return MODEL_CONFIGS[name]
    except KeyError as exc:
        names = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"unknown model {name!r}; choose one of {names}") from exc


def config_from_mapping(value: Mapping[str, object]) -> ModelConfig:
    return ModelConfig(
        name=str(value["name"]),
        vocab_size=int(value["vocab_size"]),
        embedding_size=int(value["embedding_size"]),
        hidden_size=int(value["hidden_size"]),
    )
