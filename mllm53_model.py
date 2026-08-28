from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

SPECIAL_TOKENS: Final = ("<bos>", "<eos>", "<unk>")
TOKEN_RE: Final = re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]")


class CorpusError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    context_size: int = 4
    embedding_size: int = 32
    hidden_size: int = 96


def tokenize_corpus(text: str) -> list[list[str]]:
    paragraphs = re.split(r"\n\s*\n", text)
    sequences = [TOKEN_RE.findall(paragraph.lower()) + ["<eos>"] for paragraph in paragraphs if paragraph.strip()]
    if not sequences:
        raise CorpusError("corpus contains no non-empty paragraphs")
    return sequences


def tokenize_prompt(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens: list[str]) -> str:
    text = " ".join(token for token in tokens if token not in SPECIAL_TOKENS)
    for mark in ".!?":
        text = text.replace(f" {mark}", mark)
    return text


def make_vocabulary(sequences: list[list[str]]) -> list[str]:
    observed = {token for sequence in sequences for token in sequence}
    return [*SPECIAL_TOKENS, *sorted(observed.difference(SPECIAL_TOKENS))]


def make_examples(
    sequences: list[list[str]], vocabulary: list[str], context_size: int
) -> tuple[IntArray, IntArray]:
    if context_size < 1:
        raise ValueError("context_size must be positive")
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    bos_id = token_to_id["<bos>"]
    unknown_id = token_to_id["<unk>"]
    context_rows: list[list[int]] = []
    targets: list[int] = []
    for sequence in sequences:
        ids = [token_to_id.get(token, unknown_id) for token in sequence]
        for index, target in enumerate(ids):
            history = ids[max(0, index - context_size) : index]
            context_rows.append([bos_id] * (context_size - len(history)) + history)
            targets.append(target)
    return np.asarray(context_rows, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=-1, keepdims=True)


class TinyAutocompleteModel:
    def __init__(self, vocab_size: int, config: ModelConfig, seed: int = 7) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if config.context_size < 1 or config.embedding_size < 1 or config.hidden_size < 1:
            raise ValueError("model dimensions must be positive")
        rng = np.random.default_rng(seed)
        input_size = config.context_size * config.embedding_size
        self.config = config
        self.embedding = self._normal(rng, (vocab_size, config.embedding_size))
        self.hidden_weights = self._normal(rng, (input_size, config.hidden_size))
        self.hidden_bias = np.zeros(config.hidden_size, dtype=np.float64)
        self.output_weights = self._normal(rng, (config.hidden_size, vocab_size))
        self.output_bias = np.zeros(vocab_size, dtype=np.float64)
        self._step = 0
        self._moments = {
            name: (np.zeros_like(value), np.zeros_like(value))
            for name, value in self._parameters().items()
        }

    @staticmethod
    def _normal(rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
        return rng.normal(0.0, 0.08, size=shape).astype(np.float64)

    def _parameters(self) -> dict[str, FloatArray]:
        return {
            "embedding": self.embedding,
            "hidden_weights": self.hidden_weights,
            "hidden_bias": self.hidden_bias,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
        }

    def _forward(self, contexts: IntArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        embedded = self.embedding[contexts]
        flat = embedded.reshape(contexts.shape[0], -1)
        hidden = np.tanh(flat @ self.hidden_weights + self.hidden_bias)
        logits = hidden @ self.output_weights + self.output_bias
        return logits, flat, hidden

    def logits(self, contexts: IntArray) -> FloatArray:
        return self._forward(contexts)[0]

    def loss(self, contexts: IntArray, targets: IntArray) -> float:
        probabilities = _softmax(self.logits(contexts))
        return float(-np.mean(np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12))))

    def train_batch(self, contexts: IntArray, targets: IntArray, learning_rate: float) -> float:
        logits, flat, hidden = self._forward(contexts)
        probabilities = _softmax(logits)
        loss = float(-np.mean(np.log(np.maximum(probabilities[np.arange(len(targets)), targets], 1e-12))))
        gradient_logits = probabilities
        gradient_logits[np.arange(len(targets)), targets] -= 1.0
        gradient_logits /= len(targets)
        gradients = {
            "output_weights": hidden.T @ gradient_logits,
            "output_bias": np.sum(gradient_logits, axis=0),
        }
        gradient_hidden = (gradient_logits @ self.output_weights.T) * (1.0 - hidden * hidden)
        gradients["hidden_weights"] = flat.T @ gradient_hidden
        gradients["hidden_bias"] = np.sum(gradient_hidden, axis=0)
        gradient_flat = gradient_hidden @ self.hidden_weights.T
        gradients["embedding"] = np.zeros_like(self.embedding)
        np.add.at(gradients["embedding"], contexts, gradient_flat.reshape(self.embedding[contexts].shape))
        self._step += 1
        for name, parameter in self._parameters().items():
            first, second = self._moments[name]
            gradient = gradients[name]
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient * gradient
            first_hat = first / (1.0 - 0.9**self._step)
            second_hat = second / (1.0 - 0.999**self._step)
            parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
        return loss


def save_checkpoint(path: Path, model: TinyAutocompleteModel, vocabulary: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        config=np.asarray(json.dumps(asdict(model.config)), dtype=np.str_),
        vocabulary=np.asarray(vocabulary, dtype=np.str_),
        embedding=model.embedding,
        hidden_weights=model.hidden_weights,
        hidden_bias=model.hidden_bias,
        output_weights=model.output_weights,
        output_bias=model.output_bias,
    )


def load_checkpoint(path: Path) -> tuple[TinyAutocompleteModel, list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        config_data = json.loads(str(archive["config"].item()))
        config = ModelConfig(**config_data)
        vocabulary = [str(token) for token in archive["vocabulary"].tolist()]
        model = TinyAutocompleteModel(len(vocabulary), config)
        model.embedding = archive["embedding"].copy()
        model.hidden_weights = archive["hidden_weights"].copy()
        model.hidden_bias = archive["hidden_bias"].copy()
        model.output_weights = archive["output_weights"].copy()
        model.output_bias = archive["output_bias"].copy()
        return model, vocabulary
