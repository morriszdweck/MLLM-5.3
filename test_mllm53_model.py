from __future__ import annotations

from pathlib import Path

import numpy as np

from mllm53_model import (
    ModelConfig,
    TinyAutocompleteModel,
    load_checkpoint,
    make_examples,
    save_checkpoint,
    tokenize_corpus,
)


def test_tokenize_corpus_keeps_paragraph_boundaries() -> None:
    sequences = tokenize_corpus("Hi.\n\nBye!")

    assert sequences == [["hi", ".", "<eos>"], ["bye", "!", "<eos>"]]


def test_make_examples_pads_short_contexts() -> None:
    vocab = ["<bos>", "<eos>", "<unk>", "a", "b"]

    contexts, targets = make_examples([["a", "b", "<eos>"]], vocab, context_size=4)

    assert contexts.shape == (3, 4)
    assert targets.tolist() == [3, 4, 1]
    assert contexts[0].tolist() == [0, 0, 0, 0]


def test_model_learns_a_repeated_character_sequence() -> None:
    config = ModelConfig(context_size=3, embedding_size=8, hidden_size=16)
    model = TinyAutocompleteModel(vocab_size=4, config=config, seed=7)
    contexts = np.asarray([[0, 0, 2], [0, 2, 3]], dtype=np.int64)
    targets = np.asarray([3, 2], dtype=np.int64)
    first_loss = model.train_batch(contexts, targets, learning_rate=0.05)

    for _ in range(60):
        model.train_batch(contexts, targets, learning_rate=0.05)

    assert model.loss(contexts, targets) < first_loss


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    config = ModelConfig(context_size=3, embedding_size=8, hidden_size=16)
    model = TinyAutocompleteModel(vocab_size=4, config=config, seed=7)
    contexts = np.asarray([[0, 0, 2], [0, 2, 3]], dtype=np.int64)
    checkpoint = tmp_path / "model.npz"

    save_checkpoint(checkpoint, model, ["<bos>", "<eos>", "a", "b"])
    restored, vocabulary = load_checkpoint(checkpoint)

    assert vocabulary == ["<bos>", "<eos>", "a", "b"]
    np.testing.assert_allclose(model.logits(contexts), restored.logits(contexts))
