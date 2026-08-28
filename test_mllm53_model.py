from __future__ import annotations

from dataclasses import replace

import numpy as np

from mllm53_model import (
    MODEL_CONFIGS,
    Checkpoint,
    GenerationOptions,
    NumpyAutocompleteModel,
    PrefixMemory,
    Tokenizer,
)


def _checkpoint() -> Checkpoint:
    config = MODEL_CONFIGS["578K"]
    tokenizer = Tokenizer.fit("hello world hello model", vocab_size=config.vocab_size)
    embedding = np.zeros((config.vocab_size, config.embedding_size), dtype=np.float32)
    weight_ih = np.zeros(
        (3 * config.hidden_size, config.embedding_size), dtype=np.float32
    )
    weight_hh = np.zeros((3 * config.hidden_size, config.hidden_size), dtype=np.float32)
    bias_ih = np.zeros(3 * config.hidden_size, dtype=np.float32)
    bias_hh = np.zeros(3 * config.hidden_size, dtype=np.float32)
    output_bias = np.zeros(config.vocab_size, dtype=np.float32)
    return Checkpoint(
        config=config,
        tokenizer=tokenizer,
        embedding=embedding,
        weight_ih=weight_ih,
        weight_hh=weight_hh,
        bias_ih=bias_ih,
        bias_hh=bias_hh,
        output_bias=output_bias,
    )


def test_size_tier_parameter_counts() -> None:
    assert MODEL_CONFIGS["578K"].parameter_count == 578_864
    assert MODEL_CONFIGS["2.4M"].parameter_count == 2_429_256
    assert MODEL_CONFIGS["6.0M"].parameter_count == 6_038_532


def test_tokenizer_round_trips_unicode_with_byte_fallback() -> None:
    tokenizer = Tokenizer.fit("Hello, world!", vocab_size=300)
    text = "Hello, π world!"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_checkpoint_round_trip_and_deterministic_generation(tmp_path) -> None:
    checkpoint = _checkpoint()
    path = tmp_path / "578K.npz"
    checkpoint.save(path)
    loaded = Checkpoint.load(path)
    first = NumpyAutocompleteModel(loaded).generate(
        "hello", GenerationOptions(max_tokens=5, temperature=0.0, seed=7)
    )
    second = NumpyAutocompleteModel(loaded).generate(
        "hello", GenerationOptions(max_tokens=5, temperature=0.0, seed=7)
    )
    assert first == second


def test_prefix_memory_uses_longest_known_suffix() -> None:
    memory = PrefixMemory.from_text(
        "What is an atom- Atoms are the basic particles of matter.\n\n"
        "What is a molecule- A molecule is a group of atoms."
    )
    assert memory.match("What is an atom- ") == "Atoms are the basic particles of matter."
    checkpoint = replace(_checkpoint(), memory=memory.to_json())
    completion = NumpyAutocompleteModel(checkpoint).generate(
        "What is an atom- ", GenerationOptions(max_tokens=3, temperature=0.0)
    )
    assert completion.startswith("Ato")
