# MLLM-5.3 Preview 578K

MLLM-5.3 Preview 578K is a small, trainable neural autocomplete model. It learns to predict the next word or punctuation token from a short left context and can run locally with no server.

> **Preview:** This is the first small proof-of-concept checkpoint. More model sizes are planned. The 578K model is not a final release and should not be treated as a general-purpose language model.

## Current model

- 577,533 trainable parameters
- 4-token context window
- 4,381-token vocabulary
- 32-dimensional token embeddings
- 96-unit `tanh` hidden layer
- Softmax next-token prediction
- Adam optimization implemented with NumPy
- Portable `.npz` checkpoint

The model was trained on a small, repetitive autocomplete corpus. It is best at reproducing the style and phrases represented in that corpus.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv run outputs/train_mllm53.py generate \
  outputs/mllm53_tiny.npz \
  "What is an atom" \
  --max-tokens 18 \
  --temperature 0 \
  --seed 7
```

Example output:

```text
What is an atom atoms are the basic particles of the chemical elements and the fundamental building blocks of matter.
```

## Train your own checkpoint

The training corpus is supplied as a command-line input so you can use your own text:

```bash
uv run outputs/train_mllm53.py train path/to/corpus.txt \
  --checkpoint outputs/my_model.npz \
  --epochs 18 \
  --batch-size 512 \
  --learning-rate 0.003 \
  --seed 7
```

The corpus is split by paragraph into training and validation sequences. The tokenizer recognizes words plus `.`, `!`, and `?`.

## Files

- `outputs/mllm53_model.py` - model, tokenizer, optimizer, and checkpoint serialization
- `outputs/train_mllm53.py` - `train` and `generate` commands
- `outputs/mllm53_tiny.npz` - trained MLLM-5.3 Preview 578K checkpoint
- `outputs/test_mllm53_model.py` - unit and checkpoint round-trip tests

## Roadmap

Planned future preview sizes include larger neural models, longer context windows, improved tokenization, better confidence scoring, quantized inference, and browser/editor integration. The architecture and checkpoint format may change before the stable 5.3 release.
