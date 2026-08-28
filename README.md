# MLLM-5.3

MLLM-5.3 is a from-scratch, trainable family of compact recurrent language models for local autocomplete. The repository is the preview release surface: model files are named only by parameter size, and more sizes are planned.

> Preview: these checkpoints are small autocomplete models, not general-purpose assistants. They learn patterns from the supplied MLLM-5.2 corpus and can be inspected, retrained, and run locally.

## Models

| Artifact | Parameters | Architecture | Best for |
|---|---:|---|---|
| `578K.npz` | 578,864 | 183-wide single-layer GRU, tied embedding/output | Fast inline suggestions |
| `2.4M.npz` | 2,429,256 | 380-wide single-layer GRU, tied embedding/output | Balanced local autocomplete |
| `6.0M.npz` | 6,038,532 | 530-wide single-layer GRU, tied embedding/output | Highest-capacity preview |

Every model uses a byte-fallback tokenizer with frequent text chunks, so unseen words remain representable. The checkpoint also carries a compact prefix memory learned from the training split; exact known prefixes use that high-confidence continuation, while other contexts use the trained GRU. The size labels count neural parameters only. Training uses PyTorch; inference and checkpoint loading use NumPy. The browser demo runs the same GRU equations inside a Web Worker with float16 web exports.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Run from the repository root:

```bash
uv run train_mllm53.py generate \
  578K.npz \
  "What is an atom" \
  --max-tokens 18 \
  --temperature 0 \
  --seed 7
```

Inspect a checkpoint:

```bash
uv run train_mllm53.py inspect 578K.npz
```

Open the editor locally after exporting browser artifacts:

```bash
python3 -m http.server 4173 --directory site
open http://127.0.0.1:4173/
```

## Reproduce the training data

The extractor reads the embedded `BUILT_IN_CORPUS` literals from each MLLM-5.2 model. It handles the nested Golden corpus explicitly and writes clean per-source files plus a deduplicated combined corpus and manifest.

```bash
uv run extract_mllm52_corpora.py /path/to/MLLM-5.2 \
  --output-dir work/mllm52_corpora
```

The released preview was trained from `work/mllm52_corpora/combined.txt`; raw source corpora are not copied into the repository. See `work/mllm52_corpora/manifest.json` for hashes and counts.

## Train models

Train one tier:

```bash
uv run train_mllm53.py train 578K work/mllm52_corpora/combined.txt \
  --output 578K.npz \
  --steps 10000 \
  --sequence-length 128 \
  --device cpu
```

Train all preview tiers:

```bash
uv run train_mllm53.py train-all work/mllm52_corpora/combined.txt \
  --output-dir . \
  --steps 10000 \
  --sequence-length 128 \
  --device cpu
```

The run uses a fixed seed, a 90/10 ordered holdout, AdamW, warmup plus cosine decay, gradient clipping, and saves the best validation checkpoint. Run `uv run train_mllm53.py --help` for all options.

## Browser export

Export a checkpoint to the float16 JSON format used by the static editor:

```bash
uv run export_mllm53_web.py 578K.npz site/models/578K.json
```

The editor is in `site/`. `site/showcase.html` is the primitive state harness used before the product screen.

Deploy the editor as a new Netlify site:

```bash
npx netlify deploy --prod --dir=site
```

The included `netlify.toml` publishes `site/` and keeps the exported model artifacts cacheable.

## Files

- `mllm53_model.py` - tokenizer, architecture metadata, NumPy inference, and NPZ serialization
- `mllm53_training.py` - PyTorch GRU, data split, optimizer, scheduler, and validation loop
- `train_mllm53.py` - train, train-all, generate, and inspect commands
- `extract_mllm52_corpora.py` - safe AST-based MLLM-5.2 corpus extraction and provenance manifest
- `export_mllm53_web.py` - float16 browser artifact exporter
- `578K.npz`, `2.4M.npz`, `6.0M.npz` - trained preview checkpoints
- `model_manifest.json` - corpus provenance, training settings, hashes, and validation metrics
- `site/` - offline-capable browser editor and Web Worker runtime
- `DESIGN.md` - editor design system, research log, and accessibility contract

## License and status

Released under the repository license. MLLM-5.3 is a preview release: additional sizes, longer-context variants, quantization, and stronger evaluation are planned.
