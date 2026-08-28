# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "typer>=0.16"]
# ///
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import typer

from mllm53_model import Checkpoint, PrefixMemory

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _encode_array(array: np.ndarray) -> str:
    little_endian = np.asarray(array, dtype="<f2")
    return base64.b64encode(little_endian.tobytes()).decode("ascii")


def web_payload(checkpoint: Checkpoint) -> dict[str, object]:
    config = checkpoint.config
    return {
        "version": 1,
        "name": config.name,
        "parameter_count": config.parameter_count,
        "vocab_size": config.vocab_size,
        "embedding_size": config.embedding_size,
        "hidden_size": config.hidden_size,
        "dtype": "float16-le",
        "memory": list(PrefixMemory.from_json(checkpoint.memory).snippets),
        "extra_tokens": [base64.b64encode(token).decode("ascii") for token in checkpoint.tokenizer.extra_tokens],
        "weights": {
            "embedding": _encode_array(checkpoint.embedding),
            "weight_ih": _encode_array(checkpoint.weight_ih),
            "weight_hh": _encode_array(checkpoint.weight_hh),
            "bias_ih": _encode_array(checkpoint.bias_ih),
            "bias_hh": _encode_array(checkpoint.bias_hh),
            "output_bias": _encode_array(checkpoint.output_bias),
        },
    }


@app.command("export")
def export(checkpoint_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True), output_path: Path = typer.Argument(...)) -> None:
    """Export one NPZ checkpoint to a browser-loadable float16 JSON artifact."""
    checkpoint = Checkpoint.load(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(web_payload(checkpoint), separators=(",", ":")), encoding="utf-8")
    typer.echo(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    app()
