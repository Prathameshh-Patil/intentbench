"""Build the static browser demo: DistilBERT as int8 ONNX, run by transformers.js.

Writes `web/dist/` (git-ignored), which `scripts/publish_web.sh` pushes to the gh-pages branch:

    web/dist/index.html                         copied from web/index.html
    web/dist/meta.json                          threshold and test scores shown on the page
    web/dist/models/intentbench/config.json     labels (id2label) for transformers.js
    web/dist/models/intentbench/tokenizer*.json
    web/dist/models/intentbench/onnx/model_quantized.onnx

Before writing, it checks that the int8 model picks the same intent as PyTorch on the
validation set, so the page shows the model the README reports on.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intentbench.data import load_splits  # noqa: E402
from intentbench.serve.model import MAX_LENGTH, default_threshold  # noqa: E402

MODEL_DIR = ROOT / "models" / "distilbert"
OUT = ROOT / "web" / "dist"
MODEL_OUT = OUT / "models" / "intentbench"
MIN_AGREEMENT = 0.99  # int8 must pick the same top intent as float32 on 99% of rows


def export_fp32(model, tokenizer, path: Path) -> None:
    enc = tokenizer(["what's the weather like"], return_tensors="pt")
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence", max=512)
    torch.onnx.export(
        model,
        (),
        str(path),
        kwargs={"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: batch, 1: seq},
            "attention_mask": {0: batch, 1: seq},
        },
        dynamo=True,
        external_data=False,
    )
    # The dynamo exporter leaves intermediate shape annotations that onnx's shape inference
    # rejects during quantization; drop them and let the quantizer infer shapes again.
    graph = onnx.load(path)
    del graph.graph.value_info[:]
    onnx.save(graph, path)


def onnx_top1(session, tokenizer, texts: list[str]) -> np.ndarray:
    preds = []
    for start in range(0, len(texts), 64):
        enc = tokenizer(
            texts[start : start + 64],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="np",
        )
        feeds = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        preds.append(session.run(["logits"], feeds)[0].argmax(-1))
    return np.concatenate(preds)


@torch.no_grad()
def torch_top1(model, tokenizer, texts: list[str]) -> np.ndarray:
    preds = []
    for start in range(0, len(texts), 64):
        enc = tokenizer(
            texts[start : start + 64],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        preds.append(model(**enc).logits.argmax(-1).numpy())
    return np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=3096, help="validation rows for the check")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

    if OUT.exists():
        shutil.rmtree(OUT)
    (MODEL_OUT / "onnx").mkdir(parents=True)
    fp32 = OUT / "model_fp32.onnx"
    int8 = MODEL_OUT / "onnx" / "model_quantized.onnx"

    print("Exporting to ONNX ...")
    export_fp32(model, tokenizer, fp32)
    print("Quantizing weights to int8 ...")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, per_channel=True)
    fp32.unlink()

    validation = load_splits().validation.head(args.rows)
    texts = validation["text"].tolist()
    session = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    int8_pred = onnx_top1(session, tokenizer, texts)
    agreement = float((int8_pred == torch_top1(model, tokenizer, texts)).mean())
    int8_accuracy = float((int8_pred == validation["label"].to_numpy()).mean())
    print(f"int8 vs float32 top-1 agreement on {len(texts)} validation rows: {agreement:.4f}")
    print(f"int8 top-1 accuracy on the same rows (no threshold): {int8_accuracy:.4f}")
    if agreement < MIN_AGREEMENT:
        raise SystemExit(f"Agreement below {MIN_AGREEMENT}; not writing the site.")

    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(MODEL_DIR / name, MODEL_OUT / name)
    shutil.copy(ROOT / "web" / "index.html", OUT / "index.html")

    results = json.loads((ROOT / "reports" / "results.json").read_text())
    meta = {
        "threshold": default_threshold(),
        "max_length": MAX_LENGTH,
        "int8_agreement": agreement,
        "test": results["results"]["distilbert"]["test"],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    size_mb = int8.stat().st_size / 1e6
    print(f"Wrote {OUT.relative_to(ROOT)} (model {size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
