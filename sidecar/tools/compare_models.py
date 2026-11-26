"""Compare model A (DINOv2 ViT-S/14) with the model B candidates on the same images.

    python tools/compare_models.py                       # both test images, all four models, FeatUp
    python tools/compare_models.py --images a.png b.png --models dinov2_vits14 clip_vitb16 --runs 10
    python tools/compare_models.py --upsampler nearest   # without FeatUp (laptop): patch blocks

Writes to --out (default comparison-out/): <image>_grid.png per image, metrics.md and
metrics.json. With FeatUp (default, plan p.7) every model uses its own FeatUp JBU
checkpoint and each pixel takes the nearest patch-level group centroid, as model A
does in the server. Numbers from a laptop are for development; the lab PC numbers
are the ones that count.
"""

import argparse
import datetime
import gc
import glob
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIDECAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIDECAR))

from contract import IMAGE_SIZE, K  # noqa: E402
from processing.grouping import (  # noqa: E402
    cluster_patches,
    group_attention,
    normalize_attention,
    order_centroids,
    pixel_labels,
    relabel_by_height,
    upsample_nearest,
)

MODEL_A = "dinov2_vits14"
ALL_MODELS = ["dinov2_vits14", "clip_vitb16", "dino_vits16", "vit_s16_sup"]
PALETTE = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
     [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]], dtype=np.float64
)
CELL, LABEL_H, HEADER_H = IMAGE_SIZE, 28, 58
COLUMNS = [  # (title, explanation) per grid column
    ("Input", "the 224 x 224 capture"),
    ("Groups", "8 K-means groups, one colour each"),
    ("Attention", "per patch: dark = low, yellow = high"),
    ("Groups x attention", "group colour, brighter = more attention"),
]


def agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index of two label maps: 1 = same grouping, ~0 = unrelated."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(labels_a.ravel(), labels_b.ravel()))


def _spread(values: list[float]) -> dict:
    return {"median": round(statistics.median(values), 2), "min": round(min(values), 2), "max": round(max(values), 2)}


def _memory_mb(device: str) -> float | None:
    import torch

    if device.startswith("cuda"):
        return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    if device == "mps":
        allocated = getattr(torch.mps, "driver_allocated_memory", None)
        return round(allocated() / 2**20, 1) if allocated else None
    return None


def analyse(encoder, image: np.ndarray, runs: int) -> dict:
    """Encode and group one image `runs` times (after one warm-up)."""
    import torch

    if encoder.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    warm = encoder.encode(image)  # warm-up: model, FeatUp kernels and scikit-learn's thread pool
    if warm.upsample is not None:
        warm.upsample()
    encoder.synchronize()
    cluster_patches(warm.features, K)

    encode_ms, cluster_ms, featup_ms = [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = encoder.encode(image)
        encoder.synchronize()
        t1 = time.perf_counter()
        raw, centroids = cluster_patches(out.features, K)
        patch_labels = relabel_by_height(raw.reshape(encoder.grid, encoder.grid), K)
        t2 = time.perf_counter()
        encode_ms.append((t1 - t0) * 1000)
        cluster_ms.append((t2 - t1) * 1000)
        if out.upsample is not None:
            pixels = out.upsample()  # ends with the copy to the CPU, so the GPU work is done
            featup_ms.append((time.perf_counter() - t2) * 1000)

    if out.upsample is not None:
        labels = pixel_labels(pixels, patch_labels, order_centroids(raw, patch_labels, centroids), (IMAGE_SIZE, IMAGE_SIZE))
        featup = {"ms": _spread(featup_ms), "execution": encoder.execution_path}
    else:
        labels, featup = upsample_nearest(patch_labels, IMAGE_SIZE), None
    # Paper and plan p.6: group attention stays the group mean over each group's patches.
    attention = group_attention(patch_labels, out.patch_attention, K)
    return {
        "model_id": encoder.model_id,
        "grid": encoder.grid,
        "encode_ms": _spread(encode_ms),
        "cluster_ms": _spread(cluster_ms),
        "memory_mb": _memory_mb(encoder.device),
        "attention": {
            "row_sum": round(out.attention_row_sum, 5),
            "finite": bool(np.isfinite(out.patch_attention).all()),
            "peakedness": round(float(out.patch_attention.max() / out.patch_attention.mean()), 2),
            "top3_share": round(float(np.sort(out.patch_attention.ravel())[-3:].sum() / out.patch_attention.sum()), 3),
        },
        "attention_norm": normalize_attention(attention),
        "labels": labels,
        "patch_attention": out.patch_attention,
        "upsampler": "nearest" if featup is None else "featup_jbu",
        "featup": featup,
    }


def _heat(values: np.ndarray) -> np.ndarray:
    """0..1 -> dark blue / orange / yellow heat colors."""
    stops = np.array([[10, 10, 40], [200, 60, 30], [255, 230, 80]], dtype=np.float64)
    v = np.clip(values, 0, 1)[..., None]
    low = stops[0] + (stops[1] - stops[0]) * np.clip(v * 2, 0, 1)
    return np.where(v < 0.5, low, stops[1] + (stops[2] - stops[1]) * np.clip(v * 2 - 1, 0, 1))


def render_grid(image: np.ndarray, rows: list[dict]) -> Image.Image:
    """Legend header, then one row per model: input | groups | attention heatmap | groups x attention."""
    grid = Image.new("RGB", (4 * CELL, HEADER_H + len(rows) * (CELL + LABEL_H)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=11)
    for j, (title, explanation) in enumerate(COLUMNS):
        draw.text((j * CELL + 6, 6), title, fill="black", font=font)
        draw.text((j * CELL + 6, 25), explanation, fill=(70, 70, 70), font=small)
    bar = _heat(np.tile(np.linspace(0, 1, CELL - 12), (8, 1)))  # colour scale for the attention column
    grid.paste(Image.fromarray(bar.astype(np.uint8)), (2 * CELL + 6, 42))
    draw.line([(0, HEADER_H - 1), (4 * CELL, HEADER_H - 1)], fill=(180, 180, 180))
    img = image.astype(np.float64)
    for i, row in enumerate(rows):
        top = HEADER_H + i * (CELL + LABEL_H)
        labels = row["labels"]
        attn = np.asarray(row["patch_attention"], dtype=np.float64)
        attn_px = np.asarray(Image.fromarray((attn / attn.max() * 255).astype(np.uint8)).resize((CELL, CELL), Image.NEAREST)) / 255
        brightness = 0.35 + 0.65 * np.asarray(row["attention_norm"])[labels]
        cells = [
            img,
            0.45 * img + 0.55 * PALETTE[labels],
            _heat(attn_px),
            0.45 * img + 0.55 * PALETTE[labels] * brightness[..., None],
        ]
        for j, cell in enumerate(cells):
            grid.paste(Image.fromarray(np.clip(cell, 0, 255).astype(np.uint8)), (j * CELL, top + LABEL_H))
        ari = row.get("ari_vs_a")
        name = f"{row['model_id']} (model A)" if row["model_id"] == MODEL_A else row["model_id"]
        featup = row.get("featup")
        timing = f"encode {row['encode_ms']['median']:.0f} ms" + (f" + FeatUp {featup['ms']['median']:.0f} ms" if featup else "")
        groups = "FeatUp JBU" if featup else "nearest"
        text = f"{name}   ARI vs A: {'-' if ari is None else f'{ari:.2f}'}   {timing}   groups: {groups}"
        draw.text((6, top + 6), text, fill="black", font=font)
    return grid


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SIDECAR, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_images(paths: list[str]) -> dict[str, np.ndarray]:
    images = {}
    for path in paths:
        image = Image.open(path)
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise SystemExit(f"{path}: image must be {IMAGE_SIZE}x{IMAGE_SIZE}, got {image.width}x{image.height}")
        images[Path(path).stem] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return images


def _fmt(spread: dict) -> str:
    return f"{spread['median']:.1f} ({spread['min']:.1f}–{spread['max']:.1f})"


def write_markdown(path: Path, env: dict, results: dict) -> None:
    lines = [
        "# Model B comparison",
        "",
        f"{env['date']} · {env['platform']} · device `{env['device']}` · torch {env['torch']} · commit `{env['commit']}` · {env['runs']} runs per model",
        "",
        "Groups: FeatUp JBU pixel labels (each model with its own FeatUp checkpoint)."
        if env["upsampler"] == "featup"
        else "Groups: nearest (patch blocks, FeatUp off).",
        "",
    ]
    if not env["device"].startswith("cuda"):
        lines += ["Development numbers (not the lab PC).", ""]
    for stem, rows in results.items():
        lines += [
            f"## {stem}",
            "",
            "| Model | Grid | Encode ms, median (min–max) | K-means ms | FeatUp ms | Memory MB | Attention row sum | Peakedness | Top-3 share | ARI vs A |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            if "error" in r:
                lines.append(f"| {r['model_id']} | failed: {r['error']} | | | | | | | | |")
                continue
            ari = "-" if r["ari_vs_a"] is None else f"{r['ari_vs_a']:.2f}"
            memory = "n/a" if r["memory_mb"] is None else f"{r['memory_mb']:.0f}"
            featup = "off" if r["featup"] is None else f"{_fmt(r['featup']['ms'])}, {r['featup']['execution']}"
            lines.append(
                f"| {r['model_id']} | {r['grid']}×{r['grid']} | {_fmt(r['encode_ms'])} | {_fmt(r['cluster_ms'])} | {featup} | {memory} | "
                f"{r['attention']['row_sum']:.3f} | {r['attention']['peakedness']:.1f} | {r['attention']['top3_share']:.0%} | {ari} |"
            )
        lines += ["", f"![{stem}]({stem}_grid.png)", ""]
    lines += [
        "Grid columns: input | groups (8 K-means groups, one colour each) | attention heatmap (dark = low, yellow = high) |",
        "groups x attention (group colour, brighter = more attention; roughly what the headset paints). One row per model, model A first.",
        "With FeatUp each pixel takes the nearest patch-level group centroid, so groups follow object edges; without it they are",
        "patch blocks (16x16 grid for DINOv2, 14x14 for the others). Attention stays per patch (paper), so the heatmap is blocky either way.",
        "FeatUp ms = JBU upsampling plus the copy of the 224x224 features to the CPU. Memory = peak GPU memory, FeatUp included.",
        "",
        "Peakedness = max / mean of the CLS-to-patch attention (1 = flat, useless for brightness).",
        "Top-3 share = fraction of all patch attention taken by the 3 strongest patches. A high value with high peakedness",
        "means a few 'sink' tokens soak up the attention (known for CLIP and supervised ViTs), not focus on objects:",
        "check the heatmap column before reading peakedness as good.",
        "ARI vs A = adjusted Rand index between the candidate's and DINOv2's 8-group maps (1 = same grouping, ~0 = unrelated).",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", default=sorted(glob.glob(str(SIDECAR / "fixtures" / "images" / "*.png"))))
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--upsampler", choices=["featup", "nearest"], default="featup")
    parser.add_argument("--out", default=str(SIDECAR / "comparison-out"))
    args = parser.parse_args(argv)

    import torch

    from models.base import pick_device
    from models.candidates import load_encoder

    if args.upsampler == "featup":  # refuse up front, as the server does: no silent fallback
        from models.featup_jbu import FeatUpUnavailable, check_available

        try:
            check_available()
        except FeatUpUnavailable as exc:
            parser.error(str(exc))
    images = _load_images(args.images)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict]] = {stem: {} for stem in images}
    for model_id in args.models:  # one model in memory at a time
        try:
            encoder = load_encoder(model_id, device, args.upsampler)
            for stem, image in images.items():
                results[stem][model_id] = analyse(encoder, image, args.runs)
            print(f"{model_id}: done", flush=True)
        except Exception as exc:  # report and carry on with the other models
            for stem in images:
                results[stem][model_id] = {"model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{model_id}: failed ({exc})", flush=True)
        encoder = None
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()

    ordered: dict[str, list[dict]] = {}
    for stem, by_model in results.items():
        a = by_model.get(MODEL_A)
        rows = [by_model[m] for m in args.models]
        for row in rows:
            if "error" not in row:
                ok_a = a is not None and "error" not in a
                row["ari_vs_a"] = round(agreement(a["labels"], row["labels"]), 3) if ok_a else None
        ordered[stem] = rows
        drawable = [r for r in rows if "error" not in r]
        if drawable:
            render_grid(images[stem], drawable).save(out_dir / f"{stem}_grid.png")

    env = {
        "date": datetime.datetime.now().isoformat(timespec="minutes"),
        "platform": platform.platform(),
        "device": device,
        "torch": torch.__version__,
        "commit": _git_commit(),
        "runs": args.runs,
        "upsampler": args.upsampler,
    }
    write_markdown(out_dir / "metrics.md", env, ordered)
    skip = {"labels", "patch_attention", "attention_norm"}
    serializable = {stem: [{k: v for k, v in r.items() if k not in skip} for r in rows] for stem, rows in ordered.items()}
    (out_dir / "metrics.json").write_text(json.dumps({"environment": env, "images": serializable}, indent=2) + "\n")
    print(f"wrote {out_dir}/metrics.md, metrics.json and grids", flush=True)


if __name__ == "__main__":
    main()
