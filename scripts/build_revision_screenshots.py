#!/usr/bin/env python3
"""Build response-letter screenshots from the latest revised manuscript PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "paper" / "revision_1" / "main.pdf"
DEFAULT_OUTPUT = ROOT / "paper" / "revision_1" / "revision_screenshots"

# (1-based page number, top, bottom). Horizontal bounds retain the manuscript
# text column while removing most of the surrounding page whitespace.
CROPS: dict[str, list[tuple[int, float, float]]] = {
    "abstract": [(1, 470, 650)],
    "main_and_thresholding": [
        (15, 100, 165),
        (15, 410, 720),
        (22, 190, 355),
    ],
    "calibration": [(22, 190, 700)],
    "main_multiseed": [(15, 410, 720)],
    "implementation_details": [(14, 500, 670), (16, 100, 475)],
    "data_availability": [(30, 105, 225)],
    "notation": [(7, 300, 620)],
    "claims_scope": [(20, 100, 520)],
    "interpretability": [(18, 430, 720), (19, 90, 600)],
    "threshold_free": [(22, 350, 700), (23, 80, 390)],
    "theory": [(13, 100, 710)],
    "negative_controls": [(25, 150, 690)],
    "scenario_prior": [(9, 160, 410)],
    "difficulty_results": [(17, 210, 720)],
    "proofreading": [(6, 345, 480), (31, 395, 470)],
    "ambiguity_analysis": [(4, 280, 720), (5, 430, 720)],
    "purpose": [(3, 520, 700), (4, 100, 230)],
    "recent_mllm_work": [(3, 500, 700), (15, 500, 720)],
    "discussion_scope": [(29, 100, 520)],
    "zero_shot_and_tsne": [(15, 500, 720), (17, 100, 205), (19, 90, 390)],
    "roc_pr": [(22, 350, 700), (23, 80, 390)],
    "discussion": [(28, 290, 720)],
    "efficiency": [(26, 440, 700), (27, 100, 500)],
    "framework_caption": [(6, 345, 480)],
    "prior_ablation": [(20, 520, 720), (21, 100, 500)],
    "k_sensitivity": [(21, 480, 700), (22, 100, 355)],
    "terminology": [(6, 345, 480), (15, 610, 700)],
    "bibliography": [(5, 430, 720), (31, 330, 720)],
    "format_and_page_limit": [(7, 300, 620), (30, 105, 225), (35, 640, 790)],
}


def render_crop(
    document: fitz.Document,
    page_number: int,
    top: float,
    bottom: float,
    zoom: float,
) -> Image.Image:
    page = document[page_number - 1]
    clip = fitz.Rect(70, top, page.rect.width - 70, min(bottom, page.rect.height))
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return ImageOps.expand(image, border=2, fill="#9a9a9a")


def stack_images(images: list[Image.Image], gap: int = 20) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zoom", type=float, default=2.4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.pdf)
    for name, specs in CROPS.items():
        images = [
            render_crop(document, page_number, top, bottom, args.zoom)
            for page_number, top, bottom in specs
        ]
        output = args.output_dir / f"{name}.png"
        stack_images(images).save(output, optimize=True)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
