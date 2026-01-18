#!/usr/bin/env python3
"""
tests/performance/oom_repro/generate_synthetic_pdf.py

Generate a small synthetic PDF for CI smoke tests.

Design goals:
- Deterministic output for reproducible CI runs.
- Small memory footprint.
- Exercises text + image parsing paths without stressing allocators.
- Intended for CI / dev only (not production runtime).
"""

from __future__ import annotations

import argparse
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageDraw


def make_small_image() -> ImageReader:
    """
    Create a small in‑memory PNG image.
    """
    img = Image.new("RGB", (200, 120), color=(200, 220, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Synthetic Image", fill=(10, 10, 80))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def generate_pdf(path: str, pages: int = 4) -> None:
    """
    Generate a multi‑page synthetic PDF.

    Args:
        path: Output PDF path.
        pages: Number of pages to generate.
    """
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    image = make_small_image()

    for p in range(1, pages + 1):
        c.setFont("Helvetica", 12)
        c.drawString(20 * mm, height - 30 * mm, f"Synthetic PDF - Page {p}")
        c.drawString(
            20 * mm,
            height - 40 * mm,
            "This is a small synthetic page used for CI smoke tests.",
        )

        c.drawImage(
            image,
            20 * mm,
            height - 120 * mm,
            width=80 * mm,
            height=48 * mm,
        )

        # Repeated text to exercise chunking logic
        for i in range(10):
            c.drawString(
                20 * mm,
                height - (60 + 5 * i) * mm,
                f"Line {i + 1} on page {p}: sample text for chunking.",
            )

        c.showPage()
    c.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic PDF for CI smoke tests.")
    parser.add_argument("--out", required=True, help="Output PDF path")
    parser.add_argument("--pages", type=int, default=4, help="Number of pages (default: 4)")
    args = parser.parse_args()
    generate_pdf(args.out, pages=args.pages)
    print(f"Generated synthetic PDF: {args.out}")