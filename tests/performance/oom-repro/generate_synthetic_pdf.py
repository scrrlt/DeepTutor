# tests/performance/oom_repro/generate_synthetic_pdf.py
"""
Generate a small synthetic PDF for CI smoke tests.
Creates a multi-page PDF with text and a small image to exercise parser code paths.
Requires reportlab (add to requirements.txt for CI).
"""

import argparse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
import io
from PIL import Image, ImageDraw, ImageFont


def make_small_image():
    # Create a small RGB image in memory
    img = Image.new("RGB", (200, 120), color=(200, 220, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 40), "Synthetic Image", fill=(10, 10, 80))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return ImageReader(bio)


def generate_pdf(path: str, pages: int = 4):
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    img = make_small_image()
    for p in range(1, pages + 1):
        c.setFont("Helvetica", 12)
        c.drawString(20 * mm, h - 30 * mm, f"Synthetic PDF - Page {p}")
        c.drawString(20 * mm, h - 40 * mm, "This is a small synthetic page used for CI smoke tests.")
        c.drawImage(img, 20 * mm, h - 120 * mm, width=80 * mm, height=48 * mm)
        # Add some repeated text to create chunking
        for i in range(10):
            c.drawString(20 * mm, h - (60 + 5 * i) * mm, f"Line {i+1} on page {p}: sample text for chunking.")
        c.showPage()
    c.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a small synthetic PDF for CI.")
    parser.add_argument("--out", required=True, help="Output PDF path")
    parser.add_argument("--pages", type=int, default=4, help="Number of pages (default 4)")
    args = parser.parse_args()
    generate_pdf(args.out, pages=args.pages)
    print(f"Generated synthetic PDF: {args.out}")