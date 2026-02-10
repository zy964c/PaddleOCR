#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from pipeline import GeneticSequencePipeline
import fitz
from PIL import Image, ImageDraw
import io

# Initialize pipeline
pipeline = GeneticSequencePipeline()

# Open PDF and get last page
pdf_path = "applications/202592892_A1_F4_Sum.pdf"
pdf_doc = fitz.open(pdf_path)
page_idx = len(pdf_doc) - 1  # Last page
page = pdf_doc[page_idx]

# Render to image
zoom = 2.0
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

# Convert pixmap to PIL Image directly
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# Save for reference
img.save("output/page_100_direct.png")

# Run OCR on the image directly (not from file)
import numpy as np
img_array = np.array(img)
result = pipeline.predict(img_array)
layout_det = result[0]
boxes = layout_det.get("boxes", [])

# Count genetic sequences
genetic_seqs = [b for b in boxes if b.get('label') == 'genetic_sequence']
print(f"\nPage {page_idx + 1}: Found {len(genetic_seqs)} genetic sequences")

# Draw boxes
draw = ImageDraw.Draw(img)

print(f"Image size: {img.size}")
print(f"Zoom factor: {zoom}")

for box in genetic_seqs:
    coord = box.get('coordinate', [])
    if len(coord) == 4:
        # Don't scale - coordinates are already in image space
        x1, y1, x2, y2 = coord
        draw.rectangle([x1, y1, x2, y2], outline='red', width=4)
        print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] - {box.get('text', '')[:30]}")

img.save("output/last_page_annotated.png")
print(f"\nSaved: output/last_page_annotated.png")

pdf_doc.close()
