"""Test genetic sequence detection on PDF with timing."""
import sys
import time
sys.path.insert(0, '/home/elarai/PaddleOCR')

from paddleocr import PPStructureV3

pdf_path = "/home/elarai/PaddleOCR/applications/202592892_A1_F4_Sum.pdf"

# Initialize
pipeline = PPStructureV3(
    device="gpu",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print(f"Processing: {pdf_path}\n")
start_time = time.time()

# Run pipeline
results = list(pipeline.predict(pdf_path))

# Find genetic sequences
def is_genetic_sequence(text, min_len=10):
    text = text.upper().replace(" ", "").replace("\n", "")
    if len(text) < min_len:
        return None
    if set(text).issubset({'A', 'T', 'C', 'G'}):
        return "DNA"
    if set(text).issubset({'A', 'U', 'C', 'G'}):
        return "RNA"
    if len(text) >= 5 and set(text).issubset(set('ACDEFGHIKLMNPQRSTVWY')):
        return "PROTEIN"
    return None

total_genetic = 0
total_elements = 0

for page_idx, res in enumerate(results):
    layout_det = res.get("layout_det_res", {})
    boxes = layout_det.get("boxes", [])
    
    ocr_res = res.get("overall_ocr_res", {})
    rec_texts = ocr_res.get("rec_texts", [])
    dt_polys = ocr_res.get("dt_polys", [])
    rec_scores = ocr_res.get("rec_scores", [])
    
    genetic_count = 0
    for i, text in enumerate(rec_texts):
        seq_type = is_genetic_sequence(text)
        if seq_type:
            poly = dt_polys[i] if i < len(dt_polys) else None
            if poly is not None:
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                boxes.append({
                    'cls_id': 999,
                    'label': 'genetic_sequence',
                    'score': float(rec_scores[i]) if i < len(rec_scores) else 0.0,
                    'coordinate': [min(xs), min(ys), max(xs), max(ys)]
                })
                genetic_count += 1
    
    total_genetic += genetic_count
    total_elements += len(boxes)
    
    print(f"Page {page_idx + 1}: {genetic_count} genetic sequences, {len(boxes)} total elements")

end_time = time.time()
elapsed = end_time - start_time

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Total pages: {len(results)}")
print(f"Total genetic sequences: {total_genetic}")
print(f"Total elements: {total_elements}")
print(f"Processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(f"Time per page: {elapsed/len(results):.2f} seconds")

# Save results to PDF with visualizations
print(f"\nSaving results to PDF...")
import fitz  # PyMuPDF
import io
from PIL import Image, ImageDraw

# Open original PDF
pdf_doc = fitz.open(pdf_path)
output_pdf = fitz.open()

for page_idx, res in enumerate(results):
    # Get original page
    page = pdf_doc[page_idx]
    
    # Get boxes
    layout_det = res.get("layout_det_res", {})
    boxes = layout_det.get("boxes", [])
    
    # Only draw on pages with genetic sequences
    has_genetic = any(b.get('label') == 'genetic_sequence' for b in boxes)
    
    if has_genetic:
        # Render page to image at higher resolution
        zoom = 2.0  # Render at 2x for better quality
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        
        # Draw genetic sequence boxes
        for box in boxes:
            if box.get('label') == 'genetic_sequence':
                coord = box.get('coordinate', [])
                if len(coord) == 4:
                    x1, y1, x2, y2 = coord
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=4)
        
        # Convert to PDF page
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', quality=85, optimize=True)
        img_bytes.seek(0)
        
        new_page = output_pdf.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(page.rect, stream=img_bytes.read())
    else:
        # Copy original page without modification
        output_pdf.insert_pdf(pdf_doc, from_page=page_idx, to_page=page_idx)

# Save with compression
output_path = "output/pdf_with_genetic_sequences.pdf"
output_pdf.save(output_path, garbage=4, deflate=True, clean=True)
output_pdf.close()
pdf_doc.close()

print(f"Saved annotated PDF: {output_path}")
print(f"{'='*80}")
