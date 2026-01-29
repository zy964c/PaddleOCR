"""Extend PP-StructureV3 result with genetic sequences and use save_to_img()."""
import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from paddleocr import PPStructureV3

# Initialize PP-StructureV3 with MOBILE models (same as standalone pipeline)
pipeline = PPStructureV3(
    device="gpu",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

image_path = "/home/elarai/PaddleOCR/applications/genetics.PNG"
print("Running PP-StructureV3...")
results = list(pipeline.predict(image_path))

for res in results:
    # Get layout boxes
    layout_det = res.get("layout_det_res", {})
    boxes = layout_det.get("boxes", [])
    
    # Get RAW OCR results (before layout grouping)
    ocr_res = res.get("overall_ocr_res", {})
    rec_texts = ocr_res.get("rec_texts", [])
    dt_polys = ocr_res.get("dt_polys", [])
    rec_scores = ocr_res.get("rec_scores", [])
    
    print(f"DEBUG: Total OCR texts detected: {len(rec_texts)}")
    
    # Find genetic sequences from RAW OCR (not grouped by layout)
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
    
    # Add genetic sequences to boxes from RAW OCR
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
    
    print(f"Genetic sequences found: {genetic_count}")
    print(f"\nOriginal layout elements: {len(boxes) - genetic_count}")
    print(f"Added genetic sequences: {genetic_count}")
    print(f"Total elements: {len(boxes)}\n")
    
    # Count by type
    label_counts = {}
    for box in boxes:
        label = box.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("All element types:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Save using standard PP-StructureV3 methods
    print("\nSaving outputs...")
    res.save_to_json("output/ppstructure_with_genseq.json")
    res.save_to_markdown("output/ppstructure_with_genseq.md")
    
    # Save layout detection visualization (this draws boxes!)
    layout_det.save_to_img("output/ppstructure_layout_boxes.jpg")
    
    print("\nOutput files:")
    print("  JSON: output/ppstructure_with_genseq.json")
    print("  Markdown: output/ppstructure_with_genseq.md")
    print("  Layout boxes: output/ppstructure_layout_boxes.jpg")
