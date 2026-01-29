"""Compare genetic sequences found by simple OCR vs PP-StructureV3."""
import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from paddleocr import PaddleOCR, PPStructureV3

image_path = "/home/elarai/PaddleOCR/applications/genetics.PNG"

def find_genetic_sequences(rec_texts):
    """Find genetic sequences in text list."""
    sequences = []
    for text in rec_texts:
        cleaned = text.upper().replace(" ", "").replace("\n", "")
        if len(cleaned) >= 10 and set(cleaned).issubset({'A', 'T', 'C', 'G'}):
            sequences.append(cleaned)
    return sequences

# Test 1: Simple OCR
print("="*80)
print("TEST 1: Simple OCR (mobile models)")
print("="*80)
ocr = PaddleOCR(
    device="gpu",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
result1 = list(ocr.predict(image_path))
texts1 = result1[0].get("rec_texts", [])
seqs1 = find_genetic_sequences(texts1)
print(f"Total texts detected: {len(texts1)}")
print(f"Genetic sequences found: {len(seqs1)}\n")

# Test 2: PP-StructureV3
print("="*80)
print("TEST 2: PP-StructureV3 (server models)")
print("="*80)
structure = PPStructureV3(device="gpu")
result2 = list(structure.predict(image_path))
ocr_res = result2[0].get("overall_ocr_res", {})
texts2 = ocr_res.get("rec_texts", [])
seqs2 = find_genetic_sequences(texts2)
print(f"Total texts detected: {len(texts2)}")
print(f"Genetic sequences found: {len(seqs2)}\n")

# Compare
print("="*80)
print("COMPARISON")
print("="*80)
print(f"Difference: {len(seqs1) - len(seqs2)} sequences")

# Find missing sequences
set1 = set(seqs1)
set2 = set(seqs2)
missing = set1 - set2
if missing:
    print(f"\nMissing in PP-StructureV3 ({len(missing)}):")
    for seq in list(missing)[:3]:
        print(f"  {seq[:50]}...")
