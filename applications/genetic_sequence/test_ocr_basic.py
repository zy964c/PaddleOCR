"""Quick test to verify OCR is working."""
import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec"
)

# Test on genetics.PNG
result = ocr.predict(input="/home/elarai/PaddleOCR/applications/genetics.PNG")

print(f"Result type: {type(result)}")
for i, res in enumerate(result):
    print(f"\nResult {i+1} type: {type(res)}")
    print(f"Result {i+1} keys: {res.keys() if hasattr(res, 'keys') else 'N/A'}")
    print(f"Result {i+1} dir: {[x for x in dir(res) if not x.startswith('_')][:10]}")
    
    # Try to get text
    if hasattr(res, 'rec_text'):
        print(f"  rec_text attribute: {res.rec_text[:5] if res.rec_text else 'Empty'}")
    if hasattr(res, 'get'):
        print(f"  get('rec_text'): {res.get('rec_text', 'N/A')[:5] if res.get('rec_text') else 'Empty'}")
    
    break  # Just check first result
