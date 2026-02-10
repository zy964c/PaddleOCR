#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from applications.genetic_sequence.detector import GeneticSequenceDetector
from paddleocr import PaddleOCRVL
import time

# Initialize PaddleOCR-VL
print("Initializing PaddleOCR-VL...")
vl_pipeline = PaddleOCRVL()

# Initialize genetic detector
detector = GeneticSequenceDetector()

# Test on genetics.PNG
print("\nTesting on genetics.PNG...")
start = time.time()
result = vl_pipeline.predict("applications/genetics.PNG")
vl_time = time.time() - start

# Extract text from VL results
print(f"Result type: {type(result)}")
print(f"Result length: {len(result)}")

all_texts = []
for res in result:
    # Get parsing results which contain the text
    parsing_res = res.get("parsing_res_list", [])
    for item in parsing_res:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                all_texts.append(text)
    
    # Also check spotting_res
    spotting = res.get("spotting_res", {})
    if isinstance(spotting, dict):
        boxes = spotting.get("boxes", [])
        for box in boxes:
            text = box.get("text", "")
            if text:
                all_texts.append(text)

# Validate genetic sequences
genetic_seqs = [t for t in all_texts if detector.is_genetic_sequence(t)]

print(f"VL Time: {vl_time:.2f}s")
print(f"Total texts: {len(all_texts)}")
print(f"Genetic sequences: {len(genetic_seqs)}")
