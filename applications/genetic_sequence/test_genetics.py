"""Test genetic sequence detector on sample image."""

import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from applications.genetic_sequence import GeneticSequenceDetector

# Initialize detector
detector = GeneticSequenceDetector(min_length=10)

# Test on the genetics.PNG image
image_path = "/home/elarai/PaddleOCR/applications/genetics.PNG"

print("Extracting genetic sequences from genetics.PNG...\n")
sequences = detector.extract_sequences(image_path)

# Save visualization of all OCR results
print("Saving OCR visualization...")
raw_result = detector.ocr.predict(input=image_path)
for res in raw_result:
    res.save_to_img(save_path="output/genetics_all_text.jpg")
    print(f"  Saved all text visualization to: output/genetics_all_text.jpg")

# Save visualization of only genetic sequences
detector.save_visualization(image_path, sequences, "output/genetics_sequences_only.jpg")

print(f"\nFound {len(sequences)} genetic sequences:\n")
print("=" * 80)

for i, seq in enumerate(sequences, 1):
    print(f"\nSequence {i}:")
    print(f"  Type: {seq['sequence_type']}")
    print(f"  Text: {seq['text'][:100]}{'...' if len(seq['text']) > 100 else ''}")
    print(f"  Length: {len(seq['text'])} bases/residues")
    print(f"  Confidence: {seq['confidence']:.3f}")
    print(f"  BBox: {seq['bbox']}")
    print("-" * 80)
