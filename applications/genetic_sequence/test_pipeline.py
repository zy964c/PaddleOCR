"""Test genetic sequence pipeline with JSON output."""

import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from applications.genetic_sequence.pipeline import GeneticSequencePipeline

# Initialize pipeline
pipeline = GeneticSequencePipeline(min_length=10, device="gpu")

# Run on genetics image
image_path = "/home/elarai/PaddleOCR/applications/genetics.PNG"
results = pipeline.predict(image_path)

# Print summary
print(f"\n{'='*80}")
print(f"GENETIC SEQUENCE PIPELINE - PP-StructureV3 Format")
print(f"{'='*80}\n")

for result in results:
    boxes = result.get('boxes', [])
    
    # Count by label
    label_counts = {}
    for box in boxes:
        label = box.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Input: {result.get('input_path')}")
    print(f"Total elements: {len(boxes)}\n")
    
    print("Element counts:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Show genetic sequences
    genetic_seqs = [b for b in boxes if b.get('label') == 'genetic_sequence']
    print(f"\nGenetic sequences found: {len(genetic_seqs)}")
    for i, seq in enumerate(genetic_seqs[:3], 1):
        print(f"  {i}. {seq['sequence_type']}: {seq['text'][:40]}...")
        print(f"     Coordinate: {seq['coordinate']}")
        print(f"     Score: {seq['score']:.3f}")

# Save JSON
json_path = "output/genetic_sequences_result.json"
pipeline.save_to_json(results, json_path)

# Save visualization
vis_path = "output/genetic_sequences_visualization.jpg"
pipeline.save_visualization(image_path, results, vis_path)

print(f"\n{'='*80}\n")
print("Output files:")
print(f"  JSON: {json_path}")
print(f"  Visualization: {vis_path}")
print(f"\n{'='*80}\n")
