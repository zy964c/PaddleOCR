#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/elarai/PaddleOCR')

from applications.genetic_sequence.ppstructure_genetics import PPStructureV3WithGenetics

# Initialize with mobile models
pipeline = PPStructureV3WithGenetics(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec"
)

# Run
results = pipeline.predict("applications/genetics.PNG")

# Save visualization
for res in results:
    pipeline.visualize(res, "output/library_test_visualization.jpg")

# Save JSON manually with boxes
import json
for res in results:
    # Get the boxes we added
    boxes = res.get("boxes", [])
    
    # Create output structure
    output = {
        "input_path": res.get("input_path"),
        "boxes": boxes
    }
    
    with open("output/library_test.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

# Show results
for result in results:
    print(f"Result keys: {result.keys()}")
    print(f"Has ocr_res: {'ocr_res' in result}")
    print(f"Overall OCR items: {len(result.get('overall_ocr_res', []))}")
    
    if 'ocr_res' in result:
        ocr_res = result['ocr_res']
        print(f"OCR res type: {type(ocr_res)}")
        if isinstance(ocr_res, dict):
            print(f"OCR res keys: {ocr_res.keys()}")
    
    boxes = result.get("boxes", [])
    
    # Count by type
    counts = {}
    for box in boxes:
        label = box.get("label")
        counts[label] = counts.get(label, 0) + 1
    
    print(f"Total elements: {len(boxes)}")
    print("\nBy type:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    
    # Show genetic sequences
    genetic = [b for b in boxes if b.get("label") == "genetic_sequence"]
    print(f"\nGenetic sequences ({len(genetic)}):")
    for g in genetic[:5]:
        print(f"  {g.get('text')[:50]}")
    if len(genetic) > 5:
        print(f"  ... and {len(genetic)-5} more")

print("\nVisualization saved to output/library_test_visualization.jpg")
print("JSON saved to output/library_test.json")
