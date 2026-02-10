# PaddleOCR Genetic Sequence Detection - Installation Guide

## Overview

This package extends PaddleOCR's PP-StructureV3 pipeline with genetic sequence detection capabilities. It detects DNA, RNA, and protein sequences alongside standard layout elements (tables, text, figures, etc.).

## Installation

### Install the genetic sequence module

```bash
cd /home/elarai/PaddleOCR
python setup_genetics.py install
```

This installs the `paddleocr-genetic-sequence` package which uses your existing PaddleOCR installation.

## Usage

```python
from ppstructure_genetics import PPStructureV3WithGenetics

# Initialize pipeline
pipeline = PPStructureV3WithGenetics(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,  # Important: must be False for optimal detection
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec"
)

# Process document
results = pipeline.predict("document.png")

# Access results
for result in results:
    boxes = result.get("boxes", [])
    
    for box in boxes:
        label = box.get("label")
        text = box.get("text")
        coordinate = box.get("coordinate")
        score = box.get("score")
        
        if label == "genetic_sequence":
            seq_type = box.get("sequence_type")  # DNA/RNA/PROTEIN
            print(f"{seq_type}: {text}")
        elif label == "table":
            print(f"Table at {coordinate}")
        # ... handle other labels

# Save visualization
pipeline.visualize(result, "output.jpg")

# Save JSON
import json
with open("output.json", "w") as f:
    json.dump({"boxes": result.get("boxes", [])}, f, indent=2, default=str)
```

## Output Format

The pipeline returns PP-StructureV3 format with genetic sequences as additional elements:

```json
{
  "boxes": [
    {
      "cls_id": 999,
      "label": "genetic_sequence",
      "score": 0.99,
      "coordinate": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "text": "ATCGATCG...",
      "sequence_type": "DNA"
    },
    {
      "cls_id": 21,
      "label": "table",
      "score": 0.95,
      "coordinate": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

## Detected Elements

- **genetic_sequence** (cls_id: 999): DNA, RNA, or protein sequences
- **table** (cls_id: 21): Tables
- **text** (cls_id: 22): Text blocks
- **paragraph_title** (cls_id: 17): Titles
- **number** (cls_id: varies): Numbers
- And all other standard PP-StructureV3 elements

## Key Configuration

**Important:** Set `use_textline_orientation=False` to detect all genetic sequences. With this setting enabled, some sequences may be missed due to text orientation processing.

## Requirements

- paddleocr >= 3.0.0
- pillow
- Python >= 3.8
