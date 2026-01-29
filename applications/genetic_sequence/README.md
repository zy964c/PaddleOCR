# Genetic Sequence Detection Pipeline

Extends PaddleOCR with genetic sequence detection, returning PP-StructureV3-compatible JSON format.

## Features

- **DNA Detection**: Sequences containing A, T, C, G
- **RNA Detection**: Sequences containing A, U, C, G  
- **Protein Detection**: Sequences with 20 standard amino acids
- **PP-StructureV3 Format**: Compatible JSON output with standard layout classes
- **No Duplicates**: Genetic sequences are separate from text elements
- **Visualization**: Color-coded bounding boxes for all element types

## Output Format

Returns PP-StructureV3-compatible JSON with all elements:

```json
{
  "results": [{
    "input_path": "document.png",
    "page_index": 0,
    "boxes": [
      {
        "cls_id": 22,
        "label": "text",
        "score": 0.998,
        "coordinate": [100.0, 200.0, 300.0, 250.0],
        "text": "Regular text content"
      },
      {
        "cls_id": 999,
        "label": "genetic_sequence",
        "score": 0.999,
        "coordinate": [100.0, 300.0, 500.0, 350.0],
        "text": "ATCGATCGATCG",
        "sequence_type": "DNA"
      }
    ]
  }]
}
```

## Class IDs

Standard PP-StructureV3 classes:
- `22` - text
- `21` - table
- `14` - image
- `17` - paragraph_title
- `6` - doc_title
- `7` - figure_title

**New class:**
- `999` - **genetic_sequence** (with additional `sequence_type` field)

## Installation

```bash
uv pip install paddlepaddle-gpu==3.2.1 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install "paddleocr[doc-parser]"
```

## Usage

```python
from applications.genetic_sequence.pipeline import GeneticSequencePipeline

# Initialize
pipeline = GeneticSequencePipeline(min_length=10, device="gpu")

# Run detection
results = pipeline.predict("document.png")

# Save JSON (PP-StructureV3 format)
pipeline.save_to_json(results, "output.json")

# Save visualization
pipeline.save_visualization("document.png", results, "output.jpg")

# Access results
for result in results:
    for box in result['boxes']:
        if box['label'] == 'genetic_sequence':
            print(f"{box['sequence_type']}: {box['text']}")
```

## Test Results

Tested on `genetics.PNG`:
- **28 genetic sequences** (cls_id: 999)
- **27 text elements** (cls_id: 22)
- **Total: 55 elements**
- No duplicates between text and genetic_sequence classes
- All coordinates as floats: `[x1, y1, x2, y2]`
