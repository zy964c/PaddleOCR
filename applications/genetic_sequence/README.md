# Genetic Sequence Detection

Hybrid OCR + rule-based validation for detecting DNA, RNA, and protein sequences in documents.

## Features

- **DNA Detection**: Sequences containing A, T, C, G
- **RNA Detection**: Sequences containing A, U, C, G  
- **Protein Detection**: Sequences with 20 standard amino acids
- **Configurable minimum length** for sequence validation
- **Returns bounding boxes** and confidence scores
- **Visualization support**: Save images with detected sequences highlighted

## Installation

Requires PaddlePaddle-GPU 3.2.1 with CUDA 12.6:

```bash
uv pip install paddlepaddle-gpu==3.2.1 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install paddleocr
```

## Usage

```python
from applications.genetic_sequence import GeneticSequenceDetector

# Initialize
detector = GeneticSequenceDetector(min_length=10)

# Extract sequences
sequences = detector.extract_sequences("document.png")

# Save visualization (genetic sequences only)
detector.save_visualization("document.png", sequences, "output.jpg")

# Process results
for seq in sequences:
    print(f"{seq['sequence_type']}: {seq['text']}")
    print(f"Confidence: {seq['confidence']:.2f}")
```

## Output Format

```python
{
    "type": "genetic_sequence",
    "sequence_type": "DNA" | "RNA" | "PROTEIN",
    "text": "ATCGATCG...",
    "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "confidence": 0.95
}
```

## Test Results

Tested on `genetics.PNG`:
- **28 DNA sequences detected**
- Average confidence: 0.99
- Visualizations saved to `output/` directory

## Visualization

Two types of visualizations are generated:

1. **All text** (`genetics_all_text.jpg`): Shows all OCR-detected text
2. **Sequences only** (`genetics_sequences_only.jpg`): Highlights only genetic sequences with green boxes

## Requirements

- PaddlePaddle-GPU 3.2.1
- PaddleOCR 3.4.0
- Python 3.12
- CUDA 12.6
- Minimum sequence length: 10 (configurable)
