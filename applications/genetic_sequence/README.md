# Genetic Sequence Detection

Hybrid OCR + rule-based validation for detecting DNA, RNA, and protein sequences in documents.

## Features

- **DNA Detection**: Sequences containing A, T, C, G
- **RNA Detection**: Sequences containing A, U, C, G  
- **Protein Detection**: Sequences with 20 standard amino acids
- **Configurable minimum length** for sequence validation
- **Returns bounding boxes** and confidence scores

## Usage

```python
from applications.genetic_sequence import GeneticSequenceDetector

# Initialize
detector = GeneticSequenceDetector(min_length=10)

# Extract sequences
sequences = detector.extract_sequences("document.png")

# Process results
for seq in sequences:
    print(f"{seq['sequence_type']}: {seq['text']}")
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

## Requirements

- PaddleOCR installed
- Minimum sequence length: 10 (configurable)
