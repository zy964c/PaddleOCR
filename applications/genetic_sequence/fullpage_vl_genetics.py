#!/usr/bin/env python3
"""Full-page VL with genetic sequence detection."""

from paddleocr import PaddleOCRVL
from .detector import GeneticSequenceDetector
from typing import List, Dict, Any
import numpy as np


class PaddleOCRVLWithGenetics:
    """Full-page PaddleOCR-VL with genetic sequence merging."""
    
    def __init__(self, **kwargs):
        self.vl_pipeline = PaddleOCRVL(**kwargs)
        self.genetic_detector = GeneticSequenceDetector(min_length=6)
    
    def predict(self, input_path: str) -> List[Dict[str, Any]]:
        """Run full-page VL OCR then detect genetic sequences."""
        results = self.vl_pipeline.predict(input_path)
        
        for result in results:
            # Extract text lines from markdown
            markdown = result.get("markdown", "")
            lines = [line.strip() for line in markdown.split('\n') if line.strip()]
            
            # Merge genetic sequences
            genetic_sequences = self._merge_sequence_lines(lines)
            
            # Add to boxes
            boxes = result.get("boxes", [])
            for seq in genetic_sequences:
                boxes.append(seq)
            result["boxes"] = boxes
        
        return results
    
    def _merge_sequence_lines(self, lines):
        """Merge consecutive lines that form genetic sequences."""
        genetic_sequences = []
        used = set()
        
        for i in range(len(lines)):
            if i in used:
                continue
                
            text = lines[i]
            if not self.genetic_detector.is_genetic_sequence(text):
                continue
            
            merged_text = text
            indices = [i]
            
            j = i + 1
            while j < len(lines):
                combined = merged_text + lines[j]
                if self.genetic_detector.is_genetic_sequence(combined):
                    merged_text = combined
                    indices.append(j)
                    j += 1
                else:
                    break
            
            for idx in indices:
                used.add(idx)
            
            genetic_sequences.append({
                "cls_id": 999,
                "label": "genetic_sequence",
                "score": 0.99,
                "coordinate": [],  # No coordinates from markdown
                "text": merged_text,
                "sequence_type": self.genetic_detector.get_sequence_type(merged_text)
            })
        
        return genetic_sequences
