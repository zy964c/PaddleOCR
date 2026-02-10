#!/usr/bin/env python3
"""PaddleOCR-VL pipeline with genetic sequence detection."""

from paddleocr import PaddleOCRVL
from .detector import GeneticSequenceDetector
import numpy as np


class PaddleOCRVLWithGenetics:
    """PaddleOCR-VL extended with genetic sequence detection via VLM prompt."""
    
    def __init__(self, use_vl_detection=True, **kwargs):
        self.vl_pipeline = PaddleOCRVL(**kwargs)
        self.genetic_detector = GeneticSequenceDetector(min_length=10)
        self.use_vl_detection = use_vl_detection
    
    def predict(self, input_path: str):
        results = self.vl_pipeline.predict(input_path)
        
        for result in results:
            layout_blocks = result.get("layout_blocks", [])
            merged_sequences = self._merge_sequence_blocks(layout_blocks)
            
            boxes = result.get("boxes", [])
            for seq in merged_sequences:
                boxes.append(seq)
            result["boxes"] = boxes
        
        return results
    
    def _merge_sequence_blocks(self, blocks):
        """Merge consecutive text blocks that form genetic sequences."""
        genetic_sequences = []
        i = 0
        
        while i < len(blocks):
            text = blocks[i].get("text", "")
            
            if self.genetic_detector.is_genetic_sequence(text):
                merged_text = text
                coords = [blocks[i].get("bbox", [])]
                seq_type = self.genetic_detector.get_sequence_type(text)
                
                j = i + 1
                while j < len(blocks):
                    next_text = blocks[j].get("text", "")
                    combined = merged_text + next_text
                    
                    if self.genetic_detector.is_genetic_sequence(combined):
                        merged_text = combined
                        coords.append(blocks[j].get("bbox", []))
                        j += 1
                    else:
                        break
                
                if coords:
                    coord = self._merge_bboxes(coords)
                    if isinstance(coord, np.ndarray):
                        coord = coord.tolist()
                    
                    genetic_sequences.append({
                        "label": "genetic_sequence",
                        "coordinate": coord,
                        "text": merged_text,
                        "sequence_type": seq_type
                    })
                
                i = j
            else:
                i += 1
        
        return genetic_sequences
    
    def _merge_bboxes(self, bboxes):
        """Merge multiple bounding boxes into one."""
        all_coords = []
        for bbox in bboxes:
            if bbox:
                if isinstance(bbox[0], (list, tuple)):
                    all_coords.extend(bbox)
                else:
                    x1, y1, x2, y2 = bbox
                    all_coords.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        if not all_coords:
            return []
        
        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
