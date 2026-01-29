"""
Genetic Sequence Detector
Hybrid OCR + rule-based validation for detecting DNA, RNA, and protein sequences.
"""

from paddleocr import PaddleOCR
from typing import List, Dict, Optional


class GeneticSequenceDetector:
    """Detect and extract genetic sequences from documents."""
    
    def __init__(self, min_length: int = 10):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        self.min_length = min_length
    
    def extract_sequences(self, image_path: str) -> List[Dict]:
        """Extract genetic sequences from image."""
        result = self.ocr.predict(input=image_path)
        sequences = []
        
        for res in result:
            text = res.get("rec_text", "")
            bbox = res.get("dt_polys")
            confidence = res.get("rec_score", 0.0)
            
            cleaned = self._clean_text(text)
            seq_type = self._validate_sequence(cleaned)
            
            if seq_type:
                sequences.append({
                    "type": "genetic_sequence",
                    "sequence_type": seq_type,
                    "text": cleaned,
                    "bbox": bbox,
                    "confidence": confidence
                })
        
        return sequences
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        return text.upper().replace(" ", "").replace("\n", "")
    
    def _validate_sequence(self, text: str) -> Optional[str]:
        """Validate if text is a genetic sequence and return type."""
        if len(text) < self.min_length:
            return None
        
        # DNA (A, T, C, G)
        if set(text).issubset({'A', 'T', 'C', 'G'}):
            return "DNA"
        
        # RNA (A, U, C, G)
        if set(text).issubset({'A', 'U', 'C', 'G'}):
            return "RNA"
        
        # Protein (20 amino acids)
        if len(text) >= 5 and set(text).issubset(set('ACDEFGHIKLMNPQRSTVWY')):
            return "PROTEIN"
        
        return None
