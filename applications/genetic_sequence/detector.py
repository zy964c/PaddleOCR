"""
Genetic Sequence Detector
Hybrid OCR + rule-based validation for detecting DNA, RNA, and protein sequences.
"""

from paddleocr import PaddleOCR
from typing import List, Dict, Optional


class GeneticSequenceDetector:
    """Detect and extract genetic sequences from documents."""
    
    def __init__(self, min_length: int = 5):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec"
        )
        self.min_length = min_length
    
    def extract_sequences(self, image_path: str) -> List[Dict]:
        """Extract genetic sequences from image."""
        result = self.ocr.predict(input=image_path)
        sequences = []
        
        for res in result:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            dt_polys = res.get("dt_polys", [])
            
            for i, text in enumerate(rec_texts):
                cleaned = self._clean_text(text)
                seq_type = self._validate_sequence(cleaned)
                
                if seq_type:
                    sequences.append({
                        "type": "genetic_sequence",
                        "sequence_type": seq_type,
                        "text": cleaned,
                        "bbox": dt_polys[i] if i < len(dt_polys) else None,
                        "confidence": rec_scores[i] if i < len(rec_scores) else 0.0
                    })
        
        return sequences
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        return text.upper().replace(" ", "").replace("\n", "")
    
    def is_genetic_sequence(self, text: str) -> bool:
        """Check if text is a genetic sequence."""
        cleaned = self._clean_text(text)
        return self._validate_sequence(cleaned) is not None
    
    def get_sequence_type(self, text: str) -> Optional[str]:
        """Get the type of genetic sequence."""
        cleaned = self._clean_text(text)
        return self._validate_sequence(cleaned)
    
    def _validate_sequence(self, text: str) -> Optional[str]:
        """Validate if text is a genetic sequence and return type."""
        # Strip stop codon marker and SEQ ID NO labels
        text = text.rstrip('*')
        import re
        text = re.sub(r'\(?\s*SEQ\s*ID\s*NO\.?\s*\d+\s*\)?', '', text, flags=re.IGNORECASE).strip()
        
        if len(text) < self.min_length:
            return None
        
        # Filter out Cyrillic - debug
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in text)
        if has_cyrillic:
            # print(f"[DEBUG] Rejected Cyrillic text: {text[:50]}")
            return None
        
        # DNA (A, T, C, G)
        if set(text).issubset({'A', 'T', 'C', 'G'}):
            if self._has_sequence_pattern(text, {'A', 'T', 'C', 'G'}):
                return "DNA"
        
        # RNA (A, U, C, G)
        if set(text).issubset({'A', 'U', 'C', 'G'}):
            if self._has_sequence_pattern(text, {'A', 'U', 'C', 'G'}):
                return "RNA"
        
        # Protein (20 standard + common OCR errors)
        if len(text) >= 5 and set(text).issubset(set('ACDEFGHIKLMNPQRSTVWYO0')):
            # Require at least 3 different amino acids to avoid false positives
            if len(set(text)) >= 3:
                return "PROTEIN"
        
        return None
    
    def _has_sequence_pattern(self, text: str, valid_chars: set) -> bool:
        """Check if text has patterns typical of genetic sequences."""
        # Require at least 2 different nucleotides
        unique_chars = set(text) & valid_chars
        if len(unique_chars) < 2:
            return False
        
        # Check character distribution - no single char should dominate >70%
        for char in unique_chars:
            if text.count(char) / len(text) > 0.7:
                return False
        
        return True
    
    def save_visualization(self, image_path: str, sequences: List[Dict], save_path: str):
        """Save visualization with only genetic sequences highlighted."""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw rectangles for each genetic sequence
        for seq in sequences:
            bbox = seq['bbox']
            if bbox is not None:
                # Convert bbox to list of tuples
                points = [(int(p[0]), int(p[1])) for p in bbox]
                
                # Draw polygon
                draw.polygon(points, outline='green', width=3)
                
                # Add label
                label = f"{seq['sequence_type']} ({seq['confidence']:.2f})"
                draw.text((points[0][0], points[0][1] - 20), label, fill='green')
        
        # Save
        image.save(save_path)
        # print(f"Saved genetic sequence visualization to: {save_path}")
