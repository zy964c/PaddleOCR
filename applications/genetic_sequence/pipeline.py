"""
Genetic Sequence Pipeline - Extends PP-StructureV3 with genetic sequence detection.
Adds genetic_sequence as a new layout class in the standard PP-StructureV3 output.
"""

from typing import List, Dict, Optional
import json
from paddleocr import PPStructureV3


class GeneticSequencePipeline:
    """
    Pipeline that extends PP-StructureV3 with genetic sequence detection.
    Returns standard PP-StructureV3 format with genetic_sequence added as a layout class.
    
    Output format matches PP-StructureV3:
    {
        'input_path': str,
        'page_index': int,
        'boxes': [
            {
                'cls_id': int,
                'label': str,  # 'text', 'table', 'figure', 'genetic_sequence', etc.
                'score': float,
                'coordinate': [x1, y1, x2, y2],
                'text': str,  # for text-based elements
                'sequence_type': str  # 'DNA', 'RNA', 'PROTEIN' (only for genetic_sequence)
            }
        ]
    }
    """
    
    def __init__(
        self,
        min_length: int = 10,
        device: str = "gpu",
        **ocr_kwargs
    ):
        self.min_length = min_length
        # Use simple OCR instead of full PP-StructureV3 to avoid memory issues
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            **ocr_kwargs
        )
    
    def predict(self, input_path: str) -> List[Dict]:
        """
        Run OCR + genetic sequence detection.
        Returns PP-StructureV3-compatible format with genetic_sequence class added.
        """
        # Run OCR
        ocr_results = list(self.ocr.predict(input=input_path))
        
        enhanced_results = []
        
        for res in ocr_results:
            # Get OCR results
            rec_texts = res.get("rec_texts", [])
            dt_polys = res.get("dt_polys", [])
            rec_scores = res.get("rec_scores", [])
            
            # First, identify which texts are genetic sequences
            genetic_indices = set()
            genetic_boxes = []
            
            for i, text in enumerate(rec_texts):
                cleaned = self._clean_text(text)
                seq_type = self._validate_sequence(cleaned)
                
                if seq_type:
                    genetic_indices.add(i)
                    poly = dt_polys[i] if i < len(dt_polys) else None
                    if poly is not None:
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        coordinate = [min(xs), min(ys), max(xs), max(ys)]
                    else:
                        coordinate = [0.0, 0.0, 0.0, 0.0]
                    
                    genetic_boxes.append({
                        'cls_id': 999,
                        'label': 'genetic_sequence',
                        'score': float(rec_scores[i]) if i < len(rec_scores) else 0.0,
                        'coordinate': coordinate,
                        'text': cleaned,
                        'sequence_type': seq_type
                    })
            
            # Create boxes for non-genetic text only
            text_boxes = []
            for i, text in enumerate(rec_texts):
                if i not in genetic_indices:  # Skip genetic sequences
                    poly = dt_polys[i] if i < len(dt_polys) else None
                    if poly is not None:
                        xs = [float(p[0]) for p in poly]
                        ys = [float(p[1]) for p in poly]
                        coordinate = [min(xs), min(ys), max(xs), max(ys)]
                    else:
                        coordinate = [0.0, 0.0, 0.0, 0.0]
                    
                    text_boxes.append({
                        'cls_id': 22,
                        'label': 'text',
                        'score': float(rec_scores[i]) if i < len(rec_scores) else 0.0,
                        'coordinate': coordinate,
                        'text': text
                    })
            
            # Combine: text boxes + genetic boxes
            all_boxes = text_boxes + genetic_boxes
            
            # Create result in PP-StructureV3 format
            enhanced_result = {
                'input_path': res.get('input_path', input_path),
                'page_index': res.get('page_index', 0),
                'boxes': all_boxes
            }
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _clean_text(self, text: str) -> str:
        return text.upper().replace(" ", "").replace("\n", "")
    
    def _validate_sequence(self, text: str) -> Optional[str]:
        if len(text) < self.min_length:
            return None
        
        if set(text).issubset({'A', 'T', 'C', 'G'}):
            return "DNA"
        if set(text).issubset({'A', 'U', 'C', 'G'}):
            return "RNA"
        if len(text) >= 5 and set(text).issubset(set('ACDEFGHIKLMNPQRSTVWY')):
            return "PROTEIN"
        
        return None
    
    def save_to_json(self, results: List[Dict], save_path: str):
        """Save results to JSON file in PP-StructureV3 format."""
        import numpy as np
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        output = {'results': convert_to_serializable(results)}
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved JSON to: {save_path}")
    
    def save_visualization(self, input_path: str, results: List[Dict], save_path: str):
        """Save visualization with all elements including genetic sequences."""
        from PIL import Image, ImageDraw
        
        image = Image.open(input_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        colors = {
            "genetic_sequence": "green",
            "table": "blue",
            "figure": "orange",
            "text": "gray"
        }
        
        for result in results:
            for box in result.get('boxes', []):
                label = box.get('label', 'unknown')
                color = colors.get(label, 'black')
                coord = box.get('coordinate', [])
                
                if len(coord) == 4:
                    x1, y1, x2, y2 = coord
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Label
                    if label == "genetic_sequence":
                        text = f"{box.get('sequence_type')} ({box.get('score', 0):.2f})"
                    else:
                        text = label
                    
                    draw.text((x1, y1 - 20), text, fill=color)
        
        image.save(save_path)
        print(f"Saved visualization to: {save_path}")
