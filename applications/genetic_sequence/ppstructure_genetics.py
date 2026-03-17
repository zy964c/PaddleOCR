#!/usr/bin/env python3
"""Full PP-StructureV3 pipeline with genetic sequence detection."""

from paddleocr import PPStructureV3
from .detector import GeneticSequenceDetector
from typing import List, Dict, Any
import numpy as np


class PPStructureV3WithGenetics:
    """PP-StructureV3 pipeline extended with genetic sequence detection."""
    
    def __init__(self, **kwargs):
        """Initialize PP-StructureV3 and genetic detector."""
        # Use Cyrillic OCR model explicitly
        if 'text_recognition_model_name' not in kwargs:
            kwargs['text_recognition_model_name'] = 'cyrillic_PP-OCRv5_mobile_rec'
        
        self.layout_det = PPStructureV3(**kwargs)
        self.genetic_detector = GeneticSequenceDetector(min_length=6)
    
    def visualize(self, result: Dict, save_path: str):
        """Visualize all boxes including genetic sequences."""
        from PIL import Image, ImageDraw
        
        img = Image.open(result["input_path"]).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        colors = {
            'genetic_sequence': 'green',
            'table': 'blue',
            'text': 'red',
            'paragraph_title': 'purple',
            'number': 'orange'
        }
        
        for box in result.get("boxes", []):
            label = box.get("label", "")
            coord = box.get("coordinate", [])
            
            if len(coord) == 4:
                # Handle both polygon [[x,y],[x,y]...] and bbox [x1,y1,x2,y2] formats
                if isinstance(coord[0], (list, tuple)):
                    pts = [(int(p[0]), int(p[1])) for p in coord]
                else:
                    # Convert bbox to polygon
                    x1, y1, x2, y2 = coord
                    pts = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
                
                draw.polygon(pts, outline=colors.get(label, 'gray'), width=3)
                draw.text((pts[0][0], max(0, pts[0][1]-15)), label, fill=colors.get(label, 'gray'))
        
        img.save(save_path)
    
    def predict(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Run full PP-StructureV3 with genetic sequence detection.
        
        Args:
            input_path: Path to image or PDF
            
        Returns:
            List of results with layout elements and genetic sequences
        """
        results = self.layout_det.predict(input_path)
        
        for result in results:
            ocr_res = result.get("overall_ocr_res", {})
            rec_texts = ocr_res.get("rec_texts", [])
            rec_scores = ocr_res.get("rec_scores", [])
            dt_polys = ocr_res.get("dt_polys", [])
            
            # Merge multi-line genetic sequences
            genetic_sequences = self._merge_sequence_lines(rec_texts, rec_scores, dt_polys)
            
            boxes = result.get("boxes", [])
            if not boxes:
                layout_det_res = result.get("layout_det_res", {})
                boxes = layout_det_res.get("boxes", [])
            
            for box in boxes:
                if not box.get("text"):
                    box_coord = box.get("coordinate", [])
                    if box_coord:
                        if isinstance(box_coord[0], (list, tuple)):
                            xs = [p[0] for p in box_coord]
                            ys = [p[1] for p in box_coord]
                            box_bbox = [min(xs), min(ys), max(xs), max(ys)]
                        else:
                            box_bbox = box_coord
                        
                        box_texts = []
                        for i, poly in enumerate(dt_polys):
                            if isinstance(poly, np.ndarray):
                                poly = poly.tolist()
                            xs = [p[0] for p in poly]
                            ys = [p[1] for p in poly]
                            ocr_bbox = [min(xs), min(ys), max(xs), max(ys)]
                            
                            if self._boxes_overlap(box_bbox, ocr_bbox):
                                if i < len(rec_texts):
                                    box_texts.append(rec_texts[i])
                        
                        if box_texts:
                            box["text"] = " ".join(box_texts)
            
            # Reclassify boxes containing genetic sequences
            for box in boxes:
                box_text = box.get("text", "")
                if box_text and self.genetic_detector.is_genetic_sequence(box_text):
                    # print(f"[DEBUG] Reclassifying '{box.get('label')}' as genetic_sequence: {box_text[:50]}...")
                    box["cls_id"] = 999
                    box["label"] = "genetic_sequence"
                    box["sequence_type"] = self.genetic_detector.get_sequence_type(box_text)
            
            existing_texts = {b.get("text") for b in boxes}
            for seq in genetic_sequences:
                if seq["text"] not in existing_texts:
                    boxes.append(seq)
            
            result["boxes"] = boxes
        
        return results
    
    def _merge_sequence_lines(self, rec_texts, rec_scores, dt_polys):
        """Merge consecutive lines that form genetic sequences."""
        # print(f"[DEBUG] Total OCR lines: {len(rec_texts)}")
        # for idx, txt in enumerate(rec_texts):
        #     if "ARGGVHAYAYAPAAFDP" in txt.upper():
                # print(f"[DEBUG] Found target at line {idx}: {txt}")
        
        genetic_sequences = []
        used = set()
        
        for i in range(len(rec_texts)):
            if i in used:
                continue
                
            text = rec_texts[i]
            is_valid = self.genetic_detector.is_genetic_sequence(text)
            
            # if "ARGGVHAYAYAPAAFDP" in text.upper():
                # print(f"[DEBUG] Line {i} validation: text='{text}', is_valid={is_valid}")
            
            if not is_valid:
                continue
            
            # print(f"[DEBUG] Starting merge at line {i}: {text[:50]}...")
            
            # Start merging from this line
            merged_text = text
            polys = [dt_polys[i] if i < len(dt_polys) else [[0,0],[0,0],[0,0],[0,0]]]
            scores = [rec_scores[i] if i < len(rec_scores) else 0.99]
            indices = [i]
            
            # Try merging with ALL following consecutive lines
            j = i + 1
            while j < len(rec_texts):
                next_text = rec_texts[j]
                
                # Stop if next line has Cyrillic
                if any('\u0400' <= c <= '\u04FF' for c in next_text):
                    # print(f"[DEBUG] Stopping merge - Cyrillic detected in line {j}")
                    break
                
                combined = merged_text + next_text
                is_valid = self.genetic_detector.is_genetic_sequence(combined)
                # print(f"[DEBUG] Trying to merge line {j}: {next_text[:50]}... -> valid={is_valid}")
                
                if is_valid:
                    merged_text = combined
                    polys.append(dt_polys[j] if j < len(dt_polys) else [[0,0],[0,0],[0,0],[0,0]])
                    scores.append(rec_scores[j] if j < len(rec_scores) else 0.99)
                    indices.append(j)
                    j += 1
                else:
                    break
            
            # print(f"[DEBUG] Final merged indices: {indices}, text length: {len(merged_text)}")
            
            # Mark all merged lines as used
            for idx in indices:
                used.add(idx)
            
            poly = self._merge_polys(polys)
            score = float(np.mean(scores))
            
            genetic_sequences.append({
                "cls_id": 999,
                "label": "genetic_sequence",
                "score": score,
                "coordinate": poly,
                "text": merged_text,
                "sequence_type": self.genetic_detector.get_sequence_type(merged_text)
            })
        
        return genetic_sequences
    
    def _merge_polys(self, polys):
        """Merge multiple polygons into one bounding polygon."""
        all_coords = []
        for poly in polys:
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()
            all_coords.extend(poly)
        
        xs = [float(p[0]) for p in all_coords]
        ys = [float(p[1]) for p in all_coords]
        return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
    
    def _boxes_overlap(self, bbox1, bbox2, threshold=0.5):
        """Check if two bounding boxes overlap significantly"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        if intersection == 0:
            return False
        
        # Calculate area of smaller box
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        smaller_area = min(area1, area2)
        
        # Check if overlap is significant
        overlap_ratio = intersection / smaller_area
        result = overlap_ratio > threshold
        
        # if overlap_ratio > 0.1:  # Debug significant overlaps
        #     print(f"[DEBUG] Overlap check: ratio={overlap_ratio:.2f}, threshold={threshold}, result={result}")
        #     print(f"  bbox1={bbox1}, bbox2={bbox2}")
        
        return result
