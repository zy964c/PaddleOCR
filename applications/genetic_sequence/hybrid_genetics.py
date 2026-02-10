#!/usr/bin/env python3
"""Hybrid approach: Layout detection + VL OCR on text blocks only."""

from paddleocr import PPStructureV3, PaddleOCRVL
from .detector import GeneticSequenceDetector
from typing import List, Dict, Any
import numpy as np
from PIL import Image


class HybridStructureWithGenetics:
    """Layout detection + VL OCR on blocks + genetic sequence detection."""
    
    def __init__(self, skip_layout_init=False, **kwargs):
        # Only init layout if needed
        if not skip_layout_init:
            layout_kwargs = {
                'use_doc_orientation_classify': False,
                'use_doc_unwarping': False,
                'use_textline_orientation': False,
                'use_table_recognition': False,
                'use_formula_recognition': False,
                'use_chart_recognition': False,
                'use_seal_recognition': False,
                **kwargs
            }
            self.layout_det = PPStructureV3(**layout_kwargs)
        else:
            self.layout_det = None
        
        # VL model for text extraction - minimal config
        try:
            self.vl_ocr = PaddleOCRVL(
                use_layout_detection=False,
                device=kwargs.get('device', 'gpu:0')
            )
            print("[DEBUG] VL OCR initialized successfully")
        except Exception as e:
            print(f"[DEBUG] VL OCR init error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        self.genetic_detector = GeneticSequenceDetector(min_length=6)
    
    def predict(self, input_path: str, layout_result: Dict = None) -> List[Dict[str, Any]]:
        """Run layout detection, then VL OCR on each block."""
        # Step 1: Get layout blocks (fast) or use provided
        if layout_result:
            # Handle both formats: {'res': {...}} or {...}
            if isinstance(layout_result, dict):
                if 'res' in layout_result:
                    results = [layout_result['res']]
                else:
                    results = [layout_result]
            else:
                # List of results
                results = [r['res'] if 'res' in r else r for r in layout_result]
        else:
            results = self.layout_det.predict(input_path)
        
        for result in results:
            # Get image path from result or use provided input_path
            img_path = result.get("input_path", input_path)
            img = Image.open(img_path).convert('RGB')
            boxes = result.get("boxes", [])
            
            rec_texts = []
            rec_scores = []
            dt_polys = []
            
            # Step 2: Run VL OCR on each text block (accurate)
            for box in boxes:
                print(f"[DEBUG] Processing box: label={box.get('label')}, coord={box.get('coordinate', [])[:2]}...")
                if box.get("label") in ["text", "paragraph_title"]:
                    coord = box.get("coordinate", [])
                    if coord:
                        # Crop block
                        if isinstance(coord[0], (list, tuple)):
                            xs = [p[0] for p in coord]
                            ys = [p[1] for p in coord]
                            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        else:
                            x1, y1, x2, y2 = coord
                        
                        print(f"[DEBUG] Cropping region: ({x1}, {y1}, {x2}, {y2})")
                        cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
                        
                        # Save to temp file - VL has issues with numpy arrays
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                            cropped.save(tmp.name, 'PNG')
                            tmp_path = tmp.name
                        
                        try:
                            print(f"[DEBUG] Running VL OCR on temp file: {tmp_path}")
                            vl_result = list(self.vl_ocr.predict(
                                tmp_path,
                                min_pixels=64 * 64,
                                max_pixels=2048 * 2048
                            ))
                            print(f"[DEBUG] VL result length: {len(vl_result)}")
                            if vl_result:
                                result_obj = vl_result[0]
                                if hasattr(result_obj, 'json'):
                                    text = result_obj.json.get("markdown", "").strip()
                                else:
                                    text = result_obj.get("markdown", "").strip()
                                
                                print(f"[DEBUG] Extracted text: {text[:100]}...")
                                rec_texts.append(text)
                                rec_scores.append(0.99)
                                dt_polys.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        except Exception as e:
                            print(f"[DEBUG] VL OCR error: {e}")
                        finally:
                            import os
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
            
            print(f"[DEBUG] Total extracted texts: {len(rec_texts)}")
            
            # Step 3: Merge genetic sequences
            genetic_sequences = self._merge_sequence_lines(rec_texts, rec_scores, dt_polys)
            
            for seq in genetic_sequences:
                boxes.append(seq)
            
            result["boxes"] = boxes
        
        return results
    
    def _merge_sequence_lines(self, rec_texts, rec_scores, dt_polys):
        """Merge consecutive lines that form genetic sequences."""
        genetic_sequences = []
        used = set()
        
        for i in range(len(rec_texts)):
            if i in used:
                continue
                
            text = rec_texts[i]
            if not self.genetic_detector.is_genetic_sequence(text):
                continue
            
            merged_text = text
            polys = [dt_polys[i] if i < len(dt_polys) else [[0,0],[0,0],[0,0],[0,0]]]
            scores = [rec_scores[i] if i < len(rec_scores) else 0.99]
            indices = [i]
            
            j = i + 1
            while j < len(rec_texts):
                combined = merged_text + rec_texts[j]
                if self.genetic_detector.is_genetic_sequence(combined):
                    merged_text = combined
                    polys.append(dt_polys[j] if j < len(dt_polys) else [[0,0],[0,0],[0,0],[0,0]])
                    scores.append(rec_scores[j] if j < len(rec_scores) else 0.99)
                    indices.append(j)
                    j += 1
                else:
                    break
            
            for idx in indices:
                used.add(idx)
            
            poly = self._merge_polys(polys)
            # Ensure score is a Python float
            score_values = [float(s) if isinstance(s, (np.ndarray, np.number)) else s for s in scores]
            score = float(np.mean(score_values))
            
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
