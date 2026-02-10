# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def create_config_from_structure(structure, *, unset=None, config=None):
    if config is None:
        config = {}
    for k, v in structure.items():
        if v is unset:
            continue
        idx = k.find(".")
        if idx == -1:
            config[k] = v
        else:
            sk = k[:idx]
            if sk not in config:
                config[sk] = {}
            create_config_from_structure({k[idx + 1 :]: v}, config=config[sk])
    return config


def convert_layout_json_to_paddleocr(layout_data, page_index=0):
    """Convert layout.json format to PaddleOCR-VL layout boxes format with coordinate scaling.
    
    Args:
        layout_data: Dict with page indices as keys, each containing list of layout items
        page_index: Page index to convert (default: 0)
    
    Returns:
        List of box dicts with cls_id, label, score, coordinate fields
        
    Note: Coordinates are scaled from source dimensions (stored in layout items) to 
    target dimensions (1190x1684 for A4 at 144 DPI). For different page sizes, 
    scaling is applied proportionally.
    """
    page_key = str(page_index)
    if page_key not in layout_data:
        return []
    
    # Get source dimensions from first item with width/height
    source_width = None
    source_height = None
    for item in layout_data[page_key]:
        if "width" in item and "height" in item:
            source_width = item["width"]
            source_height = item["height"]
            break
    
    # Default target dimensions (A4 at 144 DPI)
    # These will be overridden if different page size is detected
    target_width = 1190
    target_height = 1684
    
    # Calculate scaling factors
    scale_x = target_width / source_width if source_width else 1.0
    scale_y = target_height / source_height if source_height else 1.0
    
    boxes = []
    for item in layout_data[page_key]:
        if item.get("type") == "bt_layout":
            continue
        
        item_type = item.get("type", "")
        if item_type == "BT_Text":
            label = "text"
        elif item_type == "BT_RasterPicture":
            label = "image"
        else:
            continue
        
        # Scale coordinates
        left = item["left"] * scale_x
        top = item["top"] * scale_y
        right = item["right"] * scale_x
        bottom = item["bottom"] * scale_y
        
        boxes.append({
            "cls_id": 0,
            "label": label,
            "score": 1.0,
            "coordinate": [left, top, right, bottom]
        })
    
    return boxes
