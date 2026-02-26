"""
Identify captions (FigureCaption elements or text elements matching 
"Figure N:" / "Table N:" patterns) and attach them to nearby Image/Table elements.
"""
import re
import base64
from pathlib import Path

from src.logging_config import get_logger

logger = get_logger(__name__)

def to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        path = Path(image_path)
        # If path doesn't exist and is relative, try resolving from project root
        if not path.exists() and not path.is_absolute():
            # Try from current directory
            path = Path.cwd() / image_path
            if not path.exists():
                # Try removing leading '..' and resolve from project root
                clean_path = str(image_path).lstrip('.').lstrip('\\').lstrip('/')
                path = Path.cwd() / clean_path
        
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        logger.warning(f"Image not found: {image_path}")
        return ""

# Types that are considered "visual" elements needing captions
VISUAL_TYPES = {'Image', 'Table'}

# Regex to detect caption-like text in non-FigureCaption elements
# Matches: "Figure 1:", "Table 2:", "Table A.1:", "Fig. 3:" etc.
# Requires a colon after the number to avoid matching regular sentences
CAPTION_PATTERN = re.compile(
    r'^(Figure|Table|Fig\.?)\s+[\dA-Za-z][.\d]*\s*:', re.IGNORECASE
)

def _is_caption(element) -> bool:
    """Check if an element is a caption (FigureCaption type or text matching pattern)."""
    d = element.to_dict()
    etype = d.get('type', '')
    text = d.get('text', '').strip()
    
    if etype == 'FigureCaption':
        return True
    if etype not in VISUAL_TYPES and CAPTION_PATTERN.match(text):
        return True
    return False


def _find_caption(elements, visual_idx: int, visual_type: str, window: int = 3):
    """
    Find a caption for a visual element, looking in the correct direction:
      - Image: look AFTER (captions appear below images)
      - Table: look BEFORE (captions appear above tables)
    
    Returns (caption_idx, caption_text) or (None, None).
    """
    n = len(elements)
    
    if visual_type == 'Image':
        # Look forward (caption is below image)
        for offset in range(1, window + 1):
            i = visual_idx + offset
            if i < n and _is_caption(elements[i]):
                return i, elements[i].to_dict().get('text', '').strip()
    
    elif visual_type == 'Table':
        # Look backward (caption is above table)
        for offset in range(1, window + 1):
            i = visual_idx - offset
            if i >= 0 and _is_caption(elements[i]):
                return i, elements[i].to_dict().get('text', '').strip()
    
    return None, None


def attach_captions(elements, window: int = 3) -> list:
    """
    Attach captions to their corresponding Image/Table elements in-place,
    then remove the caption elements from the list.
    
    Direction convention (academic papers):
      - Image: caption is BELOW → look at next elements
      - Table: caption is ABOVE → look at previous elements
    
    Behavior:
      - Image elements: 'text' is replaced with the caption text
      - Table elements: caption is prepended to existing 'text'
      - Matched caption elements (FigureCaption / inferred) are removed
      - Unmatched FigureCaption elements are also removed
    
    Returns:
        list - filtered elements with captions merged and caption elements removed
    """
    indices_to_remove = set()

    # Pass 1: iterate visual elements and find their captions
    for idx, ele in enumerate(elements):
        d = ele.to_dict()
        etype = d.get('type', '')
        
        if etype not in VISUAL_TYPES:
            continue
        
        c_idx, caption_text = _find_caption(elements, idx, etype, window)
        
        if c_idx is None:
            continue
        
        indices_to_remove.add(c_idx)
        
        if etype == 'Image':
            elements[idx].text = caption_text
            elements[idx].metadata.caption = caption_text
        elif etype == 'Table':
            original_text = d.get('text', '').strip()
            elements[idx].text = f"{caption_text}\n\n{original_text}" if original_text else caption_text
            elements[idx].metadata.caption = caption_text

    # Pass 2: remove any remaining FigureCaption elements (unmatched ones)
    for idx, ele in enumerate(elements):
        if idx not in indices_to_remove and ele.to_dict().get('type', '') == 'FigureCaption':
            indices_to_remove.add(idx)

    filtered = [ele for idx, ele in enumerate(elements) if idx not in indices_to_remove]

    message = f"Summary: Removed {len(indices_to_remove)} caption elements. {len(filtered)} elements remaining (was {len(elements)})."

    return filtered, message

