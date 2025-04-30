import cv2
import numpy as np
from PIL import Image
import argparse
import os
import json
import pytesseract
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DEBUG_FOLDER = 'debug'
if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

CONFIG = {
    # String similarity matching
    'NAME_MATCH_THRESHOLD': 0.5,  # Threshold for string similarity matching
    
    # Image preprocessing
    'BILATERAL_FILTER_D': 11,     # Diameter of each pixel neighborhood
    'BILATERAL_FILTER_SIGMA_COLOR': 17,  # Filter sigma in the color space
    'BILATERAL_FILTER_SIGMA_SPACE': 17,  # Filter sigma in the coordinate space
    'ADAPTIVE_THRESH_BLOCK_SIZE': 15,    # Block size for adaptive thresholding
    'ADAPTIVE_THRESH_C': 5,       # Constant subtracted from the mean
    'MORPH_KERNEL_SIZE': (2, 2),  # Kernel size for morphological operations
    
    # Signature detection
    'BORDER_WIDTH': 5,            # Border width to ignore in signature detection
    'MIN_SIGNATURE_AREA': 50,     # Minimum area of signature components
    'MIN_SIGNATURE_WIDTH': 15,    # Minimum width of signature
    'MIN_SIGNATURE_HEIGHT': 8,    # Minimum height of signature
    'MIN_WHITE_PIXELS': 30,       # Minimum number of white pixels to consider as signature
    'MIN_SIGNATURE_DENSITY': 0.04, # Minimum density of white pixels in signature region (4%)
    'WHITE_THRESHOLD': 170,       # Threshold for white pixels (0-255) - higher value is more strict
    'MIN_VALID_COMPONENTS': 1,    # Minimum number of valid components to consider as signature
    'EMPTY_CELL_BRIGHTNESS': 0.9, # Brightness threshold for empty cell detection (90% bright pixels)
    
    # Table detection
    'HORIZONTAL_KERNEL_SIZES': [20, 30, 40],  # Kernel sizes for horizontal line detection
    'VERTICAL_KERNEL_SIZES': [20, 30, 40],    # Kernel sizes for vertical line detection
    'TABLE_MIN_WIDTH_FACTOR': 0.1,    # Minimum table width as factor of image width
    'TABLE_MIN_HEIGHT_FACTOR': 0.1,   # Minimum table height as factor of image height
    'TABLE_EDGE_MARGIN': 5,            # Minimum distance from table to image edge
    
    # Cell detection
    'MIN_CELL_AREA_FACTOR': 0.005,     # Minimum cell area as percentage of table area
    'MIN_CELL_WIDTH': 5,               # Minimum cell width in pixels
    'MIN_CELL_HEIGHT': 5,              # Minimum cell height in pixels
    'COLUMN_X_TOLERANCE_FACTOR': 0.04, # X-coordinate tolerance for column grouping as factor of table width
    
    # Name column detection
    'MIN_NAME_COLUMN_CELLS': 5,        # Minimum number of cells that should contain text in name column
    'MIN_NAME_LENGTH': 4,              # Minimum length of text to consider as a name
    'NAME_COLUMN_WIDTH_FACTOR': 0.2,   # Name column should be at least this wide relative to table width
    
    # Visualization
    'OVERLAY_ALPHA': 0.2,              # Alpha transparency for cell highlighting
    'TEXT_SCALE': 0.7,                 # Scale for column number text
    'TEXT_THICKNESS': 3,               # Thickness for text outline
    'SIGNATURE_TEXT_SCALE': 0.5,       # Scale for signature status text
}

def string_similarity(a, b):
    """Calculate string similarity between two strings using SequenceMatcher"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def name_similarity(name1, name2):
    """Calculate similarity between two names, handling different formats and order of parts"""
    # Clean and normalize both names
    name1 = clean_extracted_name(name1.lower())
    name2 = clean_extracted_name(name2.lower())
    
    # If either name is empty, return 0
    if not name1 or not name2:
        return 0.0
    
    direct_similarity = string_similarity(name1, name2)
    
    parts1 = name1.split()
    parts2 = name2.split()
    
    parts_diff_penalty = 1.0
    if abs(len(parts1) - len(parts2)) > 1:
        parts_diff_penalty = 0.8
    
    best_part_matches = []
    for part1 in parts1:
        if len(part1) < 2: 
            continue
        best_match = 0.0
        for part2 in parts2:
            if len(part2) < 2:
                continue
            similarity = string_similarity(part1, part2)
            best_match = max(best_match, similarity)
        best_part_matches.append(best_match)
    
    part_similarity = sum(best_part_matches) / len(best_part_matches) if best_part_matches else 0.0

    combined_similarity = (direct_similarity * 0.6 + part_similarity * 0.4) * parts_diff_penalty
    
    return combined_similarity

def find_matching_name(table_name, name_list, threshold=CONFIG['NAME_MATCH_THRESHOLD']):
    """Find the best matching name from a list based on string similarity with improved speed."""
    if not table_name or not name_list:
        return None
    
    best_match = None
    best_ratio = 0
    
    table_name = clean_extracted_name(table_name)
    
    # Quick filter: If exact match exists, return immediately
    for name in name_list:
        clean_name = clean_extracted_name(name)
        if table_name.lower() == clean_name.lower():
            return name
    
    # For longer lists, first check if any name is contained within the table name
    # This can be much faster than calculating similarity scores for all names
    for name in name_list:
        clean_name = clean_extracted_name(name).lower()
        if len(clean_name) > 5 and clean_name in table_name.lower():
            # If a substantial part of a name is found, prioritize it
            ratio = 0.8  # High starting similarity
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = name
    
    # If we found a good match with the quick check, return it
    if best_match and best_ratio > 0.7:
        return best_match
    
    # Otherwise do more comprehensive matching
    for name in name_list:
        clean_name = clean_extracted_name(name)
        # Skip very short names for performance
        if len(clean_name) < 3:
            continue
            
        ratio = name_similarity(table_name, clean_name)
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = name
    
    return best_match

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.bilateralFilter(
        gray, 
        CONFIG['BILATERAL_FILTER_D'], 
        CONFIG['BILATERAL_FILTER_SIGMA_COLOR'], 
        CONFIG['BILATERAL_FILTER_SIGMA_SPACE']
    )
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        CONFIG['ADAPTIVE_THRESH_BLOCK_SIZE'], 
        CONFIG['ADAPTIVE_THRESH_C']
    )
    
    kernel = np.ones(CONFIG['MORPH_KERNEL_SIZE'], np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def check_for_signature(image, x, y, w, h):
    """Detect signatures based on white text patterns on dark background."""
    cell_region = image[y:y+h, x:x+w]
    
    if len(cell_region.shape) == 3:
        gray = cv2.cvtColor(cell_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_region
    
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = np.sum(hist)
    bright_pixels = np.sum(hist[220:256])
    dark_pixels = np.sum(hist[0:50])
    
    bright_pixels_ratio = float(bright_pixels) / total_pixels if total_pixels > 0 else 0
    dark_pixels_ratio = float(dark_pixels) / total_pixels if total_pixels > 0 else 0
    
    avg_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    noise_level = std_intensity
    
    if bright_pixels_ratio > CONFIG['EMPTY_CELL_BRIGHTNESS'] or avg_intensity > 245:
        with open(os.path.join(DEBUG_FOLDER, 'detection_log.txt'), 'a') as f:
            f.write(f"Cell {x},{y}: EMPTY (bright_pixels_ratio={bright_pixels_ratio:.4f}, avg_intensity={avg_intensity:.2f})\n")
            
        return {
            "has_content": False,
            "confidence": 0.95,
            "pixel_density": 0,
            "debug_info": {
                "num_components": 0,
                "white_pixels": 0,
                "signature_width": 0,
                "signature_height": 0,
                "bounds": None,
                "valid_components": [],
                "bright_pixels_ratio": bright_pixels_ratio,
                "avg_intensity": avg_intensity,
                "std_intensity": std_intensity,
                "noise_level": noise_level
            }
        }
    
    # Create border mask to ignore cell borders
    border_mask = np.ones_like(gray)
    border_width = CONFIG['BORDER_WIDTH']
    border_mask[0:border_width, :] = 0
    border_mask[-border_width:, :] = 0
    border_mask[:, 0:border_width] = 0
    border_mask[:, -border_width:] = 0
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges (signatures)
    gray_filtered = cv2.bilateralFilter(gray_eq, 5, 50, 50)
    
    gray_inv = cv2.bitwise_not(gray_filtered)
    
    _, binary_otsu = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(
        gray_inv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  
        2    
    )
    
    # Combine both methods - take the intersection to reduce noise
    binary = cv2.bitwise_and(binary_otsu, binary_adaptive)
    
    binary = binary * border_mask
    
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    min_component_size = 10 
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    filtered_binary = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
            filtered_binary[labels == i] = 255
    
    binary = filtered_binary
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_components = []
    total_signature_area = 0
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    
    stroke_like_count = 0
    max_stroke_len = 0
    total_perimeter = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < CONFIG['MIN_SIGNATURE_AREA']:
            continue
            
        x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
        
        if x_c <= border_width or y_c <= border_width or \
           x_c + w_c >= w - border_width or y_c + h_c >= h - border_width:
            continue
        
        aspect_ratio = float(w_c) / h_c if h_c > 0 else 0
        if aspect_ratio > 8 or aspect_ratio < 0.1:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        total_perimeter += perimeter
        
        complexity = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else 0
        
        if complexity > 1.8:
            stroke_like_count += 1
            stroke_len = perimeter / 2
            max_stroke_len = max(max_stroke_len, stroke_len)
        
        min_x = min(min_x, x_c)
        min_y = min(min_y, y_c)
        max_x = max(max_x, x_c + w_c)
        max_y = max(max_y, y_c + h_c)
        
        total_signature_area += area
        valid_components.append(contour)
    
    # Calculate signature metrics
    if valid_components:
        signature_width = max_x - min_x
        signature_height = max_y - min_y
        
        # Create signature region mask
        signature_mask = np.zeros_like(gray)
        cv2.drawContours(signature_mask, valid_components, -1, 255, -1)
        signature_mask = signature_mask & border_mask
        
        # Calculate active signature pixels and total area
        active_pixels = np.count_nonzero(signature_mask)
        total_area = np.count_nonzero(border_mask)
        
        # Calculate pixel density within signature region
        signature_region_mask = np.zeros_like(gray)
        signature_region_mask[min_y:max_y, min_x:max_x] = 1
        signature_region_mask = signature_region_mask & border_mask
        signature_region_area = np.count_nonzero(signature_region_mask)
        
        signature_coverage = active_pixels / signature_region_area if signature_region_area > 0 else 0
        
        white_mask = (gray_inv > CONFIG['WHITE_THRESHOLD']) & signature_region_mask
        white_pixels = np.count_nonzero(white_mask)
        
        region_pixels = gray_inv[signature_region_mask > 0]
        if len(region_pixels) > 0:
            region_mean = np.mean(region_pixels)
            region_std = np.std(region_pixels)
        else:
            region_mean = 0
            region_std = 0
        
        if signature_region_area > 0 and region_std > 0:
            signal_threshold = region_mean + 1.5 * region_std
            signal_mask = (gray_inv > signal_threshold) & signature_region_mask
            signal_pixels = np.count_nonzero(signal_mask)
            
            pixel_density = signal_pixels / signature_region_area
            
            stroke_adjustment = stroke_like_count * 0.01
            pixel_density = pixel_density + stroke_adjustment
        else:
            pixel_density = 0
        
        overall_density = active_pixels / total_area if total_area > 0 else 0
        
        stroke_density = total_perimeter / total_area if total_area > 0 else 0
        
        # Check for "too perfect" shapes
        is_too_regular = False
        for contour in valid_components:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(cv2.contourArea(contour)) / hull_area if hull_area > 0 else 0
            
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if solidity > 0.98 or circularity > 0.9:
                is_too_regular = True
                break
        
        signature_score = (
            min(1.0, max_stroke_len / 50) * 0.3 +         # Longest stroke length
            min(1.0, stroke_like_count / 3) * 0.2 +       # Number of stroke-like shapes
            min(1.0, signature_coverage * 5) * 0.2 +     # Signature coverage
            min(1.0, pixel_density * 10) * 0.3           # Adjusted pixel density
        )
        
        has_signature = (
            len(valid_components) >= CONFIG['MIN_VALID_COMPONENTS'] and
            signature_width >= CONFIG['MIN_SIGNATURE_WIDTH'] and
            signature_height >= CONFIG['MIN_SIGNATURE_HEIGHT'] and
            white_pixels >= CONFIG['MIN_WHITE_PIXELS'] and
            stroke_like_count >= 1 and
            signature_score >= 0.3 and
            not is_too_regular
        )
        
        confidence = min(0.9, (
            (signature_width / w) * 0.2 +  # Width contribution
            (signature_height / h) * 0.2 +  # Height contribution
            (stroke_like_count / 3) * 0.3 +  # Number of stroke-like components
            (signature_score) * 0.3  # Overall signature score
        ))
        
        # Record result for debugging
        result = "SIGNATURE" if has_signature else "NO_SIGNATURE"
        with open(os.path.join(DEBUG_FOLDER, 'detection_log.txt'), 'a') as f:
            f.write(f"Cell {x},{y}: {result}, "
                   f"Components: {len(valid_components)}, "
                   f"StrokeComponents: {stroke_like_count}, "
                   f"Size: {signature_width}x{signature_height}, "
                   f"Density: {pixel_density:.4f}, "
                   f"StrokeDensity: {stroke_density:.4f}, "
                   f"Score: {signature_score:.4f}, "
                   f"WhitePixels: {white_pixels}, "
                   f"Regular: {is_too_regular}\n")
    else:
        has_signature = False
        confidence = 0.95  # High confidence that it's empty
        white_pixels = 0
        pixel_density = 0
        overall_density = 0
        signature_width = 0
        signature_height = 0
        is_too_regular = False
        stroke_like_count = 0
        stroke_density = 0
        signature_score = 0
        region_mean = 0
        region_std = 0
        
        # Record empty result
        with open(os.path.join(DEBUG_FOLDER, 'detection_log.txt'), 'a') as f:
            f.write(f"Cell {x},{y}: EMPTY (no valid components)\n")
    
    return {
        "has_content": has_signature,
        "confidence": round(confidence, 2),
        "pixel_density": round(pixel_density, 4),
        "debug_info": {
            "num_components": len(valid_components) if 'valid_components' in locals() else 0,
            "stroke_components": stroke_like_count if 'stroke_like_count' in locals() else 0,
            "white_pixels": white_pixels if 'white_pixels' in locals() else 0,
            "signature_width": signature_width if 'signature_width' in locals() else 0,
            "signature_height": signature_height if 'signature_height' in locals() else 0,
            "signature_score": signature_score if 'signature_score' in locals() else 0,
            "bounds": (min_x, min_y, max_x, max_y) if 'valid_components' in locals() and valid_components else None,
            "valid_components": valid_components if 'valid_components' in locals() else [],
            "bright_pixels_ratio": bright_pixels_ratio if 'bright_pixels_ratio' in locals() else None,
            "overall_density": overall_density if 'overall_density' in locals() else None,
            "is_too_regular": is_too_regular if 'is_too_regular' in locals() else None,
            "noise_level": noise_level
        }
    }

def highlight_signature(image, x, y, w, h, signature_info):
    """Highlight signature in the output image."""
    if signature_info["has_content"]:
        bounds = signature_info["debug_info"]["bounds"]
        if bounds:
            min_x, min_y, max_x, max_y = bounds
            
            overlay = image.copy()
            cv2.rectangle(overlay, 
                        (x + min_x, y + min_y), 
                        (x + max_x, y + max_y), 
                        (0, 255, 0), 2)  # Green rectangle around signature
            
            # Draw contours
            for contour in signature_info["debug_info"]["valid_components"]:
                # Adjust contour coordinates
                contour = contour + np.array([[x, y]])
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 1)
            
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            confidence = signature_info["confidence"]
            pixel_density = signature_info["pixel_density"]
            components = signature_info["debug_info"]["num_components"]
            stroke_components = signature_info["debug_info"].get("stroke_components", 0)
            signature_score = signature_info["debug_info"].get("signature_score", 0)
            
            text = f"Signed ({confidence:.2f})"
            density_text = f"Den: {pixel_density:.3f} Score: {signature_score:.2f}"
            comp_text = f"Comp: {components} Strokes: {stroke_components}"
            
            text_x = x + 5
            text_y = y + h - 35
            density_y = y + h - 20
            comp_y = y + h - 5
            
            # Draw text with dark outline for better visibility
            cv2.putText(image, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
            cv2.putText(image, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # Green text
                       
            # Draw density value with dark outline
            cv2.putText(image, density_text, (text_x, density_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
            cv2.putText(image, density_text, (text_x, density_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # Green text
                       
            # Draw components info
            cv2.putText(image, comp_text, (text_x, comp_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)  # Black outline
            cv2.putText(image, comp_text, (text_x, comp_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)  # Green text
    else:
        pixel_density = signature_info["pixel_density"]
        components = signature_info["debug_info"].get("num_components", 0)
        stroke_components = signature_info["debug_info"].get("stroke_components", 0)
        signature_score = signature_info["debug_info"].get("signature_score", 0)
        
        text_x = x + 5
        text_y = y + h - 10
        
        metrics_text = f"Den:{pixel_density:.3f} Sc:{signature_score:.2f} Cmp:{components}"
        
        cv2.putText(image, metrics_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)  # Black outline
        cv2.putText(image, metrics_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)  # Red text

def clean_extracted_name(text):
    """Clean and normalize extracted name text from OCR."""
    if not text:
        return ""
    
    text = text.replace('|', 'I').replace('1', 'I')
    text = text.replace('0', 'O').replace('5', 'S')
    
    text = text.replace('\\', '').replace('/', '')
    text = text.replace('_', '').replace('=', '').replace('+', '')
    text = text.replace('@', '').replace('#', '').replace('$', '')
    text = text.replace('%', '').replace('^', '').replace('*', '')
    text = text.replace('(', '').replace(')', '').replace('{', '')
    text = text.replace('}', '').replace('[', '').replace(']', '')
    
    text = ''.join(c for c in text if not c.isdigit())
    
    text = ''.join(c for c in text if c.isalpha() or c.isspace() or c in '.-')
    
    text = ' '.join(text.split())
    text = text.strip('.-_ ')
    
    text = ' '.join(word for word in text.split() if len(word) > 1)
    
    return text

def extract_text_from_cell(image, x, y, w, h):
    """Extract text from a cell using OCR with optimized settings for speed."""
    cell_region = image[y:y+h, x:x+w]
    
    if w < 20 or h < 10:
        return ""
    
    cell_pil = Image.fromarray(cv2.cvtColor(cell_region, cv2.COLOR_BGR2RGB))
    
    config = '--psm 6 --oem 3'
    text = pytesseract.image_to_string(cell_pil, config=config).strip()
    cleaned_text = clean_extracted_name(text)
    
    if not cleaned_text:
        gray_cell = cv2.cvtColor(cell_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        binary_pil = Image.fromarray(binary)
        text = pytesseract.image_to_string(binary_pil, config=config).strip()
        cleaned_text = clean_extracted_name(text)
    
    return cleaned_text

def detect_table_and_cells(image_path, name_list=None):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image. Please check the file path.")

        output_image = image.copy()

        thresh = preprocess_image(image)
        
        height, width = thresh.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'preprocessed.jpg'), thresh)
        print(f"Saved preprocessed image as {os.path.join(DEBUG_FOLDER, 'preprocessed.jpg')}")
        
        debug_log_path = os.path.join(DEBUG_FOLDER, 'detection_log.txt')
        with open(debug_log_path, 'w') as f:
            f.write(f"Starting detection on image: {image_path}\n")
            f.write(f"Image dimensions: {width}x{height}\n")
            f.write("-" * 50 + "\n")
            
        results = {
            "students": []
        }
        
        horizontal_lines = np.zeros_like(thresh)
        for kernel_size in CONFIG['HORIZONTAL_KERNEL_SIZES']:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
            lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            horizontal_lines = cv2.bitwise_or(horizontal_lines, lines)

        vertical_lines = np.zeros_like(thresh)
        for kernel_size in CONFIG['VERTICAL_KERNEL_SIZES']:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
            lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            vertical_lines = cv2.bitwise_or(vertical_lines, lines)

        table_mask = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
        
        kernel = np.ones((3,3), np.uint8)
        table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
        
        cv2.imwrite(os.path.join(DEBUG_FOLDER, 'table_mask.jpg'), table_mask)
        print(f"Saved table mask as {os.path.join(DEBUG_FOLDER, 'table_mask.jpg')}")
        
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} potential tables")
        
        with open(debug_log_path, 'a') as f:
            f.write(f"Found {len(contours)} potential tables\n")

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            min_width = width * CONFIG['TABLE_MIN_WIDTH_FACTOR']
            min_height = height * CONFIG['TABLE_MIN_HEIGHT_FACTOR']
            edge_margin = CONFIG['TABLE_EDGE_MARGIN']
            
            if (w > min_width and h > min_height and
                x > edge_margin and y > edge_margin and
                x + w < width - edge_margin and y + h < height - edge_margin):
                
                print(f"Processing table at position ({x}, {y}) with size {w}x{h}")
                
                with open(debug_log_path, 'a') as f:
                    f.write(f"Processing table at position ({x}, {y}) with size {w}x{h}\n")
                    f.write("-" * 40 + "\n")
                
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                table_region = thresh[y:y+h, x:x+w]
                
                cv2.imwrite(os.path.join(DEBUG_FOLDER, 'table_region.jpg'), table_region)
                print(f"Saved table region as {os.path.join(DEBUG_FOLDER, 'table_region.jpg')}")
                
                table_mask_region = table_mask[y:y+h, x:x+w]
                
                cell_contours, hierarchy = cv2.findContours(
                    table_mask_region, 
                    cv2.RETR_CCOMP, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                print(f"Found {len(cell_contours)} potential cells")
                
                valid_cells = []
                min_cell_area = (w * h) * CONFIG['MIN_CELL_AREA_FACTOR']
                min_cell_width = CONFIG['MIN_CELL_WIDTH']
                min_cell_height = CONFIG['MIN_CELL_HEIGHT']
                
                for cell_contour in cell_contours:
                    area = cv2.contourArea(cell_contour)
                    if area < min_cell_area:
                        continue
                        
                    cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cell_contour)
                    
                    if (cell_w > min_cell_width and cell_h > min_cell_height and
                        cell_x >= 0 and cell_y >= 0 and 
                        cell_x + cell_w <= w and cell_y + cell_h <= h):
                        valid_cells.append((cell_x, cell_y, cell_w, cell_h))
                
                print(f"Found {len(valid_cells)} valid cells")
                
                with open(debug_log_path, 'a') as f:
                    f.write(f"Found {len(valid_cells)} valid cells\n")
                
                if len(valid_cells) > 0:
                    valid_cells.sort(key=lambda cell: cell[0])
                    
                    min_x = min(cell[0] for cell in valid_cells)
                    max_x = max(cell[0] + cell[2] for cell in valid_cells)
                    table_width = max_x - min_x if max_x > min_x else w
                    
                    columns = {}
                    tolerance = table_width * CONFIG['COLUMN_X_TOLERANCE_FACTOR']
                    
                    for cell in valid_cells:
                        cell_x, cell_y, cell_w, cell_h = cell
                        assigned = False
                        for col_x in columns.keys():
                            if abs(cell_x - col_x) < tolerance:
                                columns[col_x].append(cell)
                                assigned = True
                                break
                        if not assigned:
                            columns[cell_x] = [cell]
                    
                    sorted_columns = sorted(columns.items(), key=lambda x: x[0])
                    print(f"Found {len(sorted_columns)} columns")
                    
                    with open(debug_log_path, 'a') as f:
                        f.write(f"Found {len(sorted_columns)} columns\n")
                        f.write(f"Column positions: {[col[0] for col in sorted_columns]}\n")
                        f.write(f"Cells per column: {[len(col[1]) for col in sorted_columns]}\n")
                        f.write("-" * 40 + "\n")
                    
                    for cell in valid_cells:
                        cell_x, cell_y, cell_w, cell_h = cell
                        cv2.rectangle(output_image, 
                                    (x + cell_x, y + cell_y), 
                                    (x + cell_x + cell_w, y + cell_y + cell_h), 
                                    (102, 255, 255), 1)  # Light yellow in BGR format
                    
                    def highlight_column(column_cells, color, column_number):
                        sorted_cells = sorted(column_cells, key=lambda cell: cell[1])
                        
                        if sorted_cells:
                            cell_x, cell_y, cell_w, cell_h = sorted_cells[0]  # Get first cell position
                            text_x = x + cell_x + cell_w // 2 - 10  # Center the number
                            text_y = y + cell_y - 10  # Place above the column
                            cv2.putText(output_image, str(column_number), (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, CONFIG['TEXT_SCALE'], (0, 0, 0), CONFIG['TEXT_THICKNESS'])
                            cv2.putText(output_image, str(column_number), (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, CONFIG['TEXT_SCALE'], (255, 255, 255), 1)
                        
                        for cell in sorted_cells[1:]:  # Skip header
                            cell_x, cell_y, cell_w, cell_h = cell
                            cv2.rectangle(output_image, 
                                        (x + cell_x, y + cell_y), 
                                        (x + cell_x + cell_w, y + cell_y + cell_h), 
                                        color, 2)
                                        
                            overlay = output_image.copy()
                            cv2.rectangle(overlay, 
                                        (x + cell_x, y + cell_y), 
                                        (x + cell_x + cell_w, y + cell_y + cell_h), 
                                        color, -1)
                            cv2.addWeighted(overlay, CONFIG['OVERLAY_ALPHA'], output_image, 1 - CONFIG['OVERLAY_ALPHA'], 0, output_image)

                    if len(sorted_columns) > 0:
                        names_column_idx = identify_names_column(sorted_columns, image, x, y, w, h)
                        if names_column_idx is not None:
                            print(f"Identified names column at index {names_column_idx + 1}")
                            
                            with open(debug_log_path, 'a') as f:
                                f.write(f"Identified names column at index {names_column_idx + 1}\n")
                        else:
                            print("Could not identify names column, using default second column")
                            
                            with open(debug_log_path, 'a') as f:
                                f.write("Could not identify names column, using default second column\n")
                                
                            names_column_idx = 1 if len(sorted_columns) > 1 else 0
                        
                        names_column = sorted_columns[names_column_idx][1]
                        signature_column = sorted_columns[-1][1] if len(sorted_columns) > names_column_idx else []
                        
                        names_column.sort(key=lambda cell: cell[1])
                        signature_column.sort(key=lambda cell: cell[1])
                        
                        # Skip headers
                        names_cells = names_column[1:] if len(names_column) > 0 else []
                        signature_cells = signature_column[1:] if len(signature_column) > 0 else []
                        
                        # Validate number of cells
                        if len(names_cells) != len(signature_cells):
                            error_message = f"Eltérés a nevek ({len(names_cells)}) és " \
                                           f"az aláírás cellák ({len(signature_cells)}) száma között. " \
                                           f"Minden hallgatóhoz pontosan egy aláírás cella szükséges."
                            print(f"ERROR: {error_message}")
                            
                            with open(debug_log_path, 'a') as f:
                                f.write(f"ERROR: {error_message}\n")
                                
                            return None, {
                                "message": error_message,
                                "names_count": len(names_cells),
                                "signatures_count": len(signature_cells)
                            }
                        
                        # Determine the number of rows we should have
                        num_rows = len(names_cells)  # Now we know both columns have the same length
                        print(f"Found {num_rows} rows (excluding header)")
                        
                        # Log to debug file
                        with open(debug_log_path, 'a') as f:
                            f.write(f"Found {num_rows} rows (excluding header)\n")
                            f.write("-" * 40 + "\n")
                        
                        results["students"] = []
                        student_names = {}
                        extracted_names = []
                        
                        # Fast name assignment - use the provided name list in order if available
                        if name_list and len(name_list) >= num_rows:
                            # Simply assign names in order from the list
                            print("Using provided name list in order for fast assignment")
                            with open(debug_log_path, 'a') as f:
                                f.write("Using provided name list in order for fast assignment\n")
                            
                            for i in range(num_rows):
                                student_names[i + 1] = name_list[i]
                                print(f"{i+1}. {name_list[i]} (assigned by order)")
                                with open(debug_log_path, 'a') as f:
                                    f.write(f"Row {i+1}: Assigned {name_list[i]} by order\n")
                        else:
                            # If no name list or not enough names, try OCR with basic matching
                            print("No complete name list provided, performing OCR detection")
                            with open(debug_log_path, 'a') as f:
                                f.write("No complete name list provided, performing OCR detection\n")
                            
                            matched_names = []
                            unmatched_extracted_names = []
                            name_list_copy = name_list.copy() if name_list else []
                            
                            # First pass: extract names and do initial matching
                            for i in range(num_rows):
                                extracted_name = "Unknown"
                                matched_name = None
                                
                                if i < len(names_cells):
                                    cell = names_cells[i]
                                    cell_x, cell_y, cell_w, cell_h = cell
                                    extracted_name = extract_text_from_cell(
                                        image,
                                        x + cell_x,
                                        y + cell_y,
                                        cell_w,
                                        cell_h
                                    )
                                    extracted_names.append(extracted_name)
                                    
                                    if extracted_name and name_list_copy:
                                        matched_name = find_matching_name(extracted_name, name_list_copy)
                                        if matched_name:
                                            print(f"{i+1}. {extracted_name} -> {matched_name}")
                                            matched_names.append(matched_name)
                                            name_list_copy.remove(matched_name)  # Remove the matched name
                                        else:
                                            print(f"{i+1}. {extracted_name} (no match found)")
                                            unmatched_extracted_names.append((i, extracted_name))
                                    elif extracted_name:
                                        print(f"{i+1}. {extracted_name}")
                                        unmatched_extracted_names.append((i, extracted_name))
                                
                                # At this point, either matched_name has a value, or we need to use the extracted name
                                if matched_name:
                                    student_names[i + 1] = matched_name
                                elif name_list_copy and i < len(name_list) - len(matched_names):
                                    # Assign the next available name from the list if we couldn't match
                                    next_name = name_list_copy[0]
                                    student_names[i + 1] = next_name
                                    name_list_copy.remove(next_name)
                                    print(f"{i+1}. Assigned next available name: {next_name}")
                                else:
                                    # As a last resort, use the extracted name or a generic placeholder
                                    student_names[i + 1] = extracted_name if extracted_name and extracted_name != "Unknown" else f"Student {i+1}"
                            
                            # If we still have unmatched cells and remaining names, assign in order
                            remaining_indices = [i+1 for i in range(num_rows) if i+1 not in student_names]
                            for idx, cell_idx in enumerate(remaining_indices):
                                if name_list_copy and idx < len(name_list_copy):
                                    student_names[cell_idx] = name_list_copy[idx]
                                    print(f"Assigned {name_list_copy[idx]} to row {cell_idx} based on order")
                                    with open(debug_log_path, 'a') as f:
                                        f.write(f"Assigned {name_list_copy[idx]} to row {cell_idx} based on order\n")
                        
                        # Second pass: detect signatures and create results
                        for i in range(num_rows):
                            name = student_names[i + 1]
                            
                            # Log name processing
                            with open(debug_log_path, 'a') as f:
                                f.write(f"Row {i+1}: Final name assignment: {name}\n")
                            
                            signature_info = {
                                "has_content": False,
                                "confidence": 0.9,
                                "pixel_density": 0.0
                            }
                            
                            if i < len(signature_cells):
                                cell = signature_cells[i]
                                cell_x, cell_y, cell_w, cell_h = cell
                                signature_info = check_for_signature(
                                    image,
                                    x + cell_x,
                                    y + cell_y,
                                    cell_w,
                                    cell_h
                                )
                                
                                with open(debug_log_path, 'a') as f:
                                    f.write(f"Row {i+1}: Signature detection: " + 
                                          f"has_signature={signature_info['has_content']}, " +
                                          f"confidence={signature_info['confidence']}, " +
                                          f"pixel_density={signature_info['pixel_density']}\n")
                                
                                highlight_signature(
                                    output_image,
                                    x + cell_x,
                                    y + cell_y,
                                    cell_w,
                                    cell_h,
                                    signature_info
                                )
                            
                            results["students"].append({
                                "row_number": i + 1,
                                "name": name,
                                "has_signed": signature_info["has_content"],
                                "confidence": signature_info["confidence"],
                                "pixel_density": signature_info["pixel_density"]
                            })
                        
                        with open(debug_log_path, 'a') as f:
                            f.write("-" * 40 + "\n")
                            f.write("Detection Summary:\n")
                            signed_count = sum(1 for student in results["students"] if student["has_signed"])
                            f.write(f"Total students: {len(results['students'])}\n")
                            f.write(f"Signed: {signed_count}\n")
                            f.write(f"Not signed: {len(results['students']) - signed_count}\n")
                            f.write("-" * 40 + "\n")
                        
                        for idx, (col_x, col_cells) in enumerate(sorted_columns, 1):
                            color = (255, 0, 0) if idx == names_column_idx + 1 else (0, 0, 255) if idx == len(sorted_columns) else (128, 128, 128)
                            highlight_column(col_cells, color, idx)

    except Exception as e:
        error_message = f"Hiba történt a kép feldolgozása során: {str(e)}"
        print(f"ERROR: {error_message}")
        
        with open(debug_log_path if 'debug_log_path' in locals() else os.path.join(DEBUG_FOLDER, 'error_log.txt'), 'a') as f:
            f.write(f"ERROR: {error_message}\n")
            import traceback
            f.write(traceback.format_exc())
            
        return None, {
            "message": error_message
        }

    output_path = os.path.join(DEBUG_FOLDER, 'output_table.jpg')
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved as: {output_path}")
    
    result_path = os.path.join(DEBUG_FOLDER, 'result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {result_path}")

    return output_image, results

def read_names_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return names
    except Exception as e:
        print(f"Error reading names file: {str(e)}")
        return None

def identify_names_column(sorted_columns, image, x, y, w, h):
    """Identify which column contains the names by analyzing text content and column characteristics."""
    best_column_idx = None
    best_score = -1
    
    for idx, (col_x, col_cells) in enumerate(sorted_columns):
        sorted_cells = sorted(col_cells, key=lambda cell: cell[1])
        
        if len(sorted_cells) < CONFIG['MIN_NAME_COLUMN_CELLS']:
            continue
            
        col_width = max(cell[0] + cell[2] for cell in sorted_cells) - min(cell[0] for cell in sorted_cells)
        
        if col_width < w * CONFIG['NAME_COLUMN_WIDTH_FACTOR']:
            continue
        
        text_cells = 0
        total_text_length = 0
        
        for cell in sorted_cells[1:]:
            cell_x, cell_y, cell_w, cell_h = cell
            text = extract_text_from_cell(image, x + cell_x, y + cell_y, cell_w, cell_h)
            if text and len(text) >= CONFIG['MIN_NAME_LENGTH']:
                text_cells += 1
                total_text_length += len(text)
        
        if text_cells > 0:
            score = (text_cells / len(sorted_cells)) * (total_text_length / (text_cells * CONFIG['MIN_NAME_LENGTH']))
            if score > best_score:
                best_score = score
                best_column_idx = idx
    
    return best_column_idx

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Detect tables and cells in an image')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--names_file', help='Path to text file containing names (one per line)')
    args = parser.parse_args()

    try:
        name_list = None
        if args.names_file:
            name_list = read_names_from_file(args.names_file)
            if name_list:
                print(f"Loaded {len(name_list)} names from file")
            else:
                print("No names loaded from file, proceeding without name matching")
        
        result, _ = detect_table_and_cells(args.image_path, name_list)
        print("Table and cell detection completed successfully!")
    except Exception as e:
        print(f"Hiba történt: {str(e)}") 