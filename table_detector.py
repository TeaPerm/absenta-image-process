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
    'MIN_SIGNATURE_DENSITY': 0.005, # Minimum density of white pixels (0.5%)
    'MAX_SIGNATURE_DENSITY': 0.4,  # Maximum density to avoid detecting filled regions (40%)
    'WHITE_THRESHOLD': 160,       # Threshold for white pixels (0-255)
    
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
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_matching_name(table_name, name_list, threshold=CONFIG['NAME_MATCH_THRESHOLD']):
    best_match = None
    best_ratio = 0
    
    table_name = clean_extracted_name(table_name)
    
    for name in name_list:
        clean_name = clean_extracted_name(name)
        ratio = string_similarity(table_name, clean_name)
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
    
    border_mask = np.ones_like(gray)
    border_width = CONFIG['BORDER_WIDTH']
    border_mask[0:border_width, :] = 0
    border_mask[-border_width:, :] = 0
    border_mask[:, 0:border_width] = 0
    border_mask[:, -border_width:] = 0
    
    gray = cv2.bitwise_not(gray)
    
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C constant
    )
    
    binary = binary * border_mask
    
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_components = []
    total_signature_area = 0
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < CONFIG['MIN_SIGNATURE_AREA']:
            continue
            
        x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
        
        if x_c <= border_width or y_c <= border_width or \
           x_c + w_c >= w - border_width or y_c + h_c >= h - border_width:
            continue
        
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
        
        white_mask = (gray > CONFIG['WHITE_THRESHOLD']) & (border_mask > 0)
        white_pixels = np.count_nonzero(white_mask)
        total_pixels = np.count_nonzero(border_mask)
        pixel_density = white_pixels / total_pixels if total_pixels > 0 else 0
        
        has_signature = (
            len(valid_components) >= 1 and
            signature_width >= CONFIG['MIN_SIGNATURE_WIDTH'] and
            signature_height >= CONFIG['MIN_SIGNATURE_HEIGHT'] and
            white_pixels >= CONFIG['MIN_WHITE_PIXELS'] and
            CONFIG['MIN_SIGNATURE_DENSITY'] <= pixel_density <= CONFIG['MAX_SIGNATURE_DENSITY']
        )
        
        confidence = min(0.9, (
            (signature_width / w) * 0.3 +  # Width contribution
            (signature_height / h) * 0.2 +  # Height contribution
            (len(valid_components) / 3) * 0.2 +  # Number of components
            (pixel_density / CONFIG['MIN_SIGNATURE_DENSITY']) * 0.3  # Density contribution
        ))
    else:
        has_signature = False
        confidence = 0.9  # High confidence that it's empty
        pixel_density = 0
        white_pixels = 0
        signature_width = 0
        signature_height = 0
    
    return {
        "has_content": has_signature,
        "confidence": round(confidence, 2),
        "pixel_density": round(pixel_density, 4),
        "debug_info": {
            "num_components": len(valid_components),
            "white_pixels": white_pixels,
            "signature_width": signature_width,
            "signature_height": signature_height,
            "bounds": (min_x, min_y, max_x, max_y) if valid_components else None,
            "valid_components": valid_components
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
            text = f"Signed ({confidence:.2f})"
            text_x = x + 5
            text_y = y + h - 5
            
            cv2.putText(image, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)  # Black outline
            cv2.putText(image, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # Green text

def clean_extracted_name(text):
    text = text.replace('|', '').replace('\\', '').replace('/', '')
    text = text.replace('_', '').replace('=', '').replace('+', '')
    
    text = ''.join(c for c in text if not c.isdigit())
    
    text = ''.join(c for c in text if c.isalpha() or c.isspace() or c in '.-')
    
    text = ' '.join(text.split())
    text = text.strip('.-_ ')
    
    text = ' '.join(word for word in text.split() if len(word) > 1)
    
    return text

def extract_text_from_cell(image, x, y, w, h):
    cell_region = image[y:y+h, x:x+w]
    
    cell_pil = Image.fromarray(cv2.cvtColor(cell_region, cv2.COLOR_BGR2RGB))
    
    text = pytesseract.image_to_string(cell_pil, config='--psm 6').strip()
    return clean_extracted_name(text)

def detect_table_and_cells(image_path, name_list=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image. Please check the file path.")

    output_image = image.copy()

    thresh = preprocess_image(image)
    
    height, width = thresh.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    cv2.imwrite(os.path.join(DEBUG_FOLDER, 'preprocessed.jpg'), thresh)
    print(f"Saved preprocessed image as {os.path.join(DEBUG_FOLDER, 'preprocessed.jpg')}")
    
    results = {
        "students": []  # Changed to list for cleaner output
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

    # Combine horizontal and vertical lines
    table_mask = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
    
    kernel = np.ones((3,3), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(os.path.join(DEBUG_FOLDER, 'table_mask.jpg'), table_mask)
    print(f"Saved table mask as {os.path.join(DEBUG_FOLDER, 'table_mask.jpg')}")
    
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} potential tables")

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
                    
                    for cell in sorted_cells[1:]:
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
                    else:
                        print("Could not identify names column, using default second column")
                        names_column_idx = 1 if len(sorted_columns) > 1 else 0
                    
                    names_column = sorted_columns[names_column_idx][1]
                    signature_column = sorted_columns[-1][1] if len(sorted_columns) > names_column_idx else []
                    
                    names_column.sort(key=lambda cell: cell[1])
                    signature_column.sort(key=lambda cell: cell[1])
                    
                    # Skip headers
                    names_cells = names_column[1:] if len(names_column) > 0 else []
                    signature_cells = signature_column[1:] if len(signature_column) > 0 else []
                    
                    if len(names_cells) != len(signature_cells):
                        raise ValueError(
                            f"Mismatch between number of names ({len(names_cells)}) and "
                            f"signature cells ({len(signature_cells)}). "
                            f"Each student must have exactly one signature cell."
                        )
                    
                    num_rows = len(names_cells)  # Now we know both columns have the same length
                    print(f"Found {num_rows} rows (excluding header)")
                    
                    results["students"] = []
                    student_names = {}
                    
                    for i in range(num_rows):
                        name = "Unknown"
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
                            if extracted_name:
                                if name_list:
                                    matched_name = find_matching_name(extracted_name, name_list)
                                    if matched_name:
                                        name = matched_name
                                        print(f"{i+1}. {extracted_name} -> {matched_name}")
                                    else:
                                        name = extracted_name
                                        print(f"{i+1}. {extracted_name} (no match found)")
                                else:
                                    name = extracted_name
                                    print(f"{i+1}. {extracted_name}")
                        
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
                        
                        results["students"].append({
                            "row_number": i + 1,
                            "name": name,
                            "has_signed": signature_info["has_content"],
                            "confidence": signature_info["confidence"],
                            "pixel_density": signature_info["pixel_density"]
                        })
                        
                        student_names[i + 1] = name
                    
                    for idx, (col_x, col_cells) in enumerate(sorted_columns, 1):
                        color = (255, 0, 0) if idx == names_column_idx + 1 else (0, 0, 255) if idx == len(sorted_columns) else (128, 128, 128)
                        highlight_column(col_cells, color, idx)
                        
                        if idx == len(sorted_columns):
                            for cell in col_cells[1:]:  # Skip header
                                cell_x, cell_y, cell_w, cell_h = cell
                                signature_info = check_for_signature(
                                    image,
                                    x + cell_x,
                                    y + cell_y,
                                    cell_w,
                                    cell_h
                                )
                                if signature_info["has_content"]:
                                    highlight_signature(
                                        output_image,
                                        x + cell_x,
                                        y + cell_y,
                                        cell_w,
                                        cell_h,
                                        signature_info
                                    )

    output_path = os.path.join(DEBUG_FOLDER, 'output_table.jpg')
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved as: {output_path}")
    
    # Save results to JSON file
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved to result.json")

    return output_image

def read_names_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read lines and remove empty lines and whitespace
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
        
        for cell in sorted_cells[1:]:  # Skip header
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
        # Read names from file if provided
        name_list = None
        if args.names_file:
            name_list = read_names_from_file(args.names_file)
            if name_list:
                print(f"Loaded {len(name_list)} names from file")
            else:
                print("No names loaded from file, proceeding without name matching")
        
        result = detect_table_and_cells(args.image_path, name_list)
        print("Table and cell detection completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 