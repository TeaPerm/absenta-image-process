import cv2
import numpy as np
from PIL import Image
import argparse
import os
import json
import pytesseract
from difflib import SequenceMatcher

# Set Tesseract executable path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create debug folder if it doesn't exist
DEBUG_FOLDER = 'debug'
if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

# Configuration constants
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
    'INK_CONTENT_THRESHOLD': 0.005,  # Threshold for determining if cell has ink content
    'EMPTY_CELL_THRESHOLD': 0.002,   # Threshold for determining if cell is empty
    'HIGH_INK_THRESHOLD': 0.02,      # Threshold for high ink content
    
    # Table detection
    'HORIZONTAL_KERNEL_SIZES': [20, 30, 40],  # Kernel sizes for horizontal line detection
    'VERTICAL_KERNEL_SIZES': [20, 30, 40],    # Kernel sizes for vertical line detection
    'TABLE_MIN_WIDTH_FACTOR': 0.05,    # Minimum table width as factor of image width
    'TABLE_MIN_HEIGHT_FACTOR': 0.05,   # Minimum table height as factor of image height
    'TABLE_EDGE_MARGIN': 5,            # Minimum distance from table to image edge
    
    # Cell detection
    'MIN_CELL_AREA_FACTOR': 0.005,     # Minimum cell area as percentage of table area
    'MIN_CELL_WIDTH': 5,               # Minimum cell width in pixels
    'MIN_CELL_HEIGHT': 5,              # Minimum cell height in pixels
    'COLUMN_X_TOLERANCE_FACTOR': 0.01, # X-coordinate tolerance for column grouping as factor of table width
    
    # Visualization
    'OVERLAY_ALPHA': 0.2,              # Alpha transparency for cell highlighting
    'TEXT_SCALE': 0.7,                 # Scale for column number text
    'TEXT_THICKNESS': 2,               # Thickness for text outline
    'SIGNATURE_TEXT_SCALE': 0.5,       # Scale for signature status text
}

def string_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_matching_name(table_name, name_list, threshold=CONFIG['NAME_MATCH_THRESHOLD']):
    best_match = None
    best_ratio = 0
    
    # Clean the table name
    table_name = clean_extracted_name(table_name)
    
    for name in name_list:
        # Clean the name from the list
        clean_name = clean_extracted_name(name)
        ratio = string_similarity(table_name, clean_name)
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = name  # Return the original name from the list
    
    return best_match

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while preserving edges
    gray = cv2.bilateralFilter(
        gray, 
        CONFIG['BILATERAL_FILTER_D'], 
        CONFIG['BILATERAL_FILTER_SIGMA_COLOR'], 
        CONFIG['BILATERAL_FILTER_SIGMA_SPACE']
    )
    
    # Apply adaptive thresholding with more aggressive parameters
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        CONFIG['ADAPTIVE_THRESH_BLOCK_SIZE'], 
        CONFIG['ADAPTIVE_THRESH_C']
    )
    
    # Remove small noise
    kernel = np.ones(CONFIG['MORPH_KERNEL_SIZE'], np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def check_for_signature(image, x, y, w, h):
    # Extract the cell region from the original image
    cell_region = image[y:y+h, x:x+w]
    
    # Create a mask for blue/black colors
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(cell_region, cv2.COLOR_BGR2HSV)
    
    # Blue mask (pen ink)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Dark mask (black ink)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combine blue and black masks
    ink_mask = cv2.bitwise_or(blue_mask, black_mask)
    
    # Create a mask to ignore the cell borders
    border_mask = np.ones_like(ink_mask)
    border_width = CONFIG['BORDER_WIDTH']
    border_mask[0:border_width, :] = 0  # Top border
    border_mask[-border_width:, :] = 0  # Bottom border
    border_mask[:, 0:border_width] = 0  # Left border
    border_mask[:, -border_width:] = 0  # Right border
    
    # Apply the border mask
    ink_mask = ink_mask * border_mask
    
    # Calculate the percentage of ink pixels
    ink_pixels = np.count_nonzero(ink_mask)
    total_pixels = np.count_nonzero(border_mask)
    pixel_density = ink_pixels / total_pixels if total_pixels > 0 else 0
    
    # Get thresholds from config
    content_threshold = CONFIG['INK_CONTENT_THRESHOLD']
    empty_threshold = CONFIG['EMPTY_CELL_THRESHOLD']
    high_ink_threshold = CONFIG['HIGH_INK_THRESHOLD']
    
    # Calculate confidence level based on pixel density
    if pixel_density < empty_threshold:  # Very few ink pixels
        confidence = 0.9  # Very confident it's empty
    elif pixel_density > high_ink_threshold:  # Lots of ink pixels
        confidence = 0.9  # Very confident it has content
    else:
        # For densities in between, confidence scales with distance from thresholds
        confidence = 0.5 + min(
            abs(pixel_density - empty_threshold) / (high_ink_threshold - empty_threshold),
            abs(pixel_density - high_ink_threshold) / (high_ink_threshold - empty_threshold)
        ) * 0.4
    
    # Consider it signed if more than threshold of pixels are ink colored
    return {
        "has_content": pixel_density > content_threshold,
        "confidence": round(confidence, 2),
        "pixel_density": round(pixel_density, 4)
    }

def clean_extracted_name(text):
    # More aggressive cleaning
    # Remove common OCR artifacts and unwanted patterns
    text = text.replace('|', '').replace('\\', '').replace('/', '')
    text = text.replace('_', '').replace('=', '').replace('+', '')
    
    # Remove any numbers
    text = ''.join(c for c in text if not c.isdigit())
    
    # Remove any symbols except letters, spaces, dots, and hyphens
    text = ''.join(c for c in text if c.isalpha() or c.isspace() or c in '.-')
    
    # Clean up multiple spaces, dots, and hyphens
    text = ' '.join(text.split())
    text = text.strip('.-_ ')
    
    # Remove very short segments (likely noise)
    text = ' '.join(word for word in text.split() if len(word) > 1)
    
    return text

def extract_text_from_cell(image, x, y, w, h):
    # Extract the cell region
    cell_region = image[y:y+h, x:x+w]
    
    # Convert to PIL Image for better OCR
    cell_pil = Image.fromarray(cv2.cvtColor(cell_region, cv2.COLOR_BGR2RGB))
    
    # Use tesseract to extract text 
    text = pytesseract.image_to_string(cell_pil, config='--psm 6').strip()
    return clean_extracted_name(text)

def detect_table_and_cells(image_path, name_list=None):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image. Please check the file path.")

    # Create a copy of the image for drawing
    output_image = image.copy()

    # Preprocess the image
    thresh = preprocess_image(image)
    
    # Get image dimensions
    height, width = thresh.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Save preprocessed image for debugging
    cv2.imwrite(os.path.join(DEBUG_FOLDER, 'preprocessed.jpg'), thresh)
    print(f"Saved preprocessed image as {os.path.join(DEBUG_FOLDER, 'preprocessed.jpg')}")
    
    # Initialize results dictionary
    results = {
        "students": []  # Changed to list for cleaner output
    }
    
    # Detect horizontal lines with varying kernel sizes
    horizontal_lines = np.zeros_like(thresh)
    for kernel_size in CONFIG['HORIZONTAL_KERNEL_SIZES']:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        horizontal_lines = cv2.bitwise_or(horizontal_lines, lines)

    # Detect vertical lines with varying kernel sizes
    vertical_lines = np.zeros_like(thresh)
    for kernel_size in CONFIG['VERTICAL_KERNEL_SIZES']:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        vertical_lines = cv2.bitwise_or(vertical_lines, lines)

    # Combine horizontal and vertical lines
    table_mask = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
    
    # Save table mask for debugging
    cv2.imwrite(os.path.join(DEBUG_FOLDER, 'table_mask.jpg'), table_mask)
    print(f"Saved table mask as {os.path.join(DEBUG_FOLDER, 'table_mask.jpg')}")
    
    # Find contours
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} potential tables")

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter and draw contours
    for contour in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get table detection parameters from config
        min_width = width * CONFIG['TABLE_MIN_WIDTH_FACTOR']
        min_height = height * CONFIG['TABLE_MIN_HEIGHT_FACTOR']
        edge_margin = CONFIG['TABLE_EDGE_MARGIN']
        
        # Filter tables based on size and position
        if (w > min_width and h > min_height and
            x > edge_margin and y > edge_margin and
            x + w < width - edge_margin and y + h < height - edge_margin):
            
            print(f"Processing table at position ({x}, {y}) with size {w}x{h}")
            
            # Draw rectangle around the table
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract the table region from the original threshold image
            table_region = thresh[y:y+h, x:x+w]
            
            # Save table region for debugging
            cv2.imwrite(os.path.join(DEBUG_FOLDER, 'table_region.jpg'), table_region)
            print(f"Saved table region as {os.path.join(DEBUG_FOLDER, 'table_region.jpg')}")
            
            # Extract the table mask region
            table_mask_region = table_mask[y:y+h, x:x+w]
            
            # Find cell contours directly from the table mask region
            cell_contours, hierarchy = cv2.findContours(
                table_mask_region, 
                cv2.RETR_CCOMP,  # Changed to RETR_CCOMP to get both external and internal contours
                cv2.CHAIN_APPROX_SIMPLE
            )
            print(f"Found {len(cell_contours)} potential cells")
            
            # Find all valid cells
            valid_cells = []
            min_cell_area = (w * h) * CONFIG['MIN_CELL_AREA_FACTOR']
            min_cell_width = CONFIG['MIN_CELL_WIDTH']
            min_cell_height = CONFIG['MIN_CELL_HEIGHT']
            
            for cell_contour in cell_contours:
                area = cv2.contourArea(cell_contour)
                if area < min_cell_area:
                    continue
                    
                cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cell_contour)
                
                # Basic size and position checks
                if (cell_w > min_cell_width and cell_h > min_cell_height and
                    cell_x >= 0 and cell_y >= 0 and 
                    cell_x + cell_w <= w and cell_y + cell_h <= h):
                    valid_cells.append((cell_x, cell_y, cell_w, cell_h))
            
            print(f"Found {len(valid_cells)} valid cells")
            
            if len(valid_cells) > 0:
                # Sort cells by x-coordinate
                valid_cells.sort(key=lambda cell: cell[0])
                
                # Find the leftmost and rightmost x-coordinates
                min_x = min(cell[0] for cell in valid_cells)
                max_x = max(cell[0] + cell[2] for cell in valid_cells)
                table_width = max_x - min_x if max_x > min_x else w
                
                # Group cells by x-coordinate to identify columns
                columns = {}
                tolerance = table_width * CONFIG['COLUMN_X_TOLERANCE_FACTOR']
                
                for cell in valid_cells:
                    cell_x, cell_y, cell_w, cell_h = cell
                    # Find the closest column
                    assigned = False
                    for col_x in columns.keys():
                        if abs(cell_x - col_x) < tolerance:
                            columns[col_x].append(cell)
                            assigned = True
                            break
                    if not assigned:
                        columns[cell_x] = [cell]
                
                # Convert to list and sort by x-coordinate
                sorted_columns = sorted(columns.items(), key=lambda x: x[0])
                print(f"Found {len(sorted_columns)} columns")
                
                # Draw all cells in light yellow
                for cell in valid_cells:
                    cell_x, cell_y, cell_w, cell_h = cell
                    cv2.rectangle(output_image, 
                                (x + cell_x, y + cell_y), 
                                (x + cell_x + cell_w, y + cell_y + cell_h), 
                                (102, 255, 255), 1)  # Light yellow in BGR format
                
                # Function to highlight a column with a specific color
                def highlight_column(column_cells, color, column_number):
                    # Sort cells by y-coordinate (top to bottom)
                    sorted_cells = sorted(column_cells, key=lambda cell: cell[1])
                    
                    # Write column number at the top of the column
                    if sorted_cells:
                        cell_x, cell_y, cell_w, cell_h = sorted_cells[0]  # Get first cell position
                        text_x = x + cell_x + cell_w // 2 - 10  # Center the number
                        text_y = y + cell_y - 10  # Place above the column
                        # Add black outline for better visibility
                        cv2.putText(output_image, str(column_number), (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, CONFIG['TEXT_SCALE'], (0, 0, 0), CONFIG['TEXT_THICKNESS'])
                        # Add white text
                        cv2.putText(output_image, str(column_number), (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, CONFIG['TEXT_SCALE'], (255, 255, 255), 1)
                    
                    # Rest of the highlight_column function remains the same
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

                        # For the signature column (last column), add text annotation
                        if column_number == len(sorted_columns):  # If it's the last column
                            signature_info = check_for_signature(
                                image,
                                x + cell_x,
                                y + cell_y,
                                cell_w,
                                cell_h
                            )
                            text = "Signed" if signature_info["has_content"] else "Empty"
                            text_color = (0, 255, 0) if signature_info["has_content"] else (0, 0, 255)
                            text_x = x + cell_x + cell_w + 5
                            text_y = y + cell_y + cell_h // 2
                            
                            cv2.putText(output_image, text, (text_x, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, CONFIG['SIGNATURE_TEXT_SCALE'], (0, 0, 0), 2)
                            cv2.putText(output_image, text, (text_x, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, CONFIG['SIGNATURE_TEXT_SCALE'], text_color, 1)

                # Modify the column detection logic
                if len(sorted_columns) > 0:
                    # Process all columns
                    for idx, (col_x, col_cells) in enumerate(sorted_columns, 1):
                        color = (255, 0, 0) if idx == 1 else (0, 0, 255) if idx == len(sorted_columns) else (128, 128, 128)  # Blue for names, Red for signatures, Gray for others
                        highlight_column(col_cells, color, idx)
                
                # Highlight the second column if it exists
                if len(sorted_columns) > 1:
                    second_column = sorted_columns[1][1]
                    print(f"Highlighting {len(second_column)-1} cells in the second column (excluding header)")
                    highlight_column(second_column, (255, 0, 0), 2)  # Blue for second column
                    
                    # Sort cells by y-coordinate (top to bottom)
                    sorted_cells = sorted(second_column, key=lambda cell: cell[1])
                    
                    # Extract student names (skip header)
                    print("\nStudent Names:")
                    print("-" * 20)
                    # Store names for later use
                    student_names = {}
                    for i, cell in enumerate(sorted_cells[1:], 1):
                        cell_x, cell_y, cell_w, cell_h = cell
                        name = extract_text_from_cell(
                            image,
                            x + cell_x,
                            y + cell_y,
                            cell_w,
                            cell_h
                        )
                        if name:  # Only store if we found text
                            if name_list:
                                # Try to match with provided name list
                                matched_name = find_matching_name(name, name_list)
                                if matched_name:
                                    student_names[i] = matched_name
                                    print(f"{i}. {name} -> {matched_name}")  # Show the replacement
                                else:
                                    student_names[i] = name
                                    print(f"{i}. {name} (no match found)")
                            else:
                                student_names[i] = name
                                print(f"{i}. {name}")
                    print("-" * 20)
                
                # Process and highlight the last column if it exists
                if len(sorted_columns) > 0:
                    last_column = sorted_columns[-1][1]
                    print(f"Highlighting {len(last_column)-1} cells in the last column (excluding header)")
                    highlight_column(last_column, (0, 0, 255), len(sorted_columns))  # Red for last column
                    
                    # Sort cells in last column by y-coordinate (top to bottom)
                    last_column.sort(key=lambda cell: cell[1])
                    
                    # Skip the header row and check each cell in the last column for signatures
                    for i, cell in enumerate(last_column[1:], start=1):  # Start from second cell
                        cell_x, cell_y, cell_w, cell_h = cell
                        signature_info = check_for_signature(
                            image,
                            x + cell_x,
                            y + cell_y,
                            cell_w,
                            cell_h
                        )
                        
                        # Add result to the dictionary
                        results["students"].append({
                            "row_number": i,  # Add row number to results
                            "name": student_names.get(i, "Unknown"),
                            "has_signed": signature_info["has_content"],
                            "confidence": signature_info["confidence"],
                            "pixel_density": signature_info["pixel_density"]
                        })
                        
                        # Print debug information
                        print(f"Row {i}: {'Signed' if signature_info['has_content'] else 'Empty'} "
                              f"(Confidence: {signature_info['confidence']:.2f}, "
                              f"Density: {signature_info['pixel_density']:.4f})")

    # Save to a fixed output file
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

if __name__ == "__main__":
    # Set up argument parser
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