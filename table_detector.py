import cv2
import numpy as np
from PIL import Image
import argparse
import os
import json
import pytesseract

# Set Tesseract executable path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while preserving edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding with more aggressive parameters
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 5  # Increased block size and constant
    )
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
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
    
    # Create a mask to ignore the cell borders (5 pixels from each edge)
    border_mask = np.ones_like(ink_mask)
    border_width = 5
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
    
    # Lower threshold for ink detection since we're more specific now
    content_threshold = 0.005  # 0.5% of pixels need to be ink colored
    
    # Calculate confidence level based on pixel density
    if pixel_density < 0.002:  # Very few ink pixels
        confidence = 0.9  # Very confident it's empty
    elif pixel_density > 0.02:  # Lots of ink pixels
        confidence = 0.9  # Very confident it has content
    else:
        # For densities in between, confidence scales with distance from thresholds
        confidence = 0.5 + min(
            abs(pixel_density - 0.002) / 0.018,  # Distance from empty threshold
            abs(pixel_density - 0.02) / 0.018    # Distance from content threshold
        ) * 0.4
    
    # Consider it signed if more than threshold of pixels are ink colored
    return {
        "has_content": pixel_density > content_threshold,
        "confidence": round(confidence, 2),
        "pixel_density": round(pixel_density, 4)
    }

def extract_text_from_cell(image, x, y, w, h):
    # Extract the cell region
    cell_region = image[y:y+h, x:x+w]
    
    # Convert to PIL Image for better OCR
    cell_pil = Image.fromarray(cv2.cvtColor(cell_region, cv2.COLOR_BGR2RGB))
    
    # Use tesseract to extract text
    text = pytesseract.image_to_string(cell_pil, config='--psm 6').strip()
    return text

def detect_table_and_cells(image_path):
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
    cv2.imwrite('debug_preprocessed.jpg', thresh)
    print("Saved preprocessed image as debug_preprocessed.jpg")
    
    # Initialize results dictionary
    results = {
        "rows": {}  # Changed to dictionary for better name association
    }
    
    # Detect horizontal lines with varying kernel sizes
    horizontal_lines = np.zeros_like(thresh)
    for kernel_size in [20, 30, 40]:  # Reduced kernel sizes for finer detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        horizontal_lines = cv2.bitwise_or(horizontal_lines, lines)

    # Detect vertical lines with varying kernel sizes
    vertical_lines = np.zeros_like(thresh)
    for kernel_size in [20, 30, 40]:  # Reduced kernel sizes for finer detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        vertical_lines = cv2.bitwise_or(vertical_lines, lines)

    # Combine horizontal and vertical lines
    table_mask = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
    
    # Save table mask for debugging
    cv2.imwrite('debug_table_mask.jpg', table_mask)
    print("Saved table mask as debug_table_mask.jpg")
    
    # Find contours
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} potential tables")

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filter and draw contours
    for contour in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # More lenient filtering for table detection
        if (w > width * 0.05 and h > height * 0.05 and  # Reduced minimum size requirements
            x > 5 and y > 5 and  # Reduced edge distance requirements
            x + w < width - 5 and y + h < height - 5):
            
            print(f"Processing table at position ({x}, {y}) with size {w}x{h}")
            
            # Draw rectangle around the table
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract the table region from the original threshold image
            table_region = thresh[y:y+h, x:x+w]
            
            # Save table region for debugging
            cv2.imwrite('debug_table_region.jpg', table_region)
            print("Saved table region as debug_table_region.jpg")
            
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
            min_cell_area = (w * h) * 0.001  # Minimum cell area as 0.1% of table area
            
            for cell_contour in cell_contours:
                area = cv2.contourArea(cell_contour)
                if area < min_cell_area:
                    continue
                    
                cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cell_contour)
                
                # Basic size and position checks
                if (cell_w > 5 and cell_h > 5 and  # Minimum size in pixels
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
                tolerance = table_width * 0.01  # 1% of table width
                
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
                
                # Draw all cells in light gray
                for cell in valid_cells:
                    cell_x, cell_y, cell_w, cell_h = cell
                    cv2.rectangle(output_image, 
                                (x + cell_x, y + cell_y), 
                                (x + cell_x + cell_w, y + cell_y + cell_h), 
                                (200, 200, 200), 1)
                
                # Function to highlight a column with a specific color
                def highlight_column(column_cells, color):
                    # Sort cells by y-coordinate (top to bottom)
                    sorted_cells = sorted(column_cells, key=lambda cell: cell[1])
                    # Skip the header (first row) when highlighting
                    for cell in sorted_cells[1:]:
                        cell_x, cell_y, cell_w, cell_h = cell
                        # Draw colored rectangle
                        cv2.rectangle(output_image, 
                                    (x + cell_x, y + cell_y), 
                                    (x + cell_x + cell_w, y + cell_y + cell_h), 
                                    color, 2)
                        
                        # Add semi-transparent overlay
                        overlay = output_image.copy()
                        cv2.rectangle(overlay, 
                                    (x + cell_x, y + cell_y), 
                                    (x + cell_x + cell_w, y + cell_y + cell_h), 
                                    color, -1)
                        cv2.addWeighted(overlay, 0.2, output_image, 0.8, 0, output_image)
                
                # Highlight the second column if it exists
                if len(sorted_columns) > 1:
                    second_column = sorted_columns[1][1]
                    print(f"Highlighting {len(second_column)-1} cells in the second column (excluding header)")
                    highlight_column(second_column, (255, 0, 0))  # Blue for second column
                    
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
                            student_names[i] = name.replace('\n', ' ').strip()
                        print(f"{i}. {student_names[i]}")
                    print("-" * 20)
                
                # Process and highlight the last column if it exists
                if len(sorted_columns) > 0:
                    last_column = sorted_columns[-1][1]
                    print(f"Highlighting {len(last_column)-1} cells in the last column (excluding header)")
                    highlight_column(last_column, (0, 0, 255))  # Red for last column
                    
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
                        results["rows"][str(i)] = {
                            "student_name": student_names.get(i, "Unknown"),
                            "row_number": i,
                            "has_signature": signature_info["has_content"],
                            "confidence": signature_info["confidence"],
                            "pixel_density": signature_info["pixel_density"]
                        }
                        
                        # Print debug information
                        print(f"Row {i}: {'Signed' if signature_info['has_content'] else 'Empty'} "
                              f"(Confidence: {signature_info['confidence']:.2f}, "
                              f"Density: {signature_info['pixel_density']:.4f})")

    # Save to a fixed output file
    output_path = 'output_table.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved as: {output_path}")
    
    # Save results to JSON file
    with open('result.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to result.json")

    return output_image

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect tables and cells in an image')
    parser.add_argument('image_path', help='Path to the input image')
    args = parser.parse_args()

    try:
        result = detect_table_and_cells(args.image_path)
        print("Table and cell detection completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 