import os
import cv2
import numpy as np
import torch
import sys
from torchvision import transforms

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractor import extract_for_phase4
from model import DigitRecognizerCNN
from solver import SudokuSolver

def load_model(model_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = DigitRecognizerCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def is_cell_empty_and_get_thresh(cell_img):
    blurred = cv2.GaussianBlur(cell_img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 15)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
            
def center_digit_mnist_style(mask):
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return mask
    digit = mask[y:y+h, x:x+w]
    size = 20
    if w > h:
        new_w = size
        new_h = max(1, int(np.round(size * h / w)))
    else:
        new_h = size
        new_w = max(1, int(np.round(size * w / h)))
    digit_scaled = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    result = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    result[start_y:start_y+new_h, start_x:start_x+new_w] = digit_scaled
    return result

def recognize_digits(cells, model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    grid = np.zeros((9, 9), dtype=int)
    global_mean = np.mean([cell["img"] for cell in cells])
    for i in range(81):
        row = i // 9
        col = i % 9
        
        cell_img = cells[i]["img"]
        cell_mean = np.mean(cell_img)
        cell_h, cell_w = cell_img.shape
        cell = cv2.GaussianBlur(cell_img, (5, 5), 0)
        
        # Adaptive thresholding with fallback for faint digits
        if global_mean > 220:
            c_vals_to_try = [2]
        elif global_mean < 170:
            # Low contrast / dark background
            if cell_mean < 130:
                c_vals_to_try = [5, 2, 0]
            else:
                c_vals_to_try = [15, 8, 5, 2, 0]
        else:
            # Normal contrast images
            if cell_mean < 130:
                c_vals_to_try = [5]
            else:
                c_vals_to_try = [15, 10, 5]
        
        valid_contour = None
        thresh_to_use = None
        
        for C_val in c_vals_to_try:
            thresh_cell = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C_val)
            contours, _ = cv2.findContours(thresh_cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            max_area = 0
            temp_valid_contour = None
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                
                # Digit size constraints (relative to cell size)
                if w > 5 and h > 10 and area > cell_w * cell_h * 0.025 and area < cell_w * cell_h * 0.4:
                    if area > max_area:
                        max_area = area
                        temp_valid_contour = c
                        
            if temp_valid_contour is not None:
                valid_contour = temp_valid_contour
                thresh_to_use = thresh_cell
                break
                
        if valid_contour is None:
            grid[row][col] = 0
            continue
            
        # Create mask from contour
        mask = np.zeros(cell_img.shape, dtype=np.uint8)
        cv2.drawContours(mask, [valid_contour], -1, 255, thickness=cv2.FILLED)
        
        # Center the digit using MNIST style
        processed_mask = center_digit_mnist_style(mask)
        
        cell_tensor = transform(processed_mask).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(cell_tensor)
            _, predicted = torch.max(output.data, 1)
            pred_digit = predicted.item()
            
            grid[row][col] = pred_digit
    return grid

def overlay_solution(original_img, warped_img, solved_grid, original_grid, M, side, color=(0, 255, 0)):
    """
    Overlays the solved numbers back onto the original image.
    Only fills empty cells, leaves pre-filled cells untouched.
    """
    # Create a blank image of the same size as warped_img to draw the numbers
    blank_warped = np.zeros_like(warped_img)
    
    cell_side = side // 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate an appropriate font scale based on cell size
    font_scale = cell_side / 30.0  
    thickness = max(2, int(font_scale * 2))
    
    for row in range(9):
        for col in range(9):
            # Only draw the number if it was originally empty (0)
            if original_grid[row][col] == 0 and solved_grid[row][col] != 0:
                digit = str(solved_grid[row][col])
                
                # Calculate text size to center it
                text_size = cv2.getTextSize(digit, font, font_scale, thickness)[0]
                text_x = col * cell_side + (cell_side - text_size[0]) // 2
                text_y = row * cell_side + (cell_side + text_size[1]) // 2
                
                # Draw text
                cv2.putText(blank_warped, digit, (text_x, text_y), font, font_scale, color, thickness)

    # Calculate inverse perspective transform
    Minv = np.linalg.inv(M)
    
    # Warp the blank image back to the original perspective
    h, w = original_img.shape[:2]
    unwarped_text = cv2.warpPerspective(blank_warped, Minv, (w, h))
    
    # Overlay the text on the original image
    gray_unwarped = cv2.cvtColor(unwarped_text, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_unwarped, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    img_bg = cv2.bitwise_and(original_img, original_img, mask=mask_inv)
    text_fg = cv2.bitwise_and(unwarped_text, unwarped_text, mask=mask)
    
    final_img = cv2.add(img_bg, text_fg)
    return final_img

def solve_sudoku_image(image_path, model_path, output_path=None, debug_dir=None):
    print(f"--- Processing {os.path.basename(image_path)} ---")
    
    # 1. Extract
    extracted_data = extract_for_phase4(image_path, debug_dir)
    if not extracted_data:
        print(f"Failed to extract sudoku from {image_path}")
        return False, None, None
        
    cells = extracted_data["processed_cells"]
    
    # 2. Recognize
    model, device = load_model(model_path)
    grid = recognize_digits(cells, model, device)
    original_grid = grid.copy()
    
    print("Recognized Puzzle:")
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            val = grid[r][c]
            row_str += str(val) + " " if val != 0 else ". "
        print(row_str)
        
    # 3. Solve
    solver = SudokuSolver(grid)
    solved_grid = None
    color = (0, 255, 0)
    
    success = solver.solve()
    if not success:
        print("Not Solved.")
        solved_grid = original_grid
        color = (0, 0, 255)
    else:
        solved_grid = solver.grid
        print("Solved Successfully!")
    
    # 4. Overlay
    final_img = overlay_solution(
        extracted_data["original_img"], 
        extracted_data["warped_img"],
        solved_grid, 
        original_grid,
        extracted_data["M"],
        extracted_data["side"],
        color=color
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final_img)
        
    return success, original_grid, solved_grid, extracted_data["warped_img"], final_img

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <image_path> <model_path> [output_path]")
        sys.exit(1)
        
    solve_sudoku_image(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "output.jpg")
