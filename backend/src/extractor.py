import cv2
import numpy as np
import os

def preprocess_image(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Extract edges (Adaptive Thresholding is robust for grids)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

import math

def rotate_points(pts, angle, center):
    # angle in radians
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    return np.dot(pts - center, R.T) + center

def line_intersection(line1, line2):
    # line: (vx, vy, x0, y0)
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2
    
    cross = vx1*vy2 - vy1*vx2
    if abs(cross) < 1e-6:
        return None
        
    dx = x2 - x1
    dy = y2 - y1
    t1 = (dx*vy2 - dy*vx2) / cross
    
    return np.array([x1 + t1*vx1, y1 + t1*vy1])

def find_sudoku_grid(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    if hierarchy is not None:
        hier = hierarchy[0]
        img_area = thresh.shape[0] * thresh.shape[1]
        
        num_valid_children = [0] * len(contours)
        for i in range(len(contours)):
            parent_idx = hier[i][3]
            if parent_idx != -1:
                area = cv2.contourArea(contours[i])
                if img_area * 0.0005 < area < img_area * 0.05:
                    num_valid_children[parent_idx] += 1
                    
        if any(num_valid_children):
            best_parent = np.argmax(num_valid_children)
            if num_valid_children[best_parent] >= 20: # reduced slightly to allow for missing blocks
                if cv2.contourArea(contours[best_parent]) > img_area * 0.15:
                    valid_children = []
                    for i in range(len(contours)):
                        if hier[i][3] == best_parent:
                            area = cv2.contourArea(contours[i])
                            if img_area * 0.0005 < area < img_area * 0.05:
                                peri = cv2.arcLength(contours[i], True)
                                approx = cv2.approxPolyDP(contours[i], 0.02 * peri, True)
                                valid_children.append(approx)
                            
                    all_points = np.vstack(valid_children).reshape(-1, 2)
                    
                    s = all_points.sum(axis=1)
                    diff = np.diff(all_points, axis=1).flatten()
                    tl = all_points[np.argmin(s)]
                    tr = all_points[np.argmin(diff)]
                    bl = all_points[np.argmax(diff)]
                    br = all_points[np.argmax(s)]
                    
                    # Get the angle of the top edge to rotate points axis-aligned
                    angle = math.atan2(tr[1] - tl[1], tr[0] - tl[0])
                    
                    rotated_pts = rotate_points(all_points, -angle, tl)
                    
                    min_x, max_x = np.min(rotated_pts[:, 0]), np.max(rotated_pts[:, 0])
                    min_y, max_y = np.min(rotated_pts[:, 1]), np.max(rotated_pts[:, 1])
                    
                    W = max_x - min_x
                    H = max_y - min_y
                    
                    # Find points near the 4 true grid edges (within ~1/2 cell width margin)
                    margin_x = W / 18.0
                    margin_y = H / 18.0
                    
                    top_pts = rotated_pts[rotated_pts[:, 1] < min_y + margin_y]
                    bottom_pts = rotated_pts[rotated_pts[:, 1] > max_y - margin_y]
                    left_pts = rotated_pts[rotated_pts[:, 0] < min_x + margin_x]
                    right_pts = rotated_pts[rotated_pts[:, 0] > max_x - margin_x]
                    
                    try:
                        # Fit straight lines to these boundary points in rotated space
                        top_line = cv2.fitLine(np.float32(top_pts), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                        bottom_line = cv2.fitLine(np.float32(bottom_pts), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                        left_line = cv2.fitLine(np.float32(left_pts), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                        right_line = cv2.fitLine(np.float32(right_pts), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                        
                        # Find the 4 mathematical intersections of the fitted lines
                        tl_rot = line_intersection(top_line, left_line)
                        tr_rot = line_intersection(top_line, right_line)
                        bl_rot = line_intersection(bottom_line, left_line)
                        br_rot = line_intersection(bottom_line, right_line)
                        
                        if all(p is not None for p in [tl_rot, tr_rot, bl_rot, br_rot]):
                            # Rotate the perfectly reconstructed corners back to original space
                            corners_rot = np.array([tl_rot, tr_rot, br_rot, bl_rot])
                            corners = rotate_points(corners_rot, angle, tl)
                            return np.array([[corners[0]], [corners[1]], [corners[2]], [corners[3]]], dtype=np.float32)
                    except Exception:
                        pass

    # Fallback: Largest Contour Extreme Points
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    pts = c.reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    return np.array([[tl], [tr], [br], [bl]], dtype=np.float32)

def perspective_transform(img, corners):
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = corners.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Make it a square and multiple of 9, with a minimum size to ensure cells are large enough for adaptive thresholding
    side = max(int(maxWidth), int(maxHeight), 450)
    side = (side // 9) * 9
    
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (side, side))
    
    return warp

def split_cells(img):
    cells = []
    # Divide into 81 cells
    rows = np.vsplit(img, 9)
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            cells.append(box)
    return cells

def process_cell(cell):
    # Remove grid lines proportionally to cell size
    margin_x = int(cell.shape[1] * 0.1)
    margin_y = int(cell.shape[0] * 0.1)
    cell = cell[margin_y:-margin_y, margin_x:-margin_x]
    
    # Check if empty (mostly black pixels after thresholding)
    _, thresh = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    white_pixels = cv2.countNonZero(thresh)
    
    # If foreground pixels are less than 2% of the cropped cell area, it's empty
    is_empty = white_pixels < (cell.size * 0.02)
    return cell, is_empty

def extract_sudoku(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return None
        
    processed = preprocess_image(img)
    grid_corners = find_sudoku_grid(processed)
    
    if grid_corners is None:
        print("No grid found.")
        return None

    # Save debug threshold image
    cv2.imwrite(os.path.join(output_dir, "debug_thresh.jpg"), processed)

    # Save debug image showing detected grid
    debug_img = img.copy()
    grid_corners_int = np.int32(grid_corners)
    cv2.drawContours(debug_img, [grid_corners_int], -1, (0, 0, 255), 5)
    for point in grid_corners_int:
        cv2.circle(debug_img, tuple(point[0]), 10, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(output_dir, "debug_grid.jpg"), debug_img)
        
    warped = perspective_transform(img, grid_corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Save step image
    cv2.imwrite(f"{output_dir}/step1_warped.jpg", warped)
    
    # Split
    cells = split_cells(warped_gray)
    
    processed_cells = []
    for i, cell in enumerate(cells):
        processed_cell, is_empty = process_cell(cell)
        processed_cells.append({"img": processed_cell, "empty": is_empty})
        
        # Save sample cells
        if i < 3:
            cv2.imwrite(f"{output_dir}/sample_cell_{i}.jpg", processed_cell)
            
    print(f"Extracted 81 cells. Grid saved to {output_dir}/step1_warped.jpg")
    return processed_cells

def perspective_transform_with_matrix(img, corners):
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = corners.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Make it a square and multiple of 9, with a minimum size to ensure cells are large enough for adaptive thresholding
    side = max(int(maxWidth), int(maxHeight), 450)
    side = (side // 9) * 9
    
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (side, side))
    
    return warp, M, side

def extract_for_phase4(image_path, output_dir=None):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return None
        
    processed = preprocess_image(img)
    grid_corners = find_sudoku_grid(processed)
    
    if grid_corners is None:
        print("No grid found.")
        return None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "debug_thresh.jpg"), processed)
        debug_img = img.copy()
        grid_corners_int = np.int32(grid_corners)
        cv2.drawContours(debug_img, [grid_corners_int], -1, (0, 0, 255), 5)
        for point in grid_corners_int:
            cv2.circle(debug_img, tuple(point[0]), 10, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(output_dir, "debug_grid.jpg"), debug_img)
        
    warped, M, side = perspective_transform_with_matrix(img, grid_corners)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "step1_warped.jpg"), warped)
    
    cells = split_cells(warped_gray)
    
    processed_cells = []
    for i, cell in enumerate(cells):
        processed_cell, is_empty = process_cell(cell)
        processed_cells.append({"img": processed_cell, "empty": is_empty})
        if output_dir and i < 3:
            cv2.imwrite(os.path.join(output_dir, f"sample_cell_{i}.jpg"), processed_cell)
            
    return {
        "original_img": img,
        "processed_cells": processed_cells,
        "warped_img": warped,
        "M": M,
        "side": side
    }

if __name__ == "__main__":
    import os
    os.makedirs("../data/output", exist_ok=True)
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "../data/sample.jpg"
    print(f"Testing with image: {image_path}")
    extract_sudoku(image_path, "../data/output")
