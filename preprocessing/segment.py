import cv2
import numpy as np
import os

def merge_vertically_aligned_boxes(bounding_boxes, vertical_threshold=10):
    # Sort bounding boxes by their x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])
    merged_boxes = []
    current_box = None

    for box in bounding_boxes:
        if current_box is None:
            current_box = box
        else:
            # Check if boxes are vertically aligned (overlap in x-axis)
            x1, y1, w1, h1 = current_box
            x2, y2, w2, h2 = box
            if (x1 < x2 + w2 and x1 + w1 > x2) and (abs(y1 + h1 - y2) <= vertical_threshold or abs(y2 + h2 - y1) <= vertical_threshold):
                # Merge boxes if they are close enough vertically
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                current_box = (new_x, new_y, new_w, new_h)
            else:
                merged_boxes.append(current_box)
                current_box = box
    if len(current_box)>0:
        merged_boxes.append(current_box)
    
    return merged_boxes

# Now, you can use `merged_boxes` for further processing, like segmenting the characters

def segment_characters_with_merge(bin_image_path, output_dir, vertical_threshold=10):
    # Read the image
    img = cv2.imread(bin_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    # Merge bounding boxes that are vertically aligned
    merged_boxes = merge_vertically_aligned_boxes(bounding_boxes, vertical_threshold=vertical_threshold)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Segment and save characters based on merged bounding boxes
    for idx, box in enumerate(merged_boxes, start=1):
        x, y, w, h = box
        char_img = binary_img[y:y+h, x:x+w]
        cv2.imwrite(f"{output_dir}/char_{idx}.png", char_img)

    print(f"Segmented characters are saved in {output_dir}.")

# Example usage
bin_image_path = '../data/test_input/test_segment.png'
output_dir = '../data/test_output'
segment_characters_with_merge(bin_image_path, output_dir, vertical_threshold=10)
