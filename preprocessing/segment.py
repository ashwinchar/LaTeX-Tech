import cv2
import numpy as np
import os

import cv2

def merge_overlapping_boxes(boxes):
    # Function to merge overlapping bounding boxes
    merged_boxes = []
    for box in sorted(boxes, key=lambda x: x[0]):  # Sort by x coordinate
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            prev_box = merged_boxes[-1]
            # Check if boxes overlap. If so, merge them
            if box[0] <= prev_box[0] + prev_box[2]:
                new_width = max(prev_box[0] + prev_box[2], box[0] + box[2]) - prev_box[0]
                new_height = max(prev_box[1] + prev_box[3], box[1] + box[3]) - prev_box[1]
                merged_boxes[-1] = (prev_box[0], prev_box[1], new_width, new_height)
            else:
                merged_boxes.append(box)
    return merged_boxes

def segment_and_classify(image_path):
    # Load the preprocessed image
    image = cv2.imread(image_path, 0)  # 0 for grayscale
    
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    # Merge overlapping bounding boxes
    #merged_boxes = merge_overlapping_boxes(bounding_boxes)
    
    # Draw bounding boxes on the image
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Show the image with bounding boxes
    cv2.imshow('Segmented Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Here, you would feed each bounding box into your classification algorithm
    # This part is omitted as it depends on your specific classifier

# Example usage
bin_image_path = '../data/test_input/test_segment.png'
output_dir = '../data/test_output'
segment_and_classify(bin_image_path)
