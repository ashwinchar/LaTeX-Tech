import cv2
import numpy as np
import os

import cv2

# def merge_overlapping_boxes(boxes):
#     # Function to merge overlapping bounding boxes
#     merged_boxes = []
#     for box in sorted(boxes, key=lambda x: x[0]):  # Sort by x coordinate
#         if not merged_boxes:
#             merged_boxes.append(box)
#         else:
#             prev_box = merged_boxes[-1]
#             # Check if boxes overlap. If so, merge them
#             if box[0] <= prev_box[0] + prev_box[2]:
#                 new_width = max(prev_box[0] + prev_box[2], box[0] + box[2]) - prev_box[0]
#                 new_height = max(prev_box[1] + prev_box[3], box[1] + box[3]) - prev_box[1]
#                 merged_boxes[-1] = (prev_box[0], prev_box[1], new_width, new_height)
#             else:
#                 merged_boxes.append(box)
#     return merged_boxes

def is_at_least_half_inside(box1, box2):
    """ Check if at least 50% of box2 is inside box1 """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the intersection coordinates
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    if ix1 < ix2 and iy1 < iy2:
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        box2_area = w2 * h2

        # Check if intersection is at least 50% of box2's area
        return intersection_area >= 0.5 * box2_area
    return False


def black_out_nested_boxes(image, bounding_boxes, main_box):
    """ Black out nested bounding boxes within a specific main bounding box on a cropped image """
    x_main, y_main, w_main, h_main = main_box
    for box in bounding_boxes:
        if is_at_least_half_inside(main_box, box):
            #Calculate relative coordinates for the nested box within the cropped image
            if box!=main_box:
                x_rel = box[0] - x_main
                y_rel = box[1] - y_main
                w_rel = box[2]
                h_rel = box[3]
                # Black out the nested box on the cropped image
                cv2.rectangle(image, (x_rel, y_rel), (x_rel + w_rel, y_rel + h_rel), (0, 0, 0), -1)

def crop_and_blackout(image_path, bounding_boxes, main_box):
    """ Crop an image based on main_box and black out any nested boxes """
    image = cv2.imread(image_path)
    x, y, w, h = main_box
    temp_img = image.copy()
    cropped_image = temp_img[y:y+h, x:x+w]
    black_out_nested_boxes(cropped_image, bounding_boxes, main_box)
    #cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return cropped_image

def segment_and_classify(image_path):
    # Load the original image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find contours
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate bounding boxes for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    # Draw bounding boxes on the image (optional, for visualization)
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Show the image with bounding boxes
    cv2.imshow('Segmented Image with Bounding Boxes', image)

    # Crop and display each bounding box
    for index, (x, y, w, h) in enumerate(bounding_boxes):
        # Crop the bounding box from the original image
        cropped_image = crop_and_blackout(bin_image_path, bounding_boxes, bounding_boxes[index])
        # Display the cropped image
        window_name = f'Cropped Image {index+1}'
        cv2.imshow(window_name, cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
bin_image_path = '../data/test_input/test_segment.png'
output_dir = '../data/test_output'
segment_and_classify(bin_image_path)
