import cv2
import numpy as np
import os
import numpy as np
import tensorflow as tf
import keras
from PIL import Image

import cv2

def is_at_least_half_inside(box1, box2):
    """ Check if at least 50% of box2 is inside box1 """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix1 < ix2 and iy1 < iy2:
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        box2_area = w2 * h2

        return intersection_area >= 0.5 * box2_area
    return False

def are_aligned_vertically(box1, box2, x_tolerance=40):
    """ Check if two boxes are aligned vertically within a tolerance. """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    center_x1 = x1 + w1 / 2
    center_x2 = x2 + w2 / 2

    if abs(center_x1 - center_x2) <= x_tolerance:
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            len_thresh = w1/w2
            if len_thresh<2 and len_thresh>0.5:
                return True
    return False

def merge_boxes(box1, box2):
    """ Merge two boxes into one that encompasses both. """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    new_x = min(x1, x2)
    new_y = min(y1, y2)
    new_w = max(x1 + w1, x2 + w2) - new_x
    new_h = max(y1 + h1, y2 + h2) - new_y

    return (new_x, new_y, new_w, new_h)

def merge_vertically_aligned_boxes(bounding_boxes):
    """ Merge all vertically aligned bounding boxes. """
    merged = []
    skip_indices = set()

    for i in range(len(bounding_boxes)):
        if i in skip_indices:
            continue

        current_box = bounding_boxes[i]
        for j in range(i + 1, len(bounding_boxes)):
            if j in skip_indices:
                continue

            if are_aligned_vertically(current_box, bounding_boxes[j]):
                current_box = merge_boxes(current_box, bounding_boxes[j])
                skip_indices.add(j)

        merged.append(current_box)

    return merged


def black_out_nested_boxes(image, bounding_boxes, main_box):
    """ Black out nested bounding boxes within a specific main bounding box on a cropped image """
    x_main, y_main, w_main, h_main = main_box
    for box in bounding_boxes:
        if is_at_least_half_inside(main_box, box):
            if box!=main_box:
                x_rel = box[0] - x_main
                y_rel = box[1] - y_main
                w_rel = box[2]
                h_rel = box[3]
                cv2.rectangle(image, (x_rel, y_rel), (x_rel + w_rel, y_rel + h_rel), (0, 0, 0), -1)

def crop_and_blackout(image_path, bounding_boxes, main_box):
    """ Crop an image based on main_box and black out any nested boxes """
    image = cv2.imread(image_path)
    x, y, w, h = main_box
    temp_img = image.copy()
    cropped_image = temp_img[y:y+h, x:x+w]
    black_out_nested_boxes(cropped_image, bounding_boxes, main_box)
    return cropped_image

def resize_with_padding(image, desired_size=32):
    """
    Resize the image to the desired size while maintaining aspect ratio by padding with white pixels.
    """
    old_size = image.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # resize the image
    image = image.resize(new_size, Image.LANCZOS)

    # create a new image and paste the resized image onto the center
    new_image = Image.new("L", (desired_size, desired_size), (0))  # 'L' for grayscale mode, 255 for white
    new_image.paste(image, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))

    return new_image

def deblur_image(image):
    """
    Apply a sharpening filter to the image to reduce blurriness.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def segment_and_classify(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes=merge_vertically_aligned_boxes(bounding_boxes)
    
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('Segmented Image with Bounding Boxes', image)

    cropped_images = []
    for index, (x, y, w, h) in enumerate(bounding_boxes):
        cropped_image = crop_and_blackout(bin_image_path, bounding_boxes, bounding_boxes[index])
        cropped_images.append((cropped_image, index))
        window_name = f'Cropped Image {index+1}'
        #cv2.imshow(window_name, cropped_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    model = keras.models.load_model('../keras/model.keras')
    mappings = {'!': 0, '(': 1, ')': 2, '+': 3, ',': 4, '-': 5, '0': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, '6': 12, 
                '7': 13, '8': 14, '9': 15, '=': 16, 'A': 17, 'C': 18, 'Delta': 19, 'G': 20, 'H': 21, 'M': 22, 'N': 23, 'R': 24, 
                'S': 25, 'T': 26, 'X': 27, '[': 28, ']': 29, 'alpha': 30, 'ascii_124': 31, 'b': 32, 'beta': 33, 'cos': 34, 'd': 35, 
                'div': 36, 'e': 37, 'exists': 38, 'f': 39, 'forall': 40, 'forward_slash': 41, 'gamma': 42, 'geq': 43, 'gt': 44, 'i': 45, 
                'in': 46, 'infty': 47, 'int': 48, 'j': 49, 'k': 50, 'l': 51, 'lambda': 52, 'ldots': 53, 'leq': 54, 'lim': 55, 'log': 56, 
                'lt': 57, 'mu': 58, 'neq': 59, 'o': 60, 'p': 61, 'phi': 62, 'pi': 63, 'pm': 64, 'prime': 65, 'q': 66, 'rightarrow': 67, 
                'sigma': 68, 'sin': 69, 'sqrt': 70, 'sum': 71, 'tan': 72, 'theta': 73, 'times': 74, 'u': 75, 'v': 76, 'w': 77, 'y': 78, 'z': 79, 
                '{': 80, '}': 81}
    inverse_mappings = dict((v,k) for k,v in mappings.items())
    transformed_imgs = []
    for img, index in cropped_images:
        img_pil = Image.fromarray(img)
        img_pil = img_pil.convert('L')

        # Convert PIL image to OpenCV format
        img_cv = np.array(img_pil)

        # Apply deblurring
        img_deblurred = deblur_image(img_cv)

        # Convert back to PIL format
        img_pil_deblurred = Image.fromarray(img_deblurred)

        img_resized = resize_with_padding(img_pil_deblurred, 32)
        img_array = np.array(img_resized)
        img_array = img_array.reshape((32, 32, 1))  # Correct channel information
        img_array = img_array / 255.0  # Normalization
        transformed_imgs.append((img_array, index))
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension for prediction
        prediction = model.predict(img_array)
        e_x = np.exp(prediction - np.max(prediction))
        probabilities = e_x / e_x.sum(axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities, axis=1)
        print(inverse_mappings[predicted_class[0]], index)

    for cropped_image, index in transformed_imgs:
        window_name = f'Cropped Image {index}'
        cv2.imshow(window_name, cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# Example usage
bin_image_path = '../data/test_input/test_segment.png'
output_dir = '../data/test_output'
segment_and_classify(bin_image_path)