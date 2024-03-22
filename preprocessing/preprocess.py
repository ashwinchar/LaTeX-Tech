import cv2
import os

def process_and_save_image(image_path, save_path):
    img = cv2.imread(image_path)
    # Apply Gaussian Blur with 5x5 kernel
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Convert to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert colors
    inverted = cv2.bitwise_not(thresh)
    # Save the processed image
    cv2.imwrite(save_path, inverted)

def process_images_in_directory(source_dir, target_dir):
    count = 0
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Walk through the directory structure
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, name)
                save_path = file_path.replace(source_dir, target_dir, 1)
                save_directory = os.path.dirname(save_path)
                # Create the target directory if it doesn't exist
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                process_and_save_image(file_path, save_path)
                if count%100==0:
                    print("processed " + str(count) + " images, last processed image: " + save_path)
                count+=1
        for name in dirs:
            # Create corresponding directories in the target structure
            dir_path = os.path.join(root, name)
            save_path = dir_path.replace(source_dir, target_dir, 1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

# source_directory = '../data/extracted_images'
# target_directory = '../data/processed_images'
source_directory = '../data/test_input'
target_directory = '../data/test_output'
process_images_in_directory(source_directory, target_directory)
