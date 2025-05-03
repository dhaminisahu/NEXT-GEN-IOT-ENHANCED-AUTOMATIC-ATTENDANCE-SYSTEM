import os
import cv2
import numpy as np
from PIL import Image
import random

class FaceDatasetAugmenter:
    def __init__(self, input_dir, output_dir):
        """
        Initialize dataset augmentation process
        
        :param input_dir: Source directory with original images
        :param output_dir: Directory to save augmented images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def apply_brightness_variation(self, image):
        """
        Apply brightness variations to the image
        
        :param image: Input image
        :return: Brightness-adjusted image
        """
        brightness_factor = random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_horizontal_flip(self, image):
        """
        Randomly flip image horizontally
        
        :param image: Input image
        :return: Flipped image
        """
        return cv2.flip(image, 1) if random.random() < 0.5 else image

    def add_noise(self, image):
        """
        Add random noise to image
        
        :param image: Input image
        :return: Noisy image
        """
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def augment_images(self, num_augmentations=5):
        """
        Augment images in input directory
        
        :param num_augmentations: Number of augmentations per original image
        """
        for person_dir in os.listdir(self.input_dir):
            person_input_path = os.path.join(self.input_dir, person_dir)
            person_output_path = os.path.join(self.output_dir, person_dir)
            
            # Create person output directory
            os.makedirs(person_output_path, exist_ok=True)
            
            if os.path.isdir(person_input_path):
                for img_file in os.listdir(person_input_path):
                    img_path = os.path.join(person_input_path, img_file)
                    
                    # Read image
                    image = cv2.imread(img_path)
                    
                    # Ensure image is valid
                    if image is not None:
                        # Generate augmented images
                        for i in range(num_augmentations):
                            augmented = image.copy()
                            
                            # Apply random augmentations
                            augmented = self.apply_brightness_variation(augmented)
                            augmented = self.apply_horizontal_flip(augmented)
                            augmented = self.add_noise(augmented)
                            
                            # Save augmented image
                            output_filename = f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg"
                            output_path = os.path.join(person_output_path, output_filename)
                            cv2.imwrite(output_path, augmented)

def main():
    input_dir = './face_dataset'
    output_dir = './augmented_face_dataset'
    
    augmenter = FaceDatasetAugmenter(input_dir, output_dir)
    augmenter.augment_images(num_augmentations=5)
    print("Dataset augmentation complete!")

if __name__ == '__main__':
    main()