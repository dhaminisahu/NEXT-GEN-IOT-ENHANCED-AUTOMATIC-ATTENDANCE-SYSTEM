import os
import sys
sys.path.append('./src')

from face_recognition import FaceRecognitionLSTM
from data_augmentation import FaceDatasetAugmenter
from face_detection import FaceDetector

def main():
    # Directories
    raw_dataset_dir = './face_dataset'
    augmented_dataset_dir = './augmented_face_dataset'
    processed_dataset_dir = './processed_face_dataset'
    model_save_path = './models/face_recognition_lstm.h5'

    # Step 1: Augment Dataset
    print("Starting Dataset Augmentation...")
    augmenter = FaceDatasetAugmenter(raw_dataset_dir, augmented_dataset_dir)
    augmenter.augment_images(num_augmentations=5)

    # Step 2: Face Detection and Alignment
    print("Starting Face Detection...")
    detector = FaceDetector()
    detector.detect_and_align_faces(augmented_dataset_dir, processed_dataset_dir)

    # Step 3: Train Face Recognition Model
    print("Starting Model Training...")
    face_recognition = FaceRecognitionLSTM(processed_dataset_dir)
    history = face_recognition.train_model(
        test_size=0.2, 
        epochs=50, 
        batch_size=16
    )

    # Step 4: Save Trained Model
    face_recognition.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()