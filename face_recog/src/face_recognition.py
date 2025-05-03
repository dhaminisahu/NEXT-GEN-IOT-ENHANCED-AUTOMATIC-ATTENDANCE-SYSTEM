from matplotlib.image import pil_to_array


def load_and_preprocess_data(self):
    """
    Load images from local directories, preprocess and prepare data.
    :return: X (image sequences), y (labels)
    """
    X = []
    y = []

    # Iterate through person directories
    for person_name in os.listdir(self.dataset_path):
        person_dir = os.path.join(self.dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        image_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(image_files) < self.sequence_length:
            print(f"Skipping {person_name}: Not enough images for sequence length.")
            continue

        # Process sequences
        for i in range(0, len(image_files) - self.sequence_length + 1, self.sequence_length):
            sequence = []
            for j in range(self.sequence_length):
                img_path = os.path.join(person_dir, image_files[i + j])
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                img = cv2.resize(img, self.img_size)
                img = pil_to_array(img) / 255.0
                sequence.append(img)
            
            if len(sequence) == self.sequence_length:
                X.append(sequence)
                y.append(person_name)

    # Convert to numpy arrays
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid data found in the dataset.")

    X = np.array(X)
    y = self.label_encoder.fit_transform(y)
    y = to_categorical(y)

    print(f"Data prepared. Total sequences: {X.shape[0]}, Classes: {len(self.label_encoder.classes_)}")
    return X, y
