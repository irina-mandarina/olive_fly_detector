# For image processing
import cv2
import numpy as np

from pathlib import Path

import joblib

# For progress bar
from tqdm import tqdm
import time

# scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction import image
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Feature extractor"""
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
    
    def extract_shape_features(self, img):
        """Extract shape-based features"""
        # Find contours
        thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Bounding box features
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w)/h if h > 0 else 0
            extent = float(area)/(w*h) if w*h > 0 else 0
            
            return [area, perimeter, circularity, aspect_ratio, extent]
        return [0, 0, 0, 0, 0]
    
    def extract_texture_features(self, img):
        """Extract texture-based features"""
        # Gray-Level Co-occurrence Matrix
        glcm = self.calculate_glcm(img)
        contrast = np.sum(((np.arange(256)[:, None] - np.arange(256)) ** 2) * glcm)
        correlation = np.sum(glcm * np.outer(np.arange(256), np.arange(256)))
        energy = np.sum(glcm ** 2)
        homogeneity = np.sum(glcm / (1 + (np.arange(256)[:, None] - np.arange(256)) ** 2))
        
        return [contrast, correlation, energy, homogeneity]
    
    def calculate_glcm(self, img):
        """Calculate Gray-Level Co-occurrence Matrix"""
        glcm = np.zeros((256, 256))
        for i in range(img.shape[0]-1):
            for j in range(img.shape[1]-1):
                i_val = img[i,j]
                j_val = img[i,j+1]
                glcm[i_val,j_val] += 1
        glcm = glcm / np.sum(glcm)
        return glcm
    
    def extract_intensity_features(self, img):
        """Extract intensity-based features"""
        hist = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
        hist = hist / np.sum(hist)  # Normalize
        
        return [
            np.mean(img),           # Mean intensity
            np.std(img),            # Standard deviation
            np.percentile(img, 25), # First quartile
            np.percentile(img, 75), # Third quartile
            np.max(img),            # Maximum intensity
            np.min(img),            # Minimum intensity
            np.median(img),         # Median intensity
            *hist                   # Histogram
        ]
    
    def transform(self, X):
        features_list = []
        
        for img in tqdm(X, desc="Extracting features"):
            # Resize and convert to grayscale
            if img.shape[:2] != self.target_size:
                img = cv2.resize(img, self.target_size)
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # HOG features
            hog_feat = hog(img, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), visualize=False)
            
            # Shape features
            shape_features = self.extract_shape_features(img)
            
            # Texture features
            texture_features = self.extract_texture_features(img)
            
            # Intensity features
            intensity_features = self.extract_intensity_features(img)
            
            # Combine all features
            combined_features = np.concatenate([
                hog_feat,
                shape_features,
                texture_features,
                intensity_features
            ])
            
            features_list.append(combined_features)
        
        return np.array(features_list)
    
    def fit(self, X, y=None):
        return self
    


class OliveFlyDetector:
    def __init__(self, n_trees=100, max_depth=10):
        """Initialize detector with a scikit-learn pipeline"""
        self.pipeline = Pipeline([
            ('feature_extractor', ImageFeatureExtractor()),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                n_jobs=1,
                random_state=42
            ))
        ])
        
    def train(self, X, y):
        """Train the model"""
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42 # 42 is a popular random Int seed
        )
        
        print("Training model...")
        start_time = time.time()
        
        self.pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Display validation results
        y_pred = self.pipeline.predict(X_val)
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print("\nModel Performance:")
        print(classification_report(y_val, y_pred))
        
    def predict_batch(self, images, image_paths):
        """Predict multiple images with progress bar"""
        results = []
        
        for img, path in tqdm(zip(images, image_paths), total=len(images), desc="Processing images"):
            prediction, probability = self.predict(img)
            results.append({
                'path': path,
                'prediction': 'Olive Fly' if prediction else 'Not Olive Fly',
                'confidence': probability[1] if prediction else probability[0]
            })
            
        return results
    
    def predict(self, image):
        """Predict single image"""
        X = np.array([image])
        prediction = self.pipeline.predict(X)[0]
        probability = self.pipeline.predict_proba(X)[0]
        return prediction, probability
    
    def predictNP(self, image):
        """Predict single image"""
        prediction = self.pipeline.predict(image)[0]
        probability = self.pipeline.predict_proba(image)[0]
        return prediction, probability
    
    def save_model(self, filepath):
        """Save pipeline"""
        joblib.dump(self.pipeline, filepath)
    
    # With this annotation the method will be bound to the class itself and not the instance of the class
    @classmethod
    def load_model(cls, filepath):
        """Load pipeline"""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance
    
def load_dataset(data_folder):
    """Load training data with progress bar"""
    images = []
    labels = []
    paths = []
    
    data_path = Path(data_folder)
    
    # Count total files for progress bar
    total_files = len(list((data_path / 'olive_fly').glob('*.jpg'))) + \
                  len(list((data_path / 'not_olive_fly').glob('*.jpg')))
    
    with tqdm(total=total_files, desc="Loading dataset") as pbar:
        # Load positive examples
        for img_path in (data_path / 'olive_fly').glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(1)
                paths.append(str(img_path))
            pbar.update(1)
        
        # Load negative examples
        for img_path in (data_path / 'not_olive_fly').glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(0)
                paths.append(str(img_path))
            pbar.update(1)
    
    return np.array(images), np.array(labels), paths

def predict_images(test_folder, model_path):
    """Predict all images in a folder"""
    detector = OliveFlyDetector.load_model(model_path)
    
    # Load test images
    images = []
    image_paths = []
    test_path = Path(test_folder)
    
    print(f"\nLoading test images from {test_folder}...")
    for img_path in tqdm(list(test_path.glob('*.jpg')), desc="Loading test images"):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            image_paths.append(str(img_path))
    
    if not images:
        print("No images found in test folder!")
        return
    
    # Process all images
    results = detector.predict_batch(images, image_paths)
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    print(f"{'Image':<50} | {'Prediction':<15} | {'Confidence':<10}")
    print("-" * 80)
    for result in results:
        img_name = Path(result['path']).name
        print(f"{img_name:<50} | {result['prediction']:<15} | {result['confidence']:.2f}")


def detect_olive_fly(image):
    MODEL_PATH = 'olive_fly_model.joblib'
    DATA_FOLDER = 'training_data' # Folder with training data
    
    if not Path(MODEL_PATH).exists():
        print("Training new model...")
        
        # Load dataset
        images, labels, _ = load_dataset(DATA_FOLDER)
        if len(images) == 0:
            print(f"No training images found in {DATA_FOLDER}")
            print("Please create folders:")
            print(f"  {DATA_FOLDER}/olive_fly/")
            print(f"  {DATA_FOLDER}/not_olive_fly/")
            return
        
        # Train detector
        detector = OliveFlyDetector()
        detector.train(images, labels)
        detector.save_model(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    # Load model
    model = OliveFlyDetector.load_model(MODEL_PATH)
    
    # Process image
    (prediction, probability) = model.predictNP(image)
    
    return prediction == 1


def main():
    """Main execution"""
    MODEL_PATH = 'olive_fly_model.joblib'
    DATA_FOLDER = 'training_data' # Folder with training data
    TEST_FOLDER = 'test_images'  # Folder with images to classify
    
    if not Path(MODEL_PATH).exists():
        print("Training new model...")
        
        # Load dataset
        images, labels, _ = load_dataset(DATA_FOLDER)
        if len(images) == 0:
            print(f"No training images found in {DATA_FOLDER}")
            print("Please create folders:")
            print(f"  {DATA_FOLDER}/olive_fly/")
            print(f"  {DATA_FOLDER}/not_olive_fly/")
            return
        
        # Train detector
        detector = OliveFlyDetector()
        detector.train(images, labels)
        detector.save_model(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    # Predict test images
    predict_images(TEST_FOLDER, MODEL_PATH)
    
    img_path = "test_images/h_1 16 referencia.JPG"
    np_img = np.array([cv2.imread(str(img_path))])
    print(detect_olive_fly(np_img))

if __name__ == "__main__":
    main()