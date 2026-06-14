import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from skimage import color as skcolor

IMG_SIZE = 64

def load_images(data_dir):
    images = []
    labels = []
    
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                labels.append(label)
            except:
                continue
    
    return np.array(images), np.array(labels)

def extract_features(images):
    features = []
    for img in images:
        # HOG features — captures texture and shape
        gray = skcolor.rgb2gray(img)
        hog_features = hog(
            gray,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        
        # Color histograms — captures color distribution
        r_hist = np.histogram(img[:,:,0], bins=16, range=(0,256))[0]
        g_hist = np.histogram(img[:,:,1], bins=16, range=(0,256))[0]
        b_hist = np.histogram(img[:,:,2], bins=16, range=(0,256))[0]
        
        # Normalize histograms
        r_hist = r_hist / (r_hist.sum() + 1e-6)
        g_hist = g_hist / (g_hist.sum() + 1e-6)
        b_hist = b_hist / (b_hist.sum() + 1e-6)
        
        # Combine all features
        combined = np.concatenate([hog_features, r_hist, g_hist, b_hist])
        features.append(combined)
    
    return np.array(features)

def encode_labels(labels):
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    return encoded, le