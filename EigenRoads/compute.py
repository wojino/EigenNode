import os
import cv2
import numpy as np
from tqdm import tqdm

class Compute:
    def __init__(self, data_path, num_eigenroads=10):
        self.data_path = data_path
        self.images = []
        self.mean_image = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.num_eigenroads = num_eigenroads
        self.find_eigenroads()
        
    def load_images(self):
        for file in tqdm(os.listdir(self.data_path)):
            img_path = os.path.join(self.data_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.images.append(img)
            
    def vectorize_images(self):
        image_vectors = [img.flatten() for img in self.images]
        return np.array(image_vectors)
    
    def calculate_mean_image(self, image_vectors):
        mean_image = np.mean(image_vectors, axis=0)
        self.mean_image = mean_image
        return mean_image
    
    def subtract_mean_image(self, image_vectors):
        centered_vectors = image_vectors - self.mean_image
        return centered_vectors
    
    def compute_svd(self, centered_vectors):
        U, S, Vt = np.linalg.svd(centered_vectors, full_matrices=False)
        self.eigenvalues = S ** 2
        self.eigenvectors = Vt  # Vt의 행들이 고유벡터들
        
        return self.eigenvalues, self.eigenvectors  
    
    def get_top_eigenvectors(self):
        top_eigenvectors = self.eigenvectors[:self.num_eigenroads]
        return top_eigenvectors
    
    def find_eigenroads(self):
        self.load_images()
        print('Loaded images')
        
        image_vectors = self.vectorize_images()
        print('Vectorized images')
        
        mean_image = self.calculate_mean_image(image_vectors)
        print('Calculated mean image')
        
        centered_vectors = self.subtract_mean_image(image_vectors)
        print('Subtracted mean image')
        
        self.compute_svd(centered_vectors)
        print('Computed eigenvectors using SVD')
        