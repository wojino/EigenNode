import cv2
import matplotlib.pyplot as plt
import EigenRoads
import os
import numpy as np

class Utils:
    def __init__(self, visualizer):
        self.save_path = './outputs/comparison.png'
        self.preprocessed_path = './datasets/deepglobe/preprocessed/'
        self.visualizer = visualizer
        
    def binarize_image(self, image, threshold=128):
        # _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        # return binary_image
    
        return image
    
    def save_eigenroads(self):
        eigenvectors = []
        for i in range(self.visualizer.num_eigenroads):
            eigenvector = self.visualizer.eigenvectors[i].reshape(self.visualizer.image_shape)
            eigenvectors.append(eigenvector)
        
        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        
        for i in range(10):
            axes[i].imshow(eigenvectors[i], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'EigenRoad {i+1}')
            
        plt.tight_layout()
        plt.savefig(self.save_path)
    
    def save_comparison_figure(self):
        image_files = [f for f in os.listdir(self.preprocessed_path) if f.endswith('.png')][:10]
        original_images = []
        reconstructed_images = []

        for image_file in image_files:
            image_path = os.path.join(self.preprocessed_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # 원본 이미지 및 재구성 이미지 저장
            original_images.append(image)
            reconstructed_image = self.visualizer.reconstruct_image(image)
            reconstructed_images.append(reconstructed_image)
        
        reconstructed_images = [self.binarize_image(img) for img in reconstructed_images]
            
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        
        for i in range(10):
            # 원본 이미지
            axes[0, i].imshow(original_images[i], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Original {i+1}')
            
            # 재구성 이미지
            axes[1, i].imshow(reconstructed_images[i], cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Reconstructed {i+1}')
        
        plt.tight_layout()
        plt.savefig(self.save_path)
        print(f'Saved comparison figure at {self.save_path}')
    