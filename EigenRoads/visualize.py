import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Visualize:
    def __init__(self, eigenroads, mean_image, num_eigenroads, output_dir='./outputs'):
        self.eigenvectors = eigenroads
        self.mean_image = mean_image
        self.image_shape = (64, 64)
        self.output_dir = output_dir
        self.num_eigenroads = num_eigenroads

    def visualize_top_eigenroads(self):
        for i in range(self.num_eigenroads):
            eigenvector = self.eigenvectors[i].reshape(self.image_shape)
            save_path = os.path.join(self.output_dir, f'eigenroad_{i+1}.png')
            plt.imsave(save_path, eigenvector, cmap='gray')
            print(f'Saved EigenRoad {i+1} at {save_path}')

    def reconstruct_image(self, image):
        # 이미지를 벡터화
        image_vector = image.flatten()
        
        image_centered = image_vector - self.mean_image
        
        # 고유 벡터를 사용한 재구성
        reconstruction = self.mean_image.copy()
        for i in range(self.num_eigenroads):
            weight = np.dot(image_centered, self.eigenvectors[i])
            reconstruction += weight * self.eigenvectors[i]

        # 재구성된 이미지를 원래 모양으로 변환
        reconstructed_image = reconstruction.reshape(self.image_shape)
        
        return reconstructed_image

    def save_reconstruction(self, original_image):
        # 이미지 재구성
        reconstructed_image = self.reconstruct_image(original_image)
        
        # 저장 경로
        original_save_path = os.path.join(self.output_dir, 'original_image.png')
        reconstructed_save_path = os.path.join(self.output_dir, 'reconstructed_eigenroads.png')
        
        # 원본 이미지 저장
        plt.imsave(original_save_path, original_image, cmap='gray')
        print(f'Saved Original Image at {original_save_path}')
        
        # 재구성된 이미지 저장
        plt.imsave(reconstructed_save_path, reconstructed_image, cmap='gray')
        print(f'Saved Reconstructed Image at {reconstructed_save_path}')