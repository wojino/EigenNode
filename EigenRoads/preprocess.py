import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

class Preprocess:
    def __init__(self, data_path, result_path, crop_size, data_size=500):
        self.data_path = data_path
        self.result_path = result_path
        self.crop_size = crop_size
        self.data_size = data_size
        self.mask_images = []
        self.cropped_images = []
        self.normalized_images = []
        self.preprocess_images()
    
    def load_images(self):
        i = 0
        for file in tqdm(os.listdir(self.data_path)):
            if file.endswith('_mask.png'):
                img_path = os.path.join(self.data_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                self.mask_images.append(img)
                i += 1
                if i == self.data_size:
                    break
    
    def crop_images(self, min_road_pixels=1024):
        for idx, img in tqdm(enumerate(self.mask_images)):
            height, width = img.shape
            
            for i in range(0, height, self.crop_size):
                for j in range(0, width, self.crop_size):
                    cropped_img = img[i:i+self.crop_size, j:j+self.crop_size]
                    
                    # 자른 이미지에서 흰색(도로 부분) 픽셀이 일정 수 이상이면 저장
                    if np.sum(cropped_img > 0) >= min_road_pixels:
                        self.cropped_images.append(cropped_img)
            
    def calculate_principal_axis(self, image):
        """이미지에서 주축을 계산"""
        # 이미지에서 흰색(도로 부분)의 좌표를 추출
        points = np.column_stack(np.where(image > 0))
        
        # PCA 적용하여 주축 계산
        pca = PCA(n_components=2)
        pca.fit(points)
        
        # 주축(첫 번째 주성분)
        principal_axis = pca.components_[0]
        return principal_axis

    def calculate_rotation_angle(self, principal_axis):
        """주축과 수직 축(y축)이 이루는 각도를 계산"""
        angle = np.arctan2(principal_axis[1], principal_axis[0]) * (180 / np.pi)
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        return angle

    def rotate_image(self, image, angle):
        """이미지를 주어진 각도만큼 회전"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 회전 행렬 계산
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        return rotated_image

    def calculate_center(self, image):
        """이미지에서 도로의 중심을 계산"""
        points = np.column_stack(np.where(image > 0))
        center = points.mean(axis=0).astype(int)
        return center

    def translate_image(self, image, center):
        """이미지를 중간으로 이동시키는 함수"""
        (h, w) = image.shape[:2]
        image_center = np.array([w // 2, h // 2])

        # 이동할 거리 계산
        shift = image_center - center

        # 이동 행렬 생성
        M = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
        translated_image = cv2.warpAffine(image, M, (w, h))

        return translated_image

    def normalize_image(self, image):
        """이미지를 정규화 (회전 및 중심 정렬)"""
        # 주축 계산 및 회전
        principal_axis = self.calculate_principal_axis(image)
        rotation_angle = self.calculate_rotation_angle(principal_axis)
        rotated_image = self.rotate_image(image, rotation_angle)

        # 회전 후 도로 중심을 이미지 중간으로 이동
        road_center = self.calculate_center(rotated_image)
        centered_image = self.translate_image(rotated_image, road_center)

        return centered_image
    
    def normalize_images(self):
        for img in tqdm(self.cropped_images):
            normalized_img = self.normalize_image(img)
            self.normalized_images.append(normalized_img)
    
    def save_images(self):
        for idx, img in tqdm(enumerate(self.normalized_images)):
            img_path = os.path.join(self.result_path, f'{idx:04d}.png')
            cv2.imwrite(img_path, img)
        
    def preprocess_images(self):
        self.load_images()
        print('Loaded images')

        self.crop_images()
        print('Cropped images')
        
        self.normalize_images()
        print('Normalized images')
        
        self.save_images()
        print('Saved images')
