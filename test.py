import EigenRoads
import cv2
import os
import matplotlib.pyplot as plt

data_path = './datasets/deepglobe/train/'
preprocessed_path = './datasets/deepglobe/preprocessed/'

# for file in os.scandir(preprocessed_path):
#     os.remove(file.path)
# EigenRoads.Preprocess(data_path, preprocessed_path, 64)
com = EigenRoads.Compute(preprocessed_path)
eigenroads = com.eigenvectors

mean_image = com.mean_image
visualizer = EigenRoads.Visualize(eigenroads, mean_image, 20)

visualizer.visualize_top_eigenroads()
EigenRoads.Utils(visualizer).save_comparison_figure()
EigenRoads.Utils(visualizer).save_eigenroads()
