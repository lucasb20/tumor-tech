import cv2 as cv
import os
import matplotlib.pyplot as plt


def claheBatch():
    root = os.getcwd()
    input_folder = os.path.join(root, 'Imagens', 'Training_filtered', 'pituitary_tumor')
    output_folder = os.path.join(root, 'Imagens', 'Training_CLAHE', 'pituitary_tumor')

    os.makedirs(output_folder, exist_ok=True)

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"ERRO: Não foi possível carregar {input_path}")
                continue

            imgCLAHE = clahe.apply(img)
            cv.imwrite(output_path, imgCLAHE)
