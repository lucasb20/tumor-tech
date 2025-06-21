import cv2 as cv
import os
import matplotlib.pyplot as plt


def bilateralFilteringBatch():
    root = os.getcwd()
    input_folder = os.path.join(root, 'Imagens', 'Training_processed', 'glioma_tumor')
    output_folder = os.path.join(root, 'Imagens', 'Training_filtered', 'glioma_tumor_denovo')

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)

            if img is None:
                print(f"ERRO: Não foi possível carregar {input_path}")
                continue

            imgFilter = cv.bilateralFilter(img, 7, 30, 5)
            cv.imwrite(output_path, imgFilter)

            print(f"Imagem processada e salva: {output_path}")
