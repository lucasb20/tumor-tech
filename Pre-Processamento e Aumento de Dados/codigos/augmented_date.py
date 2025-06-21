import cv2 as cv
import numpy as np
import os


def augment_image(image):
    augmented_images = []

    flipped = cv.flip(image, 1)
    augmented_images.append(flipped)

    bright = cv.convertScaleAbs(image, alpha=1, beta=40)
    augmented_images.append(bright)

    dark = cv.convertScaleAbs(image, alpha=1, beta=-40)
    augmented_images.append(dark)

    high_contrast = cv.convertScaleAbs(image, alpha=1.5, beta=0)
    augmented_images.append(high_contrast)

    low_contrast = cv.convertScaleAbs(image, alpha=0.7, beta=0)
    augmented_images.append(low_contrast)

    rows, cols = image.shape
    M = cv.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rotated = cv.warpAffine(image, M, (cols, rows), borderMode=cv.BORDER_REPLICATE)
    augmented_images.append(rotated)

    M = cv.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    rotated = cv.warpAffine(image, M, (cols, rows), borderMode=cv.BORDER_REPLICATE)
    augmented_images.append(rotated)

    scale = 1.1
    zoomed = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    start_x = (zoomed.shape[1] - cols) // 2
    start_y = (zoomed.shape[0] - rows) // 2
    zoomed_cropped = zoomed[start_y:start_y + rows, start_x:start_x + cols]
    augmented_images.append(zoomed_cropped)

    return augmented_images


if __name__ == '__main__':
    input_folder = 'Imagens/Training_CLAHE/pituitary_tumor'
    output_folder = 'Imagens/Training_Augmented/pituitary_tumor'

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_folder, filename)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Erro carregando {path}")
                continue

            augmented = augment_image(img)

            cv.imwrite(os.path.join(output_folder, filename)

            for idx, aug in enumerate(augmented):
                name = filename.replace('.jpg', f'_aug{idx+1}.jpg')
                save_path = os.path.join(output_folder, name)
                cv.imwrite(save_path, aug)
