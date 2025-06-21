import os
from PIL import Image


def process_images_pil(source_root, dest_root, target_size=(224, 224)):
    categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

    for category in categories:
        source_folder = os.path.join(source_root, category)
        dest_folder = os.path.join(dest_root, category)
        os.makedirs(dest_folder, exist_ok=True)

        images = os.listdir(source_folder)
        for idx, image_name in enumerate(images):
            image_path = os.path.join(source_folder, image_name)

            try:
                with Image.open(image_path) as img:
                    img = img.convert("L")
                    img = img.resize(target_size)
                    new_filename = f"{category}_{idx+1:03d}.png"
                    save_path = os.path.join(dest_folder, new_filename)
                    img.save(save_path)

            except Exception as e:
                print(f"[ERRO] Falha ao processar {image_path}: {e}")
