import os

import open_clip
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm


def read_images(image_dir):
    image_list = []
    for (path, dir, files) in os.walk(image_dir):
        for filename in files:
            ext_lower = os.path.splitext(filename)[-1].lower()
            if ext_lower == '.png' or ext_lower == '.jpg' or ext_lower == '.jpeg' or ext_lower == '.bmp' or ext_lower == '.tif':
                image_list.append(os.path.join(path, filename))
    return image_list


if __name__ == '__main__':
    print('Preprocess ROCOv2-VQARAD-SLAKE-text images.')

    input_dir = './datasets/ROCOv2-VQARAD-SLAKE-text/images'
    result_dir = f'./datasets/ROCOv2-VQARAD-SLAKE-text/images_preprocess'

    # make result directory
    os.makedirs(result_dir, exist_ok=True)

    preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')[2]

    del preprocess.transforms[-1]  # remove Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    del preprocess.transforms[-1]  # remove ToTensor()
    del preprocess.transforms[-1]  # remove _convert_to_rgb

    # read images
    image_list = read_images(input_dir)


    def run(image_path):
        img = Image.open(image_path).convert('RGB')
        img = preprocess(img)
        os.makedirs(os.path.dirname(image_path.replace(input_dir, result_dir)), exist_ok=True)
        img.save(image_path.replace(input_dir, result_dir))


    Parallel(n_jobs=48)(
        delayed(run)(i) for i in tqdm(image_list)
    )
