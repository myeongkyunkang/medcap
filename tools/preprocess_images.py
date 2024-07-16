import argparse
import open_clip
import os
from PIL import Image, ImageOps
from joblib import Parallel, delayed
from tqdm import tqdm


def preprocess_images(input_dir, result_dir):
    # make result directory
    os.makedirs(result_dir, exist_ok=True)

    preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')[2]

    del preprocess.transforms[-1]  # remove Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    del preprocess.transforms[-1]  # remove ToTensor()
    del preprocess.transforms[-1]  # remove _convert_to_rgb

    # read images
    image_list = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(('.jpg', '.png'))]  # faster reading
    if len(image_list) == 0:
        image_list = [os.path.join(dir, filename)
                      for dir, _, filenames in os.walk(input_dir)
                      for filename in filenames
                      if filename.endswith(('.jpg', '.png'))]

    def run(image_path):
        img = Image.open(image_path).convert('RGB')
        max_dim, min_dim = max(img.size), min(img.size)
        if (max_dim / min_dim) > 1.8:
            padding = [(max_dim - dim) // 2 for dim in img.size]
            img = ImageOps.expand(img, border=(padding[0], padding[1], max_dim - img.size[0] - padding[0], max_dim - img.size[1] - padding[1]))
        img = preprocess(img)
        os.makedirs(os.path.dirname(image_path.replace(input_dir, result_dir)), exist_ok=True)
        img.save(image_path.replace(input_dir, result_dir))

    Parallel(n_jobs=48)(
        delayed(run)(i) for i in tqdm(image_list)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./datasets/PMC-VQA-text/images')
    parser.add_argument('--result_dir', default='./datasets/PMC-VQA-text/images_preprocess')
    args = parser.parse_args()

    preprocess_images(args.input_dir, args.result_dir)
