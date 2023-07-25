import copy
import hashlib
import io
import os.path
import random
import threading
import traceback
from typing import List, Tuple, Union

import safetensors.torch
import torch.utils.data
from torchvision.transforms import transforms
from transformers import CLIPTokenizer

from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.shared import status
from dreambooth.utils.image_utils import make_bucket_resolutions, \
    closest_resolution, shuffle_tags, open_and_trim
from dreambooth.utils.text_utils import build_strict_tokens
from helpers.mytqdm import mytqdm

class DbDataset(torch.utils.data.Dataset):
    """
    Dataset for handling training data
    """

    def __init__(
            self,
            batch_size: int,
            instance_prompts: List[PromptData],
            class_prompts: List[PromptData],
            tokens: List[Tuple[str, str]],
            tokenizer: Union[CLIPTokenizer, None],
            resolution: int,
            hflip: bool,
            shuffle_tags: bool,
            strict_tokens: bool,
            not_pad_tokens: bool,
            debug_dataset: bool,
            model_dir: str
    ) -> None:
        super().__init__()
        self.batch_indices = []
        self.batch_samples = []
        self.cache_dir = os.path.join(model_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print("Init dataset!")
        # A dictionary of string/latent pairs matching image paths
        self.latents_cache = {}
        # A dictionary of (int, int) / List[(string, string)] of resolutions and the corresponding image paths/captions
        self.train_dict = {}
        # A dictionary of (int, int) / List[(string, string)] of resolutions and the corresponding image paths/captions
        self.class_dict = {}
        # A mash-up of the class/train dicts that is perfectly fair and balanced.
        self.sample_dict = {}
        # This is where we just keep a list of everything for batching
        self.sample_cache = []
        # This is just a list of the sample names that we can use to find where in the cache an image is
        self.sample_indices = []
        # All of the available bucket resolutions
        self.resolutions = []
        # Currently active resolution
        self.active_resolution = (0, 0)
        # The currently active image index while iterating
        self.image_index = 0
        # Total len of the dataloader
        self._length = 0
        self.batch_size = batch_size
        self.batch_sampler = torch.utils.data.BatchSampler(self, batch_size, drop_last=True)
        self.train_img_data = instance_prompts
        self.class_img_data = class_prompts
        self.num_train_images = len(self.train_img_data)
        self.num_class_images = len(self.class_img_data)

        self.tokenizer = tokenizer
        self.resolution = resolution
        self.debug_dataset = debug_dataset
        self.shuffle_tags = shuffle_tags
        self.not_pad_tokens = not_pad_tokens
        self.strict_tokens = strict_tokens
        self.tokens = tokens
        self.vae = None
        self.vae_hash = None
        self.cache_latents = False
        flip_p = 0.5 if hflip else 0.0
        if hflip:
            self.image_transforms = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(flip_p),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    def load_image(self, image_path, res):
        if self.cache_latents:
            image = self.latents_cache[image_path]
        else:
            img = open_and_trim(image_path, res, False)
            image = self.image_transforms(img)
        return image

    def _get_vae_hash(self):
        if self.vae_hash is None:
            if self.vae is not None:
                f = io.BytesIO()
                torch.save(sorted(self.vae.state_dict()), f=f)
                self.vae_hash = hashlib.sha256(f.getbuffer())
                # DEBUG
                print('vae hash:', self.vae_hash.hexdigest())
        return self.vae_hash

    def cache_latent(self, image_path, res):
        if self.vae is not None:
            image_path_dir, image_path_filename = os.path.split(image_path)
            latent_dir = os.path.join(image_path_dir,
                                      'latents',
                                      self._get_vae_hash().hexdigest())
            os.makedirs(latent_dir, exist_ok=True)
            latent_path = os.path.join(
                latent_dir,
                os.path.splitext(image_path_filename)[0] + '.pt')
            if os.path.exists(latent_path):
                self.latents_cache[image_path] = torch.load(latent_path)
            else:
                image = open_and_trim(image_path, res, False)
                img_tensor = self.image_transforms(image)
                img_tensor = img_tensor.unsqueeze(0).to(device=self.vae.device, dtype=self.vae.dtype)
                latents = self.vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
                self.latents_cache[image_path] = latents

                tmp_latent_path = latent_path\
                    + '.%08x.tmp' % threading.get_ident()
                torch.save(latents, tmp_latent_path)
                try:
                    os.rename(tmp_latent_path, latent_path)
                except (FileExistsError, PermissionError):
                    os.remove(tmp_latent_path)

    def make_buckets_with_caching(self, vae):
        self.vae = vae
        self.cache_latents = vae is not None
        state = f"Preparing Dataset ({'With Caching' if self.cache_latents else 'Without Caching'})"
        print(state)
        status.textinfo = state

        # Create a list of resolutions
        bucket_resos = make_bucket_resolutions(self.resolution)
        self.train_dict = {}

        def sort_images(img_data: List[PromptData], resos, target_dict, is_class_img):
            for prompt_data in img_data:
                path = prompt_data.src_image
                image_width, image_height = prompt_data.resolution
                reso = closest_resolution(image_width, image_height, resos)
                concept_idx = prompt_data.concept_index
                # Append the concept index to the resolution, and boom, we got ourselves split concepts.
                di = (*reso, concept_idx)
                target_dict.setdefault(di, []).append(
                    (path, is_class_img, prompt_data))

        sort_images(self.train_img_data, bucket_resos, self.train_dict, False)
        sort_images(self.class_img_data, bucket_resos, self.class_dict, True)

        def cache_images(images, reso, p_bar):
            shuffled_images = images
            random.shuffle(shuffled_images)
            for img_path, is_prior, prompt_data in shuffled_images:
                try:
                    if self.cache_latents and not self.debug_dataset:
                        self.cache_latent(img_path, reso)
                    self.sample_indices.append(img_path)
                    self.sample_cache.append(
                        (img_path, is_prior, prompt_data))
                    p_bar.update()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Exception caching: {img_path}: {e}")
                    if (img_path, is_prior, prompt_data) in self.sample_cache:
                        del self.sample_cache[(img_path, is_prior, prompt_data)]
                    if img_path in self.sample_indices:
                        del self.sample_indices[img_path]
                    if img_path in self.latents_cache:
                        del self.latents_cache[img_path]

        bucket_idx = 0
        total_len = 0
        bucket_len = {}
        max_idx_chars = len(str(len(self.train_dict.keys())))
        p_len = self.num_class_images + self.num_train_images
        nc = self.num_class_images
        ni = self.num_train_images
        ti = nc + ni
        shared.status.job_count = p_len
        shared.status.job_no = 0
        total_instances = 0
        total_classes = 0
        pbar = mytqdm(range(p_len), desc="Caching latents..." if self.cache_latents else "Processing images...", position=0)
        for dict_idx, train_images in self.train_dict.items():
            if not train_images:
                continue
            # Separate the resolution from the index where we need it
            res = (dict_idx[0], dict_idx[1])
            # This should really be the index, because we want the bucket sampler to shuffle them all
            self.resolutions.append(dict_idx)
            # Cache with the actual res, because it's used to crop
            cache_images(train_images, res, pbar)
            inst_count = len(train_images)
            class_count = 0
            if dict_idx in self.class_dict:
                # Use dict index to find class images
                class_images = self.class_dict[dict_idx]
                # Use actual res here as well
                cache_images(class_images, res, pbar)
                class_count = len(class_images)
            total_instances += inst_count
            total_classes += class_count
            example_len = inst_count + class_count
            # Use index here, not res
            bucket_len[dict_idx] = example_len
            total_len += example_len
            bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
            inst_str = str(len(train_images)).rjust(len(str(ni)), " ")
            class_str = str(class_count).rjust(len(str(nc)), " ")
            ex_str = str(example_len).rjust(len(str(ti * 2)), " ")
            # Log both here
            pbar.write(
                f"Bucket {bucket_str} {dict_idx} - Instance Images: {inst_str} | Class Images: {class_str} | Max Examples/batch: {ex_str}")
            bucket_idx += 1
        bucket_str = str(bucket_idx).rjust(max_idx_chars, " ")
        inst_str = str(total_instances).rjust(len(str(ni)), " ")
        class_str = str(total_classes).rjust(len(str(nc)), " ")
        tot_str = str(total_len).rjust(len(str(ti)), " ")
        pbar.write(
            f"Total Buckets {bucket_str} - Instance Images: {inst_str} | Class Images: {class_str} | Max Examples/batch: {tot_str}")
        self._length = total_len
        print(f"\nTotal images / batch: {self._length}, total examples: {total_len}")

    def shuffle_buckets(self):
        sample_dict = {}
        batch_indices = []
        batch_samples = []
        keys = list(self.train_dict.keys())
        if not self.debug_dataset:
            random.shuffle(keys)
        for key in keys:
            sample_list = []
            if not self.debug_dataset:
                random.shuffle(self.train_dict[key])
            for entry in self.train_dict[key]:
                sample_list.append(entry)
                batch_indices.append(entry[0])
                batch_samples.append(entry)
                if key in self.class_dict:
                    class_entries = self.class_dict[key]
                    selection = random.choice(class_entries)
                    batch_indices.append(selection[0])
                    batch_samples.append(selection)
                    sample_list.append(selection)
            sample_dict[key] = sample_list
        self.sample_dict = sample_dict
        self.batch_indices = batch_indices
        self.batch_samples = batch_samples

    def __len__(self):
        return self._length

    def get_example(self, res):
        # Select the current bucket of image paths
        bucket = self.sample_dict[res]

        # Set start position from last iteration
        img_index = self.image_index

        # Reset image index (double-check)
        if img_index >= len(bucket):
            img_index = 0

        repeats = 0
        # Grab instance image data
        image_path, is_class_image, prompt_data = bucket[img_index]
        image_index = self.sample_indices.index(image_path)

        img_index += 1

        # Reset image index
        if img_index >= len(bucket):
            img_index = 0
            repeats += 1

        self.image_index = img_index

        return image_index, repeats

    def __getitem__(self, index):
        image_path, is_class_image, prompt_data = self.sample_cache[index]
        image_data = self.load_image(image_path, self.active_resolution)
        # If we have reached the end of our bucket, increment to the next, update the count, reset image index.

        #prompt = drop_quality_tag(parameters["prompt"]) 
        example = {
            "image": image_data,
            "prompt": prompt_data.prompt,
            "negative_prompt": prompt_data.negative_prompt,
            "guidance_scale": prompt_data.scale,
            "res": self.active_resolution,
            "is_class": is_class_image
        }
        return example
