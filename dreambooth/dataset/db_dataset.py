import hashlib
import io
import os.path
import pickle
import random
import threading
from typing import List, Tuple, Dict, Optional

import torch.utils.data
from PIL import Image 
from torchvision import transforms as transforms

from dreambooth.dataclasses.prompt_data import PromptData
from helpers.mytqdm import mytqdm
from diffusers import AutoencoderKL
from diffusers.models.vae import DiagonalGaussianDistribution


def vae_encode(encoder, quant_conv, img):
    h = encoder(img)
    moments = quant_conv(h)
    return DiagonalGaussianDistribution(moments).sample()


class DbDatasetForResolution(torch.utils.data.Dataset):
    _VAE_HASH_CACHE = {}

    def __init__(
            self,
            prompts: List[PromptData],
            resolution: Tuple[int, int],
            hflip: bool,
            vae: Optional[AutoencoderKL]
    ) -> None:
        super().__init__()
        print("Init dataset!")
        # A dictionary of string/latent pairs matching image paths
        self._latents_cache = {}
        # All of the available bucket resolutions
        # Currently active resolution
        self._hflip = hflip
        self._prompts = [prompt for prompt in prompts 
                         if prompt.resolution == resolution]
        self._resolution = resolution
        self._image_transforms = transforms.Compose(
            [
                transforms.RandomCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        if vae is not None:
            self._calc_vae_hash(vae)
            self._cache_latents(vae)

        self._vae_encoder = vae.encoder
        self._vae_quant_conv = vae.quant_conv
        self._vae_scaling_factor = vae.config.scaling_factor

    def _load_image(self, image_path):
        flip_index = random.randint(0, 1) if self._hflip else 0
        if image_path in self._latents_cache:
            latent = self._latents_cache[image_path][flip_index]
            return latent
        else:
            image = Image.open(image_path)
            img_tensor = self._image_transforms(image)
            if flip_index == 1:
                img_tensor = transforms.functional.hflip(img_tensor)
            img_tensor = img_tensor.to(
                device=self._vae_encoder.device,
                dtype=self._vae_encoder.dtype)
            with torch.no_grad():
                latent = vae_encode(self._vae_encoder,
                                    self._vae_quant_conv,
                                    img_tensor)
                return latent * self._vae_scaling_factor

    def _calc_vae_hash(self, vae):
        if hash(vae) in DbDatasetForResolution._VAE_HASH_CACHE:
            self._vae_hash = DbDatasetForResolution._VAE_HASH_CACHE[hash(vae)]
        f = io.BytesIO()
        torch.save(sorted(vae.state_dict()), f=f)
        self._vae_hash = hashlib.sha256(f.getbuffer())
        # DEBUG
        print('vae hash:', self._vae_hash.hexdigest())
        DbDatasetForResolution._VAE_HASH_CACHE[hash(vae)] = self._vae_hash

    def _cache_latent(self, image_path, vae):
        image_path_dir, image_path_filename = os.path.split(image_path)
        latent_dir = os.path.join(image_path_dir,
                                  'latents',
                                  self._vae_hash.hexdigest())
        latent_path = os.path.join(
            latent_dir,
            os.path.splitext(image_path_filename)[0] + '.pt')

        if os.path.exists(latent_path):
            latents = torch.load(latent_path)
            if len(latents.shape) == 4 and latents.shape[0] == 2:
                self._latents_cache[image_path] = torch.unbind(latents)
                return True

        os.makedirs(latent_dir, exist_ok=True)

        image = Image.open(image_path)
        if image.size != self._resolution:
            return False

        img_tensor = self._image_transforms(image)
        fliped_img_tensor = transforms.functional.hflip(img_tensor)

        img_tensors = torch.stack([img_tensor, fliped_img_tensor])
        img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(img_tensors).latent_dist.sample().cpu()
            latents = latents * vae.config.scaling_factor
        self._latents_cache[image_path] = torch.unbind(latents)

        tmp_latent_path = latent_path\
            + '.%08x.tmp' % threading.get_ident()
        torch.save(latents, tmp_latent_path)
        try:
            os.rename(tmp_latent_path, latent_path)
        except (FileExistsError, PermissionError):
            os.remove(tmp_latent_path)
        return True

    def _cache_latents(self, vae):
        pbar = mytqdm(
            range(len(self._prompts)),
            desc=f"Caching latents for resolution: {self._resolution}",
            position=0)
        image_paths = [prompt.src_image for prompt in self._prompts]
        image_paths.sort()

        cache_filename = hashlib.sha256(
            pickle.dumps(image_paths)).hexdigest() + '.pickle'
        image_dir, _ = os.path.split(image_paths[0])
        cache_dir = os.path.join(image_dir, 'latent_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_filename)
        if os.path.exists(cache_path):
            self._latents_cache = torch.load(cache_path)
            pbar.update(len(self._prompts))
        else:
            random.shuffle(image_paths)
            for image_path in image_paths:
                self._cache_latent(image_path, vae)
                pbar.update()
            tmp_cache_path = (cache_path
                              + '.%08x.tmp' % threading.get_ident())
            torch.save(self._latents_cache, tmp_cache_path)
            try:
                os.rename(tmp_cache_path, cache_path)
            except (FileExistsError, PermissionError):
                os.remove(tmp_cache_path)

    def get_weights(self):
        return [prompt_data.weight for prompt_data in self._prompts]

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, index):
        prompt_data = self._prompts[index]
        image_data = self._load_image(prompt_data.src_image)
        example = {
            "image": image_data,
            "prompt": prompt_data.prompt,
            "negative_prompt": prompt_data.negative_prompt,
            "guidance_scale": prompt_data.scale,
            "res": self._resolution,
            "is_class": prompt_data.is_class_image
        }
        return example


def build_resolution_datasets(
        prompts: List[PromptData],
        hflip: bool,
        vae: Optional[AutoencoderKL]) ->\
            Dict[Tuple[int, int], DbDatasetForResolution]:
    resolutions = set()
    for prompt in prompts:
        resolutions.add(prompt.resolution)
    resolutions = list(resolutions)
    return {resolution: DbDatasetForResolution(
                prompts,
                resolution,
                hflip,
                vae) for resolution in resolutions}

