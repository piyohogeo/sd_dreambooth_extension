import hashlib
import io
import os.path
import random
import threading
import traceback
from typing import List, Tuple, Dict, Optional

import torch.utils.data
from torchvision import transforms as transforms

from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.utils.image_utils import open_and_trim
from helpers.mytqdm import mytqdm
from diffusers import AutoencoderKL


class DbDatasetForResolution(torch.utils.data.Dataset):
    _VAE_HASH_CACHE = {}

    def __init__(
            self,
            prompts: List[PromptData],
            resolution: int,
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
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        if vae is not None:
            self._calc_vae_hash(vae)
            self._cache_latents(vae)

    def _load_image(self, image_path):
        flip_index = random.randint(0, 1) if self._hflip else 0
        image = self._latents_cache[image_path][flip_index]
        return image

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
        os.makedirs(latent_dir, exist_ok=True)

        latent_path = os.path.join(
            latent_dir,
            os.path.splitext(image_path_filename)[0] + '.pt')
        if os.path.exists(latent_path):
            latents = torch.load(latent_path)
            if len(latents.shape) == 4 and latents.shape[0] == 2:
                self._latents_cache[image_path] = torch.unbind(latents)
                return

        image = open_and_trim(image_path, self._resolution, False)
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

    def _cache_latents(self, vae):
        pbar = mytqdm(
            range(len(self._prompts)),
            desc=f"Caching latents for resolution: {self._resolution}",
            position=0)
        image_paths = [prompt.src_image for prompt in self._prompts]
        random.shuffle(image_paths)
        for image_path in image_paths:
            self._cache_latent(image_path, vae)
            pbar.update()

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
        vae: Optional[AutoencoderKL]) -> Dict[Tuple[int, int], DbDatasetForResolution]:
    resolutions = set()
    for prompt in prompts:
        resolutions.add(prompt.resolution)
    resolutions = list(resolutions)
    return {resolution: DbDatasetForResolution(
                prompts,
                resolution,
                hflip,
                vae) for resolution in resolutions}

