import hashlib
import os
import pickle
import threading

from typing import Callable

from dreambooth import shared
from dreambooth.dataclasses.db_concept import Concept
from dreambooth.shared import status
from dreambooth.utils.image_utils import FilenameJsonGetter, \
    make_bucket_resolutions, \
    sort_prompts, get_images
from helpers.mytqdm import mytqdm


class ClassDataset:
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, concepts: [Concept], max_width: int):
        # Existing training image data
        self.instance_prompts = []
        # Existing class image data
        self.class_prompts = []
        # Thingy to build prompts
        json_getter = FilenameJsonGetter()

        # Create available resolutions
        bucket_resos = make_bucket_resolutions(max_width)
        class_images = {}
        instance_images = {}
        total_images = 0
        for concept_idx, concept in enumerate(concepts):
            if not concept.is_valid:
                continue

            instance_dir = concept.instance_data_dir
            class_dir = concept.class_data_dir
            instance_images[concept_idx] = get_images(instance_dir)
            class_images[concept_idx] = get_images(class_dir)
            total_images += len(instance_images[concept_idx])
            total_images += len(class_images[concept_idx])

        status.textinfo = "Sorting images..."
        pbar = mytqdm(desc="Pre-processing images.", position=0)
        pbar.reset(total_images)

        for concept_idx, concept in enumerate(concepts):
            if not concept.is_valid:
                continue
            tmp_file_postfix = '.%08x.tmp' % threading.get_ident()

            def build_cache_path(image_paths):
                filename = hashlib.sha256(pickle.dumps(
                    list(sorted(image_paths)))).hexdigest() + '.pickle'
                image_dir, _ = os.path.split(image_paths[0])
                cache_dir = os.path.join(image_dir, 'parameters_cache')
                os.makedirs(cache_dir, exist_ok=True)
                path =  os.path.join(cache_dir, filename)
                return path

            if len(instance_images[concept_idx]):
                instance_cache_path = build_cache_path(instance_images[concept_idx])
                if os.path.exists(instance_cache_path):
                    with open(instance_cache_path, 'rb') as f:
                        instance_prompt_datas = pickle.load(f)
                    pbar.update(len(instance_prompt_datas))
                else:
                    # ===== Instance =====
                    pbar.set_description(f"Pre-processing images: {os.path.split(instance_dir)[1]}")
                    instance_prompt_buckets = sort_prompts(
                        concept,
                        json_getter,
                        instance_images[concept_idx],
                        bucket_resos,
                        concept_idx,
                        is_class_image=False,
                        pbar=pbar)
                    instance_prompt_datas =\
                        sum([instance_prompt_datas
                             for _, instance_prompt_datas
                             in instance_prompt_buckets.items()], [])
                    tmp_file_path = instance_cache_path + tmp_file_postfix
                    with open(tmp_file_path, 'wb') as f:
                        pickle.dump(instance_prompt_datas, f)
                    try:
                        os.rename(tmp_file_path, instance_cache_path)
                    except (FileExistsError, PermissionError):
                        os.remove(tmp_file_path)
                    
                self.instance_prompts.extend(instance_prompt_datas)

            # ===== Class =====
            class_dir = concept.class_data_dir
            if not class_dir:
                continue

            class_cache_path = build_cache_path(class_images[concept_idx])
            if os.path.exists(class_cache_path):
                with open(class_cache_path, 'rb') as f:
                    existing_prompt_datas = pickle.load(f)
                pbar.update(len(existing_prompt_datas))
            else:
                pbar.set_description(f"Pre-processing images: {os.path.split(class_dir)[1]}")
                existing_prompt_buckets = sort_prompts(
                    concept,
                    json_getter,
                    class_images[concept_idx],
                    bucket_resos,
                    concept_idx, 
                    is_class_image=True,
                    pbar=pbar)

                # Iterate over each resolution of images, per concept
                existing_prompt_datas = []
                for res, instance_prompt_datas in instance_prompt_buckets.items():
                    if len(instance_prompt_datas) == 0:
                        continue
                    existing_prompt_datas = existing_prompt_buckets[res] if res in existing_prompt_buckets.keys() else []
                    existing_prompt_datas.extend(existing_prompt_datas)

                    # Extend class prompts by the proper amount
                tmp_file_path = class_cache_path + tmp_file_postfix
                with open(tmp_file_path, 'wb') as f:
                    pickle.dump(existing_prompt_datas, f)
                try:
                    os.rename(tmp_file_path, class_cache_path)
                except (FileExistsError, PermissionError):
                    os.remove(tmp_file_path)
                
            self.class_prompts.extend(existing_prompt_datas)

        pbar.reset(0)
