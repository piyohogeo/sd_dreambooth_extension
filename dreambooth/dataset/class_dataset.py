import os
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

            # ===== Instance =====
            pbar.set_description(f"Pre-processing images: {os.path.split(instance_dir)[1]}")
            instance_prompt_buckets = sort_prompts(concept,
                                                   json_getter,
                                                   instance_images[concept_idx],
                                                   bucket_resos,
                                                   concept_idx,
                                                   is_class_image=False,
                                                   pbar=pbar)
            for _, instance_prompt_datas in instance_prompt_buckets.items():
                # Extend instance prompts by the instance data
                self.instance_prompts.extend(instance_prompt_datas)

            # ===== Class =====
            class_dir = concept.class_data_dir
            if not class_dir:
                continue

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
            for res, instance_prompt_datas in instance_prompt_buckets.items():
                if len(instance_prompt_datas) == 0:
                    continue

                existing_prompt_datas = existing_prompt_buckets[res] if res in existing_prompt_buckets.keys() else []
                # Extend class prompts by the proper amount
                self.class_prompts.extend(existing_prompt_datas)

        pbar.reset(0)
