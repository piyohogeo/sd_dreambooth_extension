import hashlib
import os
import pickle
import threading

from joblib import Parallel, delayed
from typing import List, Optional

from dreambooth import shared
from dreambooth.dataclasses.db_concept import Concept
from dreambooth.shared import status
from dreambooth.utils.image_utils import FilenameJsonGetter, \
    make_bucket_resolutions, \
    sort_prompts, get_images
from helpers.mytqdm import mytqdm


class ClassDataset:
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, concepts: List[Concept],
                 max_width: int,
                 parameter_weight_tag: str = 'TrainingWeight',
                 prompt_tag: Optional[str] = None):
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

            def build_cache_path(is_instance, image_paths):
                def build_cache_key(image_path):
                    return (
                        image_path,
                        os.stat(json_getter.build_parameter_path(image_path)).st_mtime)
                image_path_mtimes = Parallel(n_jobs=8)(delayed(build_cache_key)(image_path) 
                                                       for image_path in sorted(image_paths))
                filename = hashlib.sha256(pickle.dumps(
                    (is_instance,
                     parameter_weight_tag,
                     prompt_tag,
                     image_path_mtimes))).hexdigest() + '.pickle'
                image_dir, _ = os.path.split(image_paths[0])
                cache_dir = os.path.join(image_dir, 'parameters_cache')
                os.makedirs(cache_dir, exist_ok=True)
                path = os.path.join(cache_dir, filename)
                return path

            if len(instance_images[concept_idx]):
                pbar.set_description(f"Pre-processing images: {os.path.split(instance_dir)[1]}")
                instance_cache_path = build_cache_path(True, instance_images[concept_idx])
                if os.path.exists(instance_cache_path):
                    with open(instance_cache_path, 'rb') as f:
                        instance_prompt_datass = pickle.load(f)
                    pbar.update(len(instance_prompt_datass))
                else:
                    # ===== Instance =====
                    instance_prompt_buckets = sort_prompts(
                        concept,
                        json_getter,
                        instance_images[concept_idx],
                        bucket_resos,
                        concept_idx,
                        is_class_image=False,
                        pbar=pbar,
                        parameter_weight_tag=parameter_weight_tag,
                        prompt_tag=prompt_tag)
                    instance_prompt_datass =\
                        sum([instance_prompt_datas
                             for _, instance_prompt_datas
                             in instance_prompt_buckets.items()], [])
                    tmp_file_path = instance_cache_path + tmp_file_postfix
                    with open(tmp_file_path, 'wb') as f:
                        pickle.dump(instance_prompt_datass, f)
                    try:
                        os.rename(tmp_file_path, instance_cache_path)
                    except (FileExistsError, PermissionError):
                        os.remove(tmp_file_path)
                    
                self.instance_prompts.extend(instance_prompt_datass)

            # ===== Class =====
            class_dir = concept.class_data_dir
            if not class_dir:
                continue

            pbar.set_description(f"Pre-processing images: {os.path.split(class_dir)[1]}")
            class_cache_path = build_cache_path(False, class_images[concept_idx])
            if os.path.exists(class_cache_path):
                with open(class_cache_path, 'rb') as f:
                    class_prompt_datass = pickle.load(f)
                pbar.update(len(class_prompt_datass))
            else:
                class_prompt_buckets = sort_prompts(
                    concept,
                    json_getter,
                    class_images[concept_idx],
                    bucket_resos,
                    concept_idx, 
                    is_class_image=True,
                    pbar=pbar,
                    parameter_weight_tag=parameter_weight_tag,
                    prompt_tag=prompt_tag)

                class_prompt_datass =\
                    sum([class_prompt_datas
                         for _, class_prompt_datas
                         in class_prompt_buckets.items()], [])
                # Iterate over each resolution of images, per concept
                    # Extend class prompts by the proper amount
                tmp_file_path = class_cache_path + tmp_file_postfix
                with open(tmp_file_path, 'wb') as f:
                    pickle.dump(class_prompt_datass, f)
                try:
                    os.rename(tmp_file_path, class_cache_path)
                except (FileExistsError, PermissionError):
                    os.remove(tmp_file_path)
                
            self.class_prompts.extend(class_prompt_datass)

        pbar.reset(0)
