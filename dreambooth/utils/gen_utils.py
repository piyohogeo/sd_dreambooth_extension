import traceback
from typing import List, Callable

from dreambooth.dataclasses.db_config import DreamboothConfig 
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataset.class_dataset import ClassDataset
from dreambooth.dataset.db_dataset import build_resolution_datasets
from dreambooth.dataset.bucket_sampler import ResolutionedInstanceBalancedBatchSampler
from dreambooth.shared import status


def build_resolution_dataset_and_sampler(
        args: DreamboothConfig,
        instance_prompts: List[PromptData], 
        class_prompts: List[PromptData],
        batch_size: int = None,
        vae=None,
        interleave_size: int = 1):
    print(f"Found {len(class_prompts)} reg images.")
    res_instance_datasets = build_resolution_datasets(
        instance_prompts,
        args.hflip,
        vae)
    res_class_datasets = build_resolution_datasets(
        class_prompts,
        args.hflip,
        vae)
    res_sampler, res_dataset =\
        ResolutionedInstanceBalancedBatchSampler.build_dataset_and_sampler(
            batch_size,
            res_instance_datasets,
            res_class_datasets,
            interleave_size)
    print(f"Total dataset length: {len(res_dataset)}")
    print(f"Total sampler length: {len(res_sampler)}")
    return (res_dataset, res_sampler)


def generate_classifiers(
        args: DreamboothConfig,
        ui=True,
        flash_cache=False):
    """

    @param args: A DreamboothConfig
    @param class_gen_method
    @param accelerator: An optional existing accelerator to use.
    @param ui: Whether this was called by th UI, or is being run during training.
    @return:
    generated: Number of images generated
    images: A list of images or image paths, depending on if returning to the UI or not.
    if ui is False, this will return a second array of paths representing the class paths.
    """
    instance_prompts = []
    class_prompts = []
    try:
        status.textinfo = "Preparing dataset..."
        prompt_dataset = ClassDataset(
            args.concepts(), args.resolution)
        instance_prompts = prompt_dataset.instance_prompts
        class_prompts = prompt_dataset.class_prompts
    except Exception as p:
        print(f"Exception generating dataset: {str(p)}")
        traceback.print_exc()

    if ui:
        return []
    else:
        return instance_prompts, class_prompts
