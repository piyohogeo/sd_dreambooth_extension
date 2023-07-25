# Borrowed heavily from https://github.com/bmaltais/kohya_ss/blob/master/train_db.py and
# https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
# With some custom bits sprinkled in and some stuff from OG diffusers as well.

import contextlib
import io
import gc
import itertools
import logging
import math
import os
import random
import time
import traceback
import numpy as np
from decimal import Decimal
from pathlib import Path

import importlib_metadata
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.nn as nn
import accelerate.utils
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils.random import set_seed as set_seed2
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    DEISMultistepScheduler,
    UniPCMultistepScheduler
)
from diffusers.utils import logging as dl, is_xformers_available
from packaging import version
from tensorflow.python.framework.random_seed import set_seed as set_seed1
from torch.cuda.profiler import profile
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dreambooth import shared
from dreambooth.dataclasses.prompt_data import PromptData
from dreambooth.dataclasses.train_result import TrainResult
from dreambooth.dataset.sample_dataset import SampleDataset
from dreambooth.deis_velocity import get_velocity
from dreambooth.diff_to_sd import compile_checkpoint, copy_diffusion_model
from dreambooth.memory import find_executable_batch_size
from dreambooth.optimization import UniversalScheduler, get_optimizer, get_noise_scheduler
from dreambooth.shared import status
from dreambooth.utils.gen_utils import (
    generate_classifiers,
    build_resolution_dataset_and_sampler
)
from dreambooth.utils.image_utils import db_save_image, get_scheduler_class
from dreambooth.utils.model_utils import (
    unload_system_models,
    import_model_class_from_model_name_or_path,
    disable_safe_unpickle,
    enable_safe_unpickle,
    xformerify,
    torch2ify,
)
from dreambooth.utils.text_utils import encode_hidden_state

from dreambooth.utils.utils import cleanup, printm, verify_locon_installed
from dreambooth.xattention import optim_to
from helpers.ema_model import EMAModel
from helpers.log_parser import LogParser
from helpers.mytqdm import mytqdm
from lora_diffusion.extra_networks import save_extra_networks
from lora_diffusion.lora import (
    save_lora_weight,
    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    get_target_module,
)

from prompt_utils import build_embed_from_prompt
from kimuraya.image_preprocess_utils import QualityTagExtracter
from kimuraya.stable_diffusion.model_utils import (
    load_lora_unet,
    load_lora_text_encoder,
    load_from_safetensor)

logger = logging.getLogger(__name__)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)
dl.set_verbosity_error()

last_samples = []
last_prompts = []

try:
    diff_version = importlib_metadata.version("diffusers")
    version_string = diff_version.split(".")
    major_version = int(version_string[0])
    minor_version = int(version_string[1])
    patch_version = int(version_string[2])
    if minor_version < 14 or (minor_version == 14 and patch_version <= 0):
        print(
            "The version of diffusers is less than or equal to 0.14.0. Performing monkey-patch..."
        )
        DEISMultistepScheduler.get_velocity = get_velocity
        UniPCMultistepScheduler.get_velocity = get_velocity
    else:
        print(
            "The version of diffusers is greater than 0.14.0, hopefully they merged the PR by now"
        )
except:
    print("Exception monkey-patching DEIS scheduler.")

export_diffusers = False
diffusers_dir = ""
try:
    from core.handlers.config import ConfigHandler
    from core.handlers.models import ModelHandler
    ch = ConfigHandler()
    mh = ModelHandler()
    export_diffusers = ch.get_item("export_diffusers", "dreambooth", True)
    diffusers_dir = os.path.join(mh.models_path, "diffusers")
except:
    pass


def set_seed(deterministic: bool):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        seed = 0
        set_seed1(seed)
        set_seed2(seed)
    else:
        torch.backends.cudnn.deterministic = False


def current_prior_loss(args, current_epoch):
    if not args.prior_loss_scale:
        return args.prior_loss_weight
    if not args.prior_loss_target:
        args.prior_loss_target = 150
    if not args.prior_loss_weight_min:
        args.prior_loss_weight_min = 0.1
    if current_epoch >= args.prior_loss_target:
        return args.prior_loss_weight_min
    percentage_completed = current_epoch / args.prior_loss_target
    prior = (
            args.prior_loss_weight * (1 - percentage_completed)
            + args.prior_loss_weight_min * percentage_completed
    )
    printm(f"Prior: {prior}")
    return prior


class Histgram(nn.Module):
    def __init__(self, bins, bin_width=None):
        super().__init__()
        if bin_width is None:
            bin_width = bins[1] - bins[0]
        self._bin_width = bin_width
        self._bins = bins
        self.tanh = nn.Tanh()

    def forward(self, x):
        return torch.cat([(torch.mean(
            0.5 - self.tanh((x - bin_center) / self._bin_width * 2.0)
            / 2.0)).unsqueeze(0) for bin_center in self._bins])


def create_vae(args):
    vae_path = (
        args.pretrained_vae_name_or_path
        if args.pretrained_vae_name_or_path
        else args.pretrained_model_name_or_path
    )
    disable_safe_unpickle()
    new_vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=args.revision,
    )
    enable_safe_unpickle()
    return new_vae


class OptimizerLoop:
    def __init__(self,
                 args,
                 logging_dir: Path,
                 result: TrainResult,
                 train_batch_size: int,
                 gradient_accumulation_steps: int,
                 profiler: profile):
        self.args = args
        self.result = result
        self._train_batch_size = train_batch_size
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._profiler = profiler

        self._is_use_ref = 'reference' in args.noise_pred_target_method
        self._is_use_uncond_null = ('uncond_cancelled_by_null'
                                    in args.noise_pred_target_method)

        self._stop_text_percentage = args.stop_text_encoder
        if not args.train_unet:
            self._stop_text_percentage = 1
        args.max_token_length = int(args.max_token_length)
        if not args.pad_tokens and args.max_token_length > 75:
            print("Cannot raise token length limit ",
                  "above 75 when pad_tokens=False")

        verify_locon_installed(args)

        precision = args.mixed_precision if not shared.force_cpu else "no"
        self._build_accelerator(precision, logging_dir)

        self._check_interrupt()
        self._build_models()

        self._check_interrupt()
        self._build_dataloader()

        if len(self.train_dataloader) == 0:
            msg = "Please provide a directory with actual images in it."
            print(msg)
            status.textinfo = msg
            self.result.msg = msg
            self.result.config = self.args
            self._stop_profiler()
            return

        sched_train_steps = self._build_lr_schueduler()
        self._prepare_objects()

        global_step, global_epoch, resume_step, first_epoch =\
            self._try_resume()

        # Train!
        max_train_steps = (self.args.num_train_epochs
                           * len(self.train_dataloader)
                           * self._total_batch_size)

        max_train_epochs = self.args.num_train_epochs
        # we calculate our number of tenc training epochs
        text_encoder_epochs = round(max_train_epochs
                                    * self._stop_text_percentage)

        if self.accelerator.is_main_process:
            print("  ***** Running training *****")
            if shared.force_cpu:
                print("  TRAINING WITH CPU ONLY")
            print("  Num batches each epoch = "
                  f"{len(self.train_dataloader)}")
            print(f"  Num Epochs = {max_train_epochs}")
            print("  Batch Size Per Device = "
                  f"{self._train_batch_size}")
            print("  Gradient Accumulation steps = "
                  f"{self._gradient_accumulation_steps}")
            print("  Total train batch size "
                  "(w. parallel, distributed) = "
                  f"{self._total_batch_size}")
            print(f"  Text Encoder Epochs: {text_encoder_epochs}")
            print(f"  Total optimization steps = {sched_train_steps}")
            print(f"  Total training steps = {max_train_steps}")
            print(f"  Resuming from checkpoint: {self.resume_from_checkpoint}")
            print(f"  First resume epoch: {first_epoch}")
            print(f"  First resume step: {resume_step}")
            print(f"  Lora: {self.args.use_lora}, Optimizer: "
                  f"{self.args.optimizer}, Prec: {precision}")
            print("  Gradient Checkpointing: "
                  f"{self.args.gradient_checkpointing}")
            print(f"  EMA: {self.args.use_ema}")
            print(f"  UNET: {self.args.train_unet}")
            print("  Freeze CLIP Normalization Layers: "
                  f"{self.args.freeze_clip_normalization}")
            print(f"  LR: {self.args.learning_rate}")
            if self.args.use_lora_extended:
                print(f"  LoRA Extended: {self.args.use_lora_extended}")
            if self.args.use_lora and self._stop_text_percentage > 0:
                print("  LoRA Text Encoder LR: "
                      f"{self.args.lora_txt_learning_rate}")
            print(f"  V2: {self.args.v2}")

        self._inner_loop(max_train_steps,
                         max_train_epochs,
                         text_encoder_epochs,
                         global_step,
                         global_epoch,
                         resume_step,
                         first_epoch)

    def _stop_profiler(self):
        if self._profiler is not None:
            try:
                print("Stopping profiler.")
                self._profiler.stop()
            except:
                pass

    def _try_resume(self):
        first_epoch = 0
        global_step = 0
        global_epoch = 0
        resume_step = 0
        self.resume_from_checkpoint = False
        new_hotness = os.path.join(
            self.args.model_dir,
            "checkpoints",
            f"checkpoint-{self.args.snapshot}"
        )
        if os.path.exists(new_hotness):
            self.accelerator.print(f"Resuming from checkpoint {new_hotness}")

            try:
                import modules.shared
                no_safe = modules.shared.cmd_opts.disable_safe_unpickle
                modules.shared.cmd_opts.disable_safe_unpickle = True
            except:
                no_safe = False

            try:
                import modules.shared
                self.accelerator.load_state(new_hotness)
                modules.shared.cmd_opts.disable_safe_unpickle = no_safe
                global_step = resume_step = self.args.revision
                self.resume_from_checkpoint = True
                first_epoch = self.args.epoch
                global_epoch = first_epoch
            except Exception as lex:
                print(f"Exception loading checkpoint: {lex}")

        return global_step, global_epoch, resume_step, first_epoch

    def _build_accelerator(self, precision, logging_dir):
        self._weight_dtype = torch.float32
        if precision == "fp16":
            self._weight_dtype = torch.float16
        elif precision == "bf16":
            self._weight_dtype = torch.bfloat16

        try:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self._gradient_accumulation_steps,
                mixed_precision=precision,
                log_with="tensorboard",
                project_dir=logging_dir,
                cpu=shared.force_cpu,
            )
        except Exception as e:
            if "AcceleratorState" in str(e):
                msg = ("Change in precision detected, "
                       "please restart the webUI entirely "
                       "to use new precision.")
            else:
                msg = f"Exception initializing accelerator: {e}"
            print(msg)
            self.result.msg = msg
            self.result.config = self.args
            self._stop_profiler()

    def _build_models(self):
        disable_safe_unpickle()
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.args.pretrained_model_name_or_path, "tokenizer"),
            revision=self.args.revision,
            use_fast=False,
        )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            self.args.pretrained_model_name_or_path, self.args.revision
        )

        # Load models and create wrapper for stable diffusion
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.args.revision,
            torch_dtype=torch.float32,
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.revision,
            torch_dtype=torch.float32,
        )
        self.unet = torch2ify(self.unet)

        if self._is_use_ref:
            self.unet_reference = UNet2DConditionModel.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="unet",
                revision=self.args.revision,
                torch_dtype=torch.float32,
            )
            self.unet_reference = torch2ify(self.unet_reference)
        else:
            self.unet_reference = None
        enable_safe_unpickle()

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights "
            "in full float32 precision when starting training - even if"
            " doing mixed precision training. "
            "copy of the weights should still be float32."
        )
        if self.args.attention == "xformers" and not shared.force_cpu:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training "
                        "in some GPUs. If you observe problems during "
                        "training, please update xFormers to "
                        "at least 0.0.17. See "
                        "https://huggingface.co/docs/diffusers/"
                        "main/en/optimization/xformers "
                        "for more details."
                    )
            else:
                raise ValueError(
                    "xformers is not available. "
                    "Make sure it is installed correctly"
                )
            xformerify(self.unet)
            if self._is_use_ref:
                xformerify(self.unet_reference)

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            print(
                "Unet loaded as datatype "
                f"{self.accelerator.unwrap_model(self.unet).dtype}"
                f". {low_precision_error_string}"
            )
        if self._is_use_ref:
            if self.accelerator.unwrap_model(
                    self.unet_reference).dtype != torch.float32:
                print(
                    "Unet Reference loaded as datatype "
                    f"{self.accelerator.unwrap_model(self.unet_reference).dtype}"
                    f". {low_precision_error_string}"
                )

        if (
                self.args.stop_text_encoder != 0
                and self.accelerator.unwrap_model(
                    self.text_encoder).dtype != torch.float32
        ):
            print(
                "Text encoder loaded as datatype "
                f"{self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/
        # cuda.html#tensorfloat-32-tf32-on-ampere-devices
        try:
            # Apparently, some versions of torch don't have
            # a cuda_version flag? IDK, but it breaks my runpod.
            if (
                    torch.cuda.is_available()
                    and float(torch.cuda_version) >= 11.0
                    and self.args.tf32_enable
            ):
                print("Attempting to enable TF32.")
                torch.backends.cuda.matmul.allow_tf32 = True
        except:
            pass

        if self.args.gradient_checkpointing:
            if self.args.train_unet:
                self.unet.enable_gradient_checkpointing()
            if self._stop_text_percentage != 0:
                self.text_encoder.gradient_checkpointing_enable()
                if self.args.use_lora:
                    self.text_encoder.text_model.embeddings.requires_grad_(True)
            else:
                self.text_encoder.to(self.accelerator.device,
                                     dtype=self._weight_dtype)

        self.ema_model = None
        if self.args.use_ema:
            if os.path.exists(
                    os.path.join(
                        self.args.pretrained_model_name_or_path,
                        "ema_unet",
                        "diffusion_pytorch_model.safetensors",
                    )
            ):
                ema_unet = UNet2DConditionModel.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="ema_unet",
                    revision=self.args.revision,
                    torch_dtype=torch.float32,
                )
                if self.args.attention == "xformers" and not shared.force_cpu:
                    xformerify(ema_unet)

                self.ema_model = EMAModel(
                    ema_unet,
                    device=self.accelerator.device,
                    dtype=self._weight_dtype
                )
            else:
                self.ema_model = EMAModel(
                    self.unet,
                    device=self.accelerator.device,
                    dtype=self._weight_dtype
                )

        if self.args.use_lora or not self.args.train_unet:
            self.unet.requires_grad_(False)
        if self._is_use_ref:
            self.unet_reference.requires_grad_(False)

        unet_lora_params = None
        text_encoder_lora_params = None
        lora_path = None
        lora_txt = None

        if self.args.use_lora:
            if (self.args.lora_model_name
                    and '.pt' in self.args.lora_model_name):
                lora_path = os.path.join(self.args.model_dir,
                                         "loras",
                                         self.args.lora_model_name)
                lora_txt = lora_path.replace(".pt", "_txt.pt")

                if (not os.path.exists(lora_path)
                        or not os.path.isfile(lora_path)):
                    lora_path = None
                    lora_txt = None

            injectable_lora = get_target_module(
                "injection", self.args.use_lora_extended)
            target_module = get_target_module(
                "module", self.args.use_lora_extended)

            unet_lora_params, _ = injectable_lora(
                self.unet,
                r=self.args.lora_unet_rank,
                loras=lora_path,
                target_replace_module=target_module,
            )

            def flatten_parameters(ps):
                if isinstance(ps, torch.nn.parameter.Parameter):
                    return [ps]
                else:
                    return sum([flatten_parameters(p) for p in ps], [])
            unet_lora_params = flatten_parameters(unet_lora_params)

            self.text_encoder.requires_grad_(False)
            if self._stop_text_percentage != 0:
                inject_trainable_txt_lora = get_target_module(
                    "injection", False)
                text_encoder_lora_params, _ = inject_trainable_txt_lora(
                    self.text_encoder,
                    target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
                    r=self.args.lora_txt_rank,
                    loras=lora_txt,
                )
                text_encoder_lora_params = flatten_parameters(
                    text_encoder_lora_params)

            def load_lora_weights():
                lora_tensors = load_from_safetensor(self.args.lora_model_name)
                load_lora_unet(self.unet, lora_tensors)
                load_lora_text_encoder(self.text_encoder, lora_tensors)

            if (self.args.lora_model_name
                    and '.safetensors' in self.args.lora_model_name):
                load_lora_weights()

            printm("Lora loaded")
            cleanup()
            printm("Cleaned")

            if self.accelerator.is_main_process:
                print('lr is scaled by num_processes '
                      '* gradient_accumulation_steps')
                print('lora_leerning_rate: ',
                      self.args.lora_learning_rate)
                print('lora_txt_learning_rate: ',
                      self.args.lora_txt_learning_rate)
            self.args.lora_learning_rate *= (
                self.accelerator.num_processes
                * self._gradient_accumulation_steps)
            self.args.lora_txt_learning_rate *= (
                self.accelerator.num_processes
                * self._gradient_accumulation_steps)
            if self.accelerator.is_main_process:
                print('-> lora_leerning_rate: ',
                      self.args.lora_learning_rate)
                print('-> lora_txt_learning_rate: ',
                      self.args.lora_txt_learning_rate)

            self.args.learning_rate = self.args.lora_learning_rate

        def build_params_to_optimize():
            if self.args.use_lora:
                if self._stop_text_percentage != 0:
                    params_to_optimize = [
                        {
                            "params": unet_lora_params,
                            "lr": self.args.lora_learning_rate,
                        },
                        {
                            "params": text_encoder_lora_params,
                            "lr": self.args.lora_txt_learning_rate,
                        },
                    ]
                else:
                    params_to_optimize = unet_lora_params
            elif self._stop_text_percentage != 0:
                if self.args.train_unet:
                    params_to_optimize = itertools.chain(
                        self.unet.parameters(), self.text_encoder.parameters())
                else:
                    params_to_optimize = itertools.chain(
                        self.text_encoder.parameters())
            else:
                params_to_optimize = self.unet.parameters()
            return params_to_optimize

        self.optimizer = get_optimizer(self.args, build_params_to_optimize())
        if self.args.split_optimizer:
            self.instance_optimizer = get_optimizer(
                self.args, build_params_to_optimize())

        self.noise_scheduler = get_noise_scheduler(self.args)

    def _build_dataloader(self):
        instance_prompts, class_prompts = generate_classifiers(
            self.args, ui=False
        )
        n_workers = 0

        if self.args.cache_latents:
            printm("Created tenc")
            vae = create_vae(self.args)
            printm("Created vae")
        else:
            vae = None
        if self.args.cache_latents:
            vae.to(self.accelerator.device, dtype=self._weight_dtype)
            vae.eval()
        printm("Loading dataset...")
        random.seed(self.accelerator.process_index)

        if self.args.split_optimizer:
            self.dataset_interleave_size = (
                self._train_batch_size
                * self.accelerator.num_processes
                * self._gradient_accumulation_steps)
        elif self.args.dataset_mix_split_size > 1:
            self.dataset_interleave_size = self.args.dataset_mix_split_size
        elif self._gradient_accumulation_steps % 2 == 0:
            self.dataset_interleave_size = (
                self._train_batch_size
                * self.accelerator.num_processes
                * self._gradient_accumulation_steps
                // 2)
        else:
            self.dataset_interleave_size = 1
        print('dataset_interleave_size:', self.dataset_interleave_size)

        train_dataset, train_sampler = build_resolution_dataset_and_sampler(
            args=self.args,
            instance_prompts=instance_prompts,
            class_prompts=class_prompts,
            batch_size=self._train_batch_size,
            vae=vae if self.args.cache_latents else None,
            interleave_size=self.dataset_interleave_size
        )

        printm("Dataset loaded.")

        if self.args.cache_latents:
            printm("Unloading vae.")
            del vae
        cleanup()

        def drop_quality_tag(prompt):
            quality_tag_prompt, removed_prompt =\
                QualityTagExtracter.extract(prompt, is_drop=True)
            return removed_prompt + ', ' + quality_tag_prompt

        def collate_fn(examples):
            # if _stop_text_percentage != 0:
            #    prompts = [drop_quality_tag(example["prompt"])
            #       for example in examples]
            # else:
            prompts = [example["prompt"] for example in examples]
            negative_prompts = [example["negative_prompt"]
                                for example in examples]
            guidance_scales = [example['guidance_scale']
                               for example in examples]
            pixel_values = [example["image"] for example in examples]
            types = [example["is_class"] for example in examples]
            pixel_values = torch.stack(pixel_values)
            if not self.args.cache_latents:
                pixel_values = pixel_values.to(
                    memory_format=torch.contiguous_format
                ).float()

            batch_data = {
                "prompt": prompts,
                "negative_prompt": negative_prompts,
                "guidance_scale": guidance_scales,
                "images": pixel_values,
                "types": types,
            }
            return batch_data

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=n_workers,
        )

    def _check_interrupt(self):
        if status.interrupted:
            self.result.msg = "Training interrupted."
            self._stop_profiler()
            raise InterruptedError()

    def _build_lr_schueduler(self):
        # This is separate, because optimizer.step is only called
        # once per "step" in training, so it's not
        # affected by batch size
        sched_train_steps = (self.args.num_train_epochs
                             * len(self.train_dataloader)
                             // (self.accelerator.num_processes
                                 * self._gradient_accumulation_steps))
        self._total_batch_size = (
                self._train_batch_size * self.accelerator.num_processes
        )

        lr_scale_pos = self.args.lr_scale_pos
        self.lr_scheduler = UniversalScheduler(
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            total_training_steps=sched_train_steps,
            min_lr=self.args.learning_rate_min,
            total_epochs=self.args.num_train_epochs,
            num_cycles=self.args.lr_cycles,
            power=self.args.lr_power,
            factor=self.args.lr_factor,
            scale_pos=lr_scale_pos,
        )
        return sched_train_steps

    def _prepare_objects(self):
        # create ema, fix OOM
        to_prepare_objects = [self.unet,
                              self.optimizer,
                              self.lr_scheduler,
                              self.train_dataloader]
        if self.args.use_ema:
            to_prepare_objects.append(self.ema_model.model)
        if self._stop_text_percentage != 0:
            to_prepare_objects.append(self.text_encoder)
        if self._is_use_ref:
            to_prepare_objects.append(self.unet_reference)
        if self.args.split_optimizer:
            to_prepare_objects.append(self.instance_optimizer)

        prepared_objects = self.accelerator.prepare(*to_prepare_objects)

        (self.unet,
         self.optimizer,
         self.lr_scheduler,
         self.train_dataloader, *rs) = prepared_objects
        if self.args.use_ema:
            self.ema_model.model, *rs = rs
        if self._stop_text_percentage != 0:
            self.text_encoder, *rs = rs
        if self._is_use_ref:
            self.unet_reference, *rs = rs
        if self.args.split_optimizer:
            self.instance_optimizer, *rs = rs
        assert rs == []

    def _build_hist(self):
        BIN_DIVS = 16
        bin_width = 1.0 / float(BIN_DIVS)
        bins = [i * bin_width for i in range(-BIN_DIVS*3, BIN_DIVS*3)]
        self._cum_hist_gen = Histgram(bins, bin_width)

        with torch.no_grad():
            self._ref_hist = self._calc_hist(
                torch.randn(
                    (2024, 2024), device='cpu')).to(
                    self.accelerator.device)

    def _calc_hist(self, xs):
        cum_hist = self._cum_hist_gen.forward(xs)
        return (torch.cat([cum_hist,
                           torch.tensor([1.0]).to(cum_hist.device)])
                - torch.cat([torch.tensor([0.0]).to(
                    device=cum_hist.device), cum_hist]))

    def _calc_target(self,
                     noise,
                     noise_pred_uncond,
                     noise_pred_uncond_null,
                     latents,
                     guidance_scale_at_step,
                     null_guidance_scale,
                     fft_method,
                     noise_norm,
                     noise_pred_uncond_reference,
                     noise_pred_reference_norm,
                     ):
        def build_hist(xs):
            assert len(xs.shape) == 4
            assert xs.shape[0] == 1
            xs = xs.squeeze(0)
            if fft_method is None:
                return torch.stack([self._calc_hist(xs)])
            if 'fft2' in fft_method:
                fft_xs = torch.stack(
                    [torch.fft.fft2(n, norm='ortho')
                        for n in noise])
            else:
                fft_xs = torch.fft.fftn(noise, norm='ortho')
            fft_xs *= np.sqrt(2.0)
            if 'flat' in fft_method:
                return torch.stack(
                    [self._calc_hist(xs),
                        self._calc_hist(torch.cat(
                            [fft_xs.real, fft_xs.imag]))], dim=0)
            else:
                return torch.stack([self._calc_hist(xs),
                                    self._calc_hist(fft_xs.real),
                                    self._calc_hist(fft_xs.imag)], dim=0)

        def build_hist_count():
            if fft_method is None:
                return 1
            if 'flat' in fft_method:
                return 2
            return 3
        if self.args.noise_pred_target_method == 'noise':
            target = noise
        elif (self.args.noise_pred_target_method
                == 'noise_reference_norm'):
            target = (noise
                      * (noise_pred_reference_norm
                         / noise_norm).reshape(
                            (*noise_norm.shape, 1, 1, 1)))
        elif (self.args.noise_pred_target_method
                == 'noise_reference_norm_inverse'):
            target = noise * (noise_norm
                              / noise_pred_reference_norm).reshape(
                            (*noise_norm.shape, 1, 1, 1))
        elif self.args.noise_pred_target_method == 'noise_hist':
            target = torch.stack(
                [torch.stack([self._ref_hist] * build_hist_count())
                 for _ in noise])
        elif self.args.noise_pred_target_method == 'latent':
            target = latents
        elif self.args.noise_pred_target_method == 'latent_hist':
            target = torch.stack([build_hist(latent.unsqueeze(0))
                                  for latent in latents], dim=0)
        elif self.args.noise_pred_target_method == 'uncond_cancelled_by_null':
            target = (noise
                      - null_guidance_scale
                      * (noise_pred_uncond_null - noise_pred_uncond))
        elif self.args.noise_pred_target_method == 'uncond_reference':
            with torch.no_grad():
                target = (noise_pred_uncond_reference
                          + guidance_scale_at_step
                          * (noise - noise_pred_uncond_reference))
        elif self.args.noise_pred_target_method == 'uncond_reference2':
            target = (noise_pred_uncond
                      + guidance_scale_at_step
                      * (noise - noise_pred_uncond_reference))

        return target

    def _calc_pred(self,
                   noise_pred,
                   latents,
                   noisy_latents,
                   timesteps,
                   noise_gain=1.0,
                   ):
        if 'split_noise_step' in self.args.noise_pred_method:
            def split_noise_from_noisy_latent(latent, noisy_latent, t):
                assert len(latent.shape) == 4
                assert len(noisy_latent.shape) == 4
                return self.noise_scheduler.split_noise(
                    latent, noisy_latent, t)
            splited_noises = []
            pred_original_latents = []
            for i, t in enumerate(timesteps):
                latent = latents[i].unsqueeze(0)
                noise_p = noise_pred[i].unsqueeze(0)
                noisy_latent = noisy_latents[i].unsqueeze(0)
                prev_step = self.noise_scheduler.step_with_noise_gain(
                    noise_p, int(t), noisy_latent, noise_gain=noise_gain)
                denoised_noisy_latent = prev_step.prev_sample
                pred_original_latent = prev_step.pred_original_sample
                splited_noise = split_noise_from_noisy_latent(
                    latent, denoised_noisy_latent, t)
                splited_noises.append(splited_noise)
                pred_original_latents.append(pred_original_latent)
            if self.args.noise_pred_method == 'split_noise_step_hist':
                pred = torch.stack([self._build_hist(n)
                                    for n in splited_noises], dim=0)
            elif (self.args.noise_pred_method
                  == 'split_noise_step_pred_latent_hist'):
                pred = torch.stack([self._build_hist(latent)
                                    for latent
                                    in pred_original_latents], dim=0)
            elif self.args.noise_pred_method == 'split_noise_step_pred_latent':
                pred = torch.cat(pred_original_latents, dim=0)
            else:
                pred = torch.cat(splited_noises, dim=0)
        else:
            pred = noise_pred
        return pred

    def _loss_fn(self, xs, ys, prod, is_instance, global_step,
                 kl_loss_gain=1.0, is_use_noise_prod_loss=False):
        prod = (prod if is_use_noise_prod_loss
                else torch.ones_like(prod))

        def calc_kl_divergence(ps, qs):
            assert len(ps.shape) == 1
            assert len(qs.shape) == 1
            assert torch.isclose(torch.sum(ps),
                                 torch.tensor(1.0, device=ps.device))
            assert torch.isclose(torch.sum(qs),
                                 torch.tensor(1.0, device=qs.device))
            qs = qs + 1e-6
            qs = qs / torch.sum(qs)
            xs = ps * torch.log(ps / qs)
            xs[torch.where(ps < 1e-5)] = 0.0
            kl = torch.sum(xs)
            return kl

        def loss_fun_batch(x, y, p):
            if ('instance_sqrt_mse' in self.args.loss_function_method
                    and is_instance):
                epsilon = 1e-4
                if global_step == 0:
                    print('instance_sqrt_mse in',
                          self.args.loss_function_method,
                          is_instance)
                return p * torch.sqrt(torch.nn.functional.mse_loss(
                    xs.float(), ys.float(), reduction="mean"
                ) + epsilon)
            if 'instance_l1' in self.args.loss_function_method and is_instance:
                if global_step == 0:
                    print('instance_l1 in',
                          self.args.loss_function_method,
                          is_instance)
                return p * torch.nn.functional.l1_loss(
                    xs.float(), ys.float(), reduction="mean"
                )
            elif ('instance_smoothl1sqrtmse' in self.args.loss_function_method
                  and is_instance):
                if global_step == 0:
                    print('instance_smoothl1sqrtmse',
                          self.args.loss_function_method, is_instance)
                sqrt_mse = torch.sqrt(
                        torch.nn.functional.mse_loss(
                            xs.float(), ys.float(), reduction="mean"))
                beta = 1.0
                return p * 2.0 * beta * torch.nn.functional.smooth_l1_loss(
                    sqrt_mse, torch.zeros_like(sqrt_mse), beta=beta
                )
            elif ('instance_smoothl1' in self.args.loss_function_method
                  and is_instance):
                if global_step == 0:
                    print('instance_smoothl1 in',
                          self.args.loss_function_method, is_instance)
                return p * torch.nn.functional.smooth_l1_loss(
                    xs.float(), ys.float(), reduction="mean", beta=1.0
                )
            elif 'kl' in self.args.loss_function_method:
                assert xs.shape == ys.shape

                def kl_loss(pss, qss):
                    assert pss.shape[0] == 1
                    assert qss.shape[0] == 1
                    pss = pss.squeeze(0)
                    qss = qss.squeeze(0)
                    return torch.mean(torch.stack(
                        [calc_kl_divergence(ps.float(), qs.float())
                         for ps, qs in zip(pss, qss)]))
                return kl_loss_gain * p * kl_loss(x, y)
            elif ('class_sqrt_mse' in self.args.loss_function_method
                  and not is_instance):
                epsilon = 1e-4
                if global_step == 0:
                    print('class_sqrt_mse in',
                          self.args.loss_function_method, is_instance)
                return p * torch.sqrt(torch.nn.functional.mse_loss(
                    xs.float(), ys.float(), reduction="mean"
                ) + epsilon)
            elif ('class_l1' in self.args.loss_function_method
                  and not is_instance):
                epsilon = 1e-4
                if global_step == 0:
                    print('class_l1 in',
                          self.args.loss_function_method, is_instance)
                return p * torch.nn.functional.l1_loss(
                    xs.float(), ys.float(), reduction="mean")
            if global_step == 0:
                print('default loss mse', is_instance)
            return p * torch.nn.functional.mse_loss(
                xs.float(), ys.float(), reduction="mean"
            )
        batched_loss = torch.stack([loss_fun_batch(x, y, p)
                                    for x, y, p in zip(xs, ys, prod)])
        assert batched_loss.shape[0] == xs.shape[0]
        assert batched_loss.shape[0] == ys.shape[0]
        assert batched_loss.shape[0] == prod.shape[0]
        return torch.sum(batched_loss)

    def _extract_optimizer_parameters(self):
        parameters = itertools.chain(
            *(p['params']
                for p
                in self.optimizer.param_groups))
        return parameters

    def _clip_grad_norm(self, train_tenc):
        if self.args.use_lora:
            params_to_clip = self._extract_optimizer_parameters()
            cliped_grad_norm = self.accelerator.clip_grad_norm_(
                params_to_clip,
                self.args.clip_grad_norm
                * np.abs(self.args.instance_loss_weight))
        else:
            if train_tenc:
                params_to_clip = itertools.chain(
                    self.unet.parameters(), self.text_encoder.parameters())
            else:
                params_to_clip = self.unet.parameters()
            cliped_grad_norm = self.accelerator.clip_grad_norm_(
                params_to_clip, self.args.clip_grad_norm_global)
        return cliped_grad_norm

    def _calc_guidance_scale(self, global_step, batch):
        if self.args.guidance_scale is None:
            guidance_scale_batch = torch.tensor(
                batch['guidance_scale']).reshape(
                (self._train_batch_size, 1, 1, 1)).to(self.accelerator.device)
        else:
            guidance_scale_batch = torch.tensor(
                self.args.guidance_scale).reshape(
                (1, 1, 1, 1)).to(self.accelerator.device)

        if self.args.guidance_scale_scheduled_steps is None:
            guidance_scale_at_step = guidance_scale_batch
        else:
            ratio = min(float(global_step) / float(
                self.args.guidance_scale_scheduled_steps), 1.0)
            guidance_scale_at_step_maximum = (
                ratio * guidance_scale_batch
                + (1 - ratio) * self.args.guidance_scale_init)
            guidance_scale_at_step = torch.minimum(
                guidance_scale_batch,
                guidance_scale_at_step_maximum)
            if global_step == 0:
                print('guidance_scale_batch:',
                      guidance_scale_batch)
                print('guidance_scale_at_step_maximum:',
                      guidance_scale_at_step_maximum)
                print('guidance_scale_at_step:',
                      guidance_scale_at_step)
        return guidance_scale_at_step

    def _calc_embeds(self, global_step, batch, train_tenc):
        if global_step == 0:
            for prompt, negative_prompt in zip(
                    batch['prompt'], batch['negative_prompt']):
                print('prompt:', prompt)
                print('negative prompt:', negative_prompt)

        def build_embed(prompt):
            if self._stop_text_percentage == 0:
                if prompt in self._embeds_cache:
                    return self._embeds_cache[prompt].to(
                        self.accelerator.device)
            if self.args.is_use_emphasis:
                embeds = build_embed_from_prompt(
                    self.text_encoder,
                    self.tokenizer,
                    prompt,
                    self.accelerator.device)
            else:
                embeds = build_embd_from_prompt_original(
                    train_tenc,
                    self.args,
                    self.text_encoder,
                    self.tokenizer,
                    prompt,
                    self.accelerator.device)
            if self._stop_text_percentage == 0:
                self._embeds_cache[prompt] = embeds.cpu()
            return embeds

        def build_embeds(prompts):
            return torch.cat(
                [build_embed(prompt) for prompt in prompts])

        encoder_hidden_states = build_embeds(batch['prompt'])
        uncond_encoder_hidden_states =\
            build_embeds(batch['negative_prompt'])
        if self._is_use_uncond_null:
            uncond_null_encoder_hidden_states =\
                build_embeds([''] * self._train_batch_size)
            encoder_hidden_states_input = torch.cat(
                [uncond_encoder_hidden_states,
                    encoder_hidden_states,
                    uncond_null_encoder_hidden_states])
        else:
            uncond_null_encoder_hidden_states = None
            encoder_hidden_states_input = torch.cat(
                [uncond_encoder_hidden_states,
                    encoder_hidden_states])

        return encoder_hidden_states_input, uncond_null_encoder_hidden_states

    def _calc_unet(self,
                   noisy_latents_input,
                   timesteps_input,
                   encoder_hidden_states_input):
        if self.args.use_ema and self.args.ema_predict:
            noise_pred_output = self.ema_model(
                noisy_latents_input, timesteps_input,
                encoder_hidden_states_input
            ).sample
        else:
            if self._is_use_ref:
                with torch.no_grad():
                    noise_pred_output_reference =\
                        self.unet_reference(
                            noisy_latents_input, timesteps_input,
                            encoder_hidden_states_input
                        ).sample
            else:
                noise_pred_output_reference = None
            noise_pred_output = self.unet(
                noisy_latents_input, timesteps_input,
                encoder_hidden_states_input
            ).sample

        if self._is_use_uncond_null:
            (noise_pred_uncond,
                noise_pred_text,
                noise_pred_uncond_null) = noise_pred_output.chunk(3)
        else:
            (noise_pred_uncond, noise_pred_text) = noise_pred_output.chunk(2)
            noise_pred_uncond_null = None

        return (noise_pred_uncond,
                noise_pred_text,
                noise_pred_uncond_null,
                noise_pred_output_reference)

    def _calc_reference_stat(self,
                             noise,
                             noise_pred_output_reference,
                             guidance_scale_at_step):
        if self._is_use_ref:
            with torch.no_grad():
                (noise_pred_uncond_reference,
                    noise_pred_text_reference) =\
                    noise_pred_output_reference.chunk(2)
                noise_pred_reference =\
                    (noise_pred_uncond_reference
                     + guidance_scale_at_step
                     * (noise_pred_text_reference
                        - noise_pred_uncond_reference))
                noise_norm = torch.linalg.norm(
                    torch.linalg.norm(
                        torch.linalg.norm(
                            noise, dim=3), dim=2), dim=1)
                noise_pred_reference_norm = torch.linalg.norm(
                    torch.linalg.norm(
                        torch.linalg.norm(
                            noise_pred_reference, dim=3), dim=2),
                    dim=1)
        else:
            noise_pred_uncond_reference = None
            noise_pred_text_reference = None
            noise_pred_output_reference = None
            noise_pred_reference = None
            noise_pred_reference_norm = None
            noise_norm = None
        return {
            'noise_pred_uncond_reference': noise_pred_uncond_reference,
            'noise_pred_reference_norm': noise_pred_reference_norm,
            'noise_norm': noise_norm,
        }

    def _split_outputs(self,
                       pred,
                       target,
                       noise_prods,
                       batch):
        def filter_chunk(is_prior):
            xss = list(zip(*[xs for xs
                             in zip(pred,
                                    target,
                                    noise_prods,
                                    batch["types"])
                             if is_prior == xs[-1]]))
            if len(xss) > 0:
                return tuple(map(torch.stack, xss[:-1]))
            else:
                return tuple([None] * 3)
            
        (model_pred_instance,
         _,
         _) = instance_chunks = filter_chunk(False)
        (model_pred_prior,
         _,
         _) = prior_chunks = filter_chunk(True)

        assert pred.shape[0] == self._train_batch_size
        assert target.shape[0] == self._train_batch_size
        if self.dataset_interleave_size >= self._train_batch_size:
            assert (model_pred_instance is None
                    or model_pred_prior is None)
        else:
            assert len(model_pred_instance) == len(model_pred_prior)

        return instance_chunks, prior_chunks

    def _inner_loop(self,
                    max_train_steps,
                    max_train_epochs,
                    text_encoder_epochs,
                    global_step,
                    global_epoch,
                    resume_step,
                    first_epoch,
                    null_guidance_scale=0.0,
                    fft_method=None,
                    train_steps=100):
        if self._stop_text_percentage == 0:
            self._embeds_cache = {}
            self.text_encoder.to(self.accelerator.device,
                                 dtype=self._weight_dtype)
        # Afterwards we recalculate our number of training epochs
        # We need to initialize the trackers we use, and also store our
        # configuration.
        # The trackers will initialize automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth")

        session_epoch = 0
        last_model_save = 0
        last_image_save = 0

        os.environ.__setattr__("CUDA_LAUNCH_BLOCKING", 1)

        def check_save(is_epoch_check=False):
            nonlocal last_model_save
            nonlocal last_image_save
            save_model_interval = self.args.save_embedding_every
            save_image_interval = self.args.save_preview_every
            save_completed = session_epoch >= max_train_epochs
            save_canceled = status.interrupted
            save_image = False
            save_model = False
            if not save_canceled and not save_completed:
                # Check to see if the number of epochs
                # since last save is gt the interval
                if 0 < save_model_interval <= session_epoch - last_model_save:
                    save_model = True
                    last_model_save = session_epoch

                # Repeat for sample images
                if 0 < save_image_interval <= session_epoch - last_image_save:
                    save_image = True
                    last_image_save = session_epoch

            else:
                print("\nSave completed/canceled.")
                if global_step > 0:
                    save_image = True
                    save_model = True

            save_snapshot = False
            save_lora = False
            save_checkpoint = False

            if is_epoch_check:
                if shared.status.do_save_samples:
                    save_image = True
                    shared.status.do_save_samples = False

                if shared.status.do_save_model:
                    save_model = True
                    shared.status.do_save_model = False

            if save_model:
                if save_canceled:
                    if global_step > 0:
                        print("Canceled, enabling saves.")
                        save_lora = self.args.save_lora_cancel
                        save_snapshot = self.args.save_state_cancel
                        save_checkpoint = self.args.save_ckpt_cancel
                elif save_completed:
                    if global_step > 0:
                        print("Completed, enabling saves.")
                        save_lora = self.args.save_lora_after
                        save_snapshot = self.args.save_state_after
                        save_checkpoint = self.args.save_ckpt_after
                else:
                    save_lora = self.args.save_lora_during
                    save_snapshot = self.args.save_state_during
                    save_checkpoint = self.args.save_ckpt_during
            if (
                    save_checkpoint
                    or save_snapshot
                    or save_lora
                    or save_image
                    or save_model
            ):
                self._save_weights(
                    save_image,
                    save_model,
                    save_snapshot,
                    save_checkpoint,
                    save_lora,
                )
            return save_model

        # Only show the progress bar once on each machine.
        progress_bar = mytqdm(
            range(global_step, max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            position=0,
        )
        progress_bar.set_description("Steps")
        progress_bar.set_postfix(refresh=True)
        self.args.revision = (
            self.args.revision if isinstance(self.args.revision, int) else
            int(self.args.revision) if str(self.args.revision).strip() else
            0
        )
        lifetime_step = self.args.revision
        lifetime_epoch = self.args.epoch
        status.job_count = max_train_steps
        status.job_no = global_step
        training_complete = False
        msg = ""

        last_tenc = 0 < text_encoder_epochs
        if self._stop_text_percentage == 0:
            last_tenc = False

        self.noise_scheduler.set_timesteps(
            train_steps, device=self.accelerator.device)

        if self.accelerator.num_processes == 1:
            self._build_hist()

        prior_grads = []
        for epoch in range(first_epoch, max_train_epochs):
            if training_complete:
                print("Training complete, breaking epoch.")
                break

            if self.args.train_unet:
                self.unet.train()

            train_tenc = epoch < text_encoder_epochs
            if self._stop_text_percentage == 0:
                train_tenc = False

            if self.args.freeze_clip_normalization:
                self.text_encoder.eval()
            else:
                self.text_encoder.train(train_tenc)

            if not self.args.use_lora:
                self.text_encoder.requires_grad_(train_tenc)
            elif train_tenc:
                # access DDP sub module
                if hasattr(self.text_encoder, 'module'):
                    self.text_encoder.module.text_model\
                        .embeddings.requires_grad_(True)
                else:
                    self.text_encoder.text_model\
                        .embeddings.requires_grad_(True)

            if last_tenc != train_tenc:
                last_tenc = train_tenc
                cleanup()

            loss_total = 0

            current_prior_loss_weight = current_prior_loss(
                self.args, current_epoch=global_epoch
            )
            random_seed = random.randint(0, 2 ** 32)
            random_seed = int(accelerate.utils.broadcast(
                torch.tensor(random_seed)))
            random.seed(random_seed)

            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if (
                        self.resume_from_checkpoint
                        and epoch == first_epoch
                        and step < resume_step
                ):
                    progress_bar.update(self._total_batch_size)
                    progress_bar.reset()
                    status.job_count = max_train_steps
                    status.job_no += self._total_batch_size
                    continue

                @contextlib.contextmanager
                def slave_sync(model):
                    if self.accelerator.sync_gradients:
                        context = contextlib.nullcontext
                    else:
                        context = self.accelerator.no_sync

                    with context(model):
                        yield

                with (self.accelerator.accumulate(None),
                      slave_sync(self.unet),
                      slave_sync(self.text_encoder)):

                    assert self.args.cache_latents
                    latents = batch["images"].to(self.accelerator.device)

                    # Sample noise that we'll add to the latents
                    if self.args.offset_noise < 0:
                        noise = torch.randn_like(latents,
                                                 device=latents.device)
                    else:
                        noise = torch.randn_like(
                            latents, device=latents.device
                        ) + self.args.offset_noise * torch.randn(
                            latents.shape[0],
                            latents.shape[1],
                            1,
                            1,
                            device=latents.device,
                        )
                    b_size = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (b_size,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()
                    if self._is_use_uncond_null:
                        unet_input_count = 3
                    else:
                        unet_input_count = 2
                    timesteps_input = torch.concat([timesteps]
                                                   * unet_input_count)

                    # Add noise to the latents according
                    # to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps)
                    noisy_latents_input = torch.cat(
                        [noisy_latents] * unet_input_count)

                    (encoder_hidden_states_input,
                     uncond_null_encoder_hidden_states) =\
                        self._calc_embeds(global_step, batch, train_tenc)

                    (noise_pred_uncond,
                     noise_pred_text,
                     noise_pred_uncond_null,
                     noise_pred_output_reference) =\
                        self._calc_unet(
                            noisy_latents_input,
                            timesteps_input,
                            encoder_hidden_states_input
                        )

                    guidance_scale_at_step = self._calc_guidance_scale(
                        global_step, batch)

                    noise_pred = (noise_pred_uncond
                                  + guidance_scale_at_step
                                  * (noise_pred_text - noise_pred_uncond))

                    reference_stats = self._calc_reference_stat(
                        noise,
                        noise_pred_output_reference,
                        guidance_scale_at_step)

                    # Get the target for loss depending on the prediction type
                    if (self.noise_scheduler.config.prediction_type
                            == "v_prediction"):
                        target = self.noise_scheduler.get_velocity(
                            latents, noise, timesteps)
                    else:
                        target = self._calc_target(
                            noise,
                            noise_pred_uncond,
                            noise_pred_uncond_null,
                            latents,
                            guidance_scale_at_step,
                            null_guidance_scale,
                            fft_method,
                            **reference_stats,
                            )

                    pred = self._calc_pred(
                                noise_pred,
                                latents,
                                noisy_latents,
                                timesteps,
                                )
                    assert pred.shape == target.shape

                    prev_ts = self.noise_scheduler.previous_timestep(timesteps)
                    noise_prods = self.noise_scheduler.calc_noise_prod(prev_ts)
                    if 'latent' in self.args.noise_pred_target_method:
                        if global_step == 0:
                            print('latent in ',
                                  self.args.noise_pred_target_method)
                        noise_prods = 1.0 - noise_prods

                    assert self.args.split_loss
                    ((model_pred_instance,
                      target,
                      instance_noise_prod),
                     (model_pred_prior,
                      target_prior,
                      prior_noise_prod)) = self._split_outputs(
                        pred,
                        target,
                        noise_prods,
                        batch)
                    # Concatenate the chunks in instance_chunks
                    # to form the model_pred_instance tensor
                    if model_pred_instance is not None:
                        instance_loss = self._loss_fn(
                            model_pred_instance,
                            target,
                            instance_noise_prod,
                            True,
                            global_step)
                        if self.args.clip_loss > 0.0:
                            if instance_loss > self.args.clip_loss:
                                instance_loss = torch.minimum(
                                    instance_loss, torch.tensor(
                                        self.args.clip_loss))

                    if model_pred_prior is not None:
                        prior_loss = self._loss_fn(
                            model_pred_prior,
                            target_prior,
                            prior_noise_prod,
                            False,
                            global_step)

                    instance_loss_weight = self.args.instance_loss_weight

                    if (model_pred_instance is not None
                            and model_pred_prior is not None):
                        loss = (instance_loss_weight * instance_loss
                                + current_prior_loss_weight * prior_loss)
                    elif model_pred_instance is not None:
                        loss = instance_loss_weight * instance_loss
                    else:
                        loss = prior_loss * current_prior_loss_weight
                    if 'celu' in self.args.loss_function_method:
                        loss = torch.nn.functional.celu(loss, alpha=0.5)

                    if self.args.l2_regularization:
                        if global_step == 0:
                            print('l2_regularization is enabled:',
                                  self.args.l2_regularization_lambda)
                        parameters = self._extract_optimizer_parameters()
                        param_l2_norm2 = sum(
                            [torch.sum(p.pow(2.0))
                             for p in parameters
                             if p.requires_grad])
                        loss += (self.args.l2_regularization_lambda
                                 * param_l2_norm2)

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        cliped_grad_norm = self._clip_grad_norm(train_tenc)
                    else:
                        cliped_grad_norm = None

                    if (self.accelerator.sync_gradients
                            and (self.args.split_optimizer
                                 or self.args.save_params_step)):
                        def collect_params():
                            parameters = self._extract_optimizer_parameters()
                            params = [p
                                      for p
                                      in parameters
                                      if (p.requires_grad
                                          and p.grad is not None)]
                            grads = [p.grad for p in params]
                            if (self.accelerator.is_main_process
                                    and (self.args.split_optimizer
                                         or self.args.save_params_step)):
                                return (params, grads)
                            else:
                                return (None, grads)
                        params, param_grads = collect_params()
                    else:
                        params, param_grads = None, None

                    @torch.no_grad()
                    def try_copy(src_ts, dst_ts):
                        if dst_ts is None or len(dst_ts) == 0:
                            dst_ts = [t.detach().clone() for t in src_ts]
                        else:
                            for src_t, dst_t in zip(src_ts, dst_ts):
                                dst_t.copy_(src_t)
                        return dst_ts

                    @torch.no_grad()
                    @self.accelerator.on_main_process
                    def save_params(filename):
                        torch.save(
                            (params, param_grads),
                            os.path.join(
                                self.args.model_dir,
                                'logging',
                                filename))
                    # wrapped .step() is called if sync_gradients
                    grads_dot = None
                    if self.accelerator.sync_gradients:
                        if self.args.split_optimizer:
                            @torch.no_grad()
                            def dot_all_tensors(xs, ys):
                                return sum([torch.sum(x * y)
                                            for x, y
                                            in zip(xs, ys)])

                            @torch.no_grad()
                            def norm_all_tensors(xs):
                                return torch.sqrt(dot_all_tensors(xs, xs))

                            @torch.no_grad()
                            def offset_scaled_(dst_ts, src_ts, scale):
                                for d, s in zip(dst_ts, src_ts):
                                    new_d = d + s * scale
                                    d.copy_(new_d)

                            if model_pred_instance is not None:
                                if (len(param_grads) > 0
                                        and len(prior_grads) > 0):
                                    save_params(
                                        'instance_params'
                                        f'_{epoch}_{step}.pt')
                                    with torch.no_grad():
                                        grads_dot = dot_all_tensors(
                                            param_grads, prior_grads)
                                        param_grads_norm = norm_all_tensors(
                                            param_grads)
                                        prior_grads_norm = norm_all_tensors(
                                            prior_grads)
                                        grads_cos = grads_dot / (
                                            param_grads_norm
                                            * prior_grads_norm)
                                        offset_coef = (-grads_cos
                                                       * param_grads_norm
                                                       / prior_grads_norm)
                                        offset_scaled_(param_grads,
                                                       prior_grads,
                                                       offset_coef)
                                        offseted_param_grads_norm =\
                                            norm_all_tensors(param_grads)
                                        offseted_grads_dot = dot_all_tensors(
                                            param_grads, prior_grads)

                                self.instance_optimizer.step()
                            else:
                                self.optimizer.step()
                                prior_grads = try_copy(param_grads,
                                                       prior_grads)
                                save_params(
                                    f'prior_params_{epoch}_{step}.pt')
                        else:
                            self.optimizer.step()
                            if self.args.save_params_step:
                                save_params(
                                    f'params_{epoch}_{step}.pt')

                    self.lr_scheduler.step()
                    if self.accelerator.sync_gradients:
                        if self.args.use_ema and self.ema_model is not None:
                            self.ema_model.step(self.unet)
                    if self._profiler is not None:
                        self._profiler.step()

                    # wrapped .zero_grad() is called if sync_gradients
                    self.optimizer.zero_grad(
                        set_to_none=self.args.gradient_set_to_none)

                allocated = round(torch.cuda.memory_allocated(0)
                                  / 1024 ** 3, 1)
                cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
                last_lr = self.lr_scheduler.get_last_lr()[0]

                global_step += self._total_batch_size
                self.args.revision += self._total_batch_size
                status.job_no += self._total_batch_size

                # logs_to_histogram = {
                #     'pred': pred.detach().cpu(),
                #     'target': target.detach().cpu(),
                #     'target_pred_dif': (target.detach().cpu()
                #                         - pred.detach().cpu())}
                del pred
                del noise_pred
                del noise_pred_text
                del noise_pred_uncond
                del noise_pred_uncond_null
                del uncond_null_encoder_hidden_states
                del noise_pred_output_reference
                del latents
                del encoder_hidden_states_input
                del noise
                del timesteps
                del timesteps_input
                del noisy_latents
                del noisy_latents_input
                del target

                loss_step = loss.detach().item()
                loss_total += loss_step
                if self.args.split_loss:
                    logs = {
                        "epoch": epoch,
                        "lr": float(last_lr),
                        "loss": float(loss_step),
                        "vram": float(cached),
                        "inst_loss_weight": instance_loss_weight,
                    }
                    if model_pred_instance is not None:
                        logs.update(
                            {
                                "inst_loss":
                                float(instance_loss.detach().item()),
                            })
                    if model_pred_prior is not None:
                        logs.update(
                            {
                                "prior_loss":
                                float(prior_loss.detach().item()),
                            })
                    if self.args.l2_regularization:
                        logs.update(
                            {
                                "param_l2_norm2":
                                float(param_l2_norm2.detach().item()),
                            })
                    if grads_dot is not None:
                        logs.update(
                            {
                                "grads_dot":
                                float(grads_dot.item()),
                                "offseted_grads_dot":
                                float(offseted_grads_dot.item()),
                                "grads_cos": float(grads_cos.item()),
                                "param_grads_norm":
                                float(param_grads_norm.item()),
                                "prior_grads_norm":
                                float(prior_grads_norm.item()),
                                "offseted_param_grads_norm":
                                float(offseted_param_grads_norm.item()),
                            })
                    if cliped_grad_norm is not None:
                        logs.update(
                            {
                                "cliped_grad_norm":
                                float(cliped_grad_norm.item()),
                            }
                        )
                else:
                    logs = {
                        "epoch": epoch, 
                        "lr": float(last_lr),
                        "loss": float(loss_step),
                        "vram": float(cached),
                    }

                status.textinfo2 = (
                    f"Loss: {'%.2f' % loss_step}, "
                    "LR: {'{:.2E}'.format(Decimal(last_lr))}, "
                    f"VRAM: {allocated}/{cached} GB"
                )
                progress_bar.update(self._total_batch_size)
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.args.revision)

                @self.accelerator.on_main_process
                def log_histogram(logs_to_histogram):
                    tracker = self.accelerator.get_tracker('tensorboard')
                    if hasattr(tracker, 'tracker'):
                        writer = tracker.tracker
                        for tag, values in logs_to_histogram.items():
                            writer.add_histogram(
                                tag, values, global_step=self.args.revision)
                # log_histogram(logs_to_histogram)

                status.job_count = max_train_steps
                status.job_no = global_step
                status.textinfo = (
                    f"Steps: {global_step}/{max_train_steps} (Current),"
                    f" {self.args.revision}/{lifetime_step + max_train_steps}"
                    f" (Lifetime), Epoch: {global_epoch}"
                )

                if math.isnan(loss_step):
                    print("Loss is NaN, your model is dead."
                          " Cancelling training.")
                    status.interrupted = True

                # Log completion message
                if training_complete or status.interrupted:
                    print("  Training complete (step check).")
                    if status.interrupted:
                        state = "cancelled"
                    else:
                        state = "complete"

                    status.textinfo = (
                        f"Training {state} {global_step}/{max_train_steps},"
                        f" {self.args.revision}"
                        f" total."
                    )

                    break

            logs = {"epoch_loss": loss_total / len(self.train_dataloader)}
            self.accelerator.log(logs, step=global_step)

            self.accelerator.wait_for_everyone()

            self.args.epoch += 1
            global_epoch += 1
            lifetime_epoch += 1
            session_epoch += 1
            self.lr_scheduler.step(is_epoch=True)
            status.job_count = max_train_steps
            status.job_no = global_step

            if self.accelerator.is_main_process:
                check_save(True)

            if self.args.num_train_epochs > 1:
                training_complete = session_epoch >= max_train_epochs

            if training_complete or status.interrupted:
                print("  Training complete (step check).")
                if status.interrupted:
                    state = "cancelled"
                else:
                    state = "complete"

                status.textinfo = (
                    f"Training {state} {global_step}/{max_train_steps},"
                    f" {self.args.revision}"
                    f" total."
                )

                break

            # Do this at the very END of the epoch,
            # only after we're sure we're not done
            if (self.args.epoch_pause_frequency > 0
                    and self.args.epoch_pause_time > 0):
                if not session_epoch % self.args.epoch_pause_frequency:
                    print(
                        "Giving the GPU a break for"
                        f" {self.args.epoch_pause_time} seconds."
                    )
                    for i in range(self.args.epoch_pause_time):
                        if status.interrupted:
                            training_complete = True
                            print("Training complete, interrupted.")
                            break
                        time.sleep(1)

        self.accelerator.end_training()
        self.result.msg = msg
        self.result.config = self.args
        self.result.samples = last_samples
        self._stop_profiler()

    def _save_weights(
            self,
            save_image, save_model, save_snapshot, save_checkpoint, save_lora
    ):
        global last_samples
        global last_prompts

        printm(" Saving weights.")
        pbar = mytqdm(
            range(4),
            desc="Saving weights",
            disable=not self.accelerator.is_local_main_process,
            position=1
        )
        pbar.set_postfix(refresh=True)

        # Create the pipeline using the trained modules and save it.
        if self.accelerator.is_main_process:
            printm("Pre-cleanup.")

            # Save random states so sample generation doesn't impact training.
            if shared.device.type == 'cuda':
                torch_rng_state = torch.get_rng_state()
                cuda_gpu_rng_state = torch.cuda.get_rng_state(device="cuda")
                cuda_cpu_rng_state = torch.cuda.get_rng_state(device="cpu")

            # optim_to(profiler, optimizer)

            if self._profiler is not None:
                cleanup()

            printm("Loading vae.")
            vae = create_vae(self.args)
            vae.to(self.accelerator.device, dtype=self._weight_dtype)

            printm("Creating pipeline.")

            s_pipeline = DiffusionPipeline.from_pretrained(
                self.args.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(
                    self.unet, keep_fp32_wrapper=True),
                text_encoder=self.accelerator.unwrap_model(
                    self.text_encoder, keep_fp32_wrapper=True
                ),
                vae=vae,
                torch_dtype=self._weight_dtype,
                revision=self.args.revision,
                safety_checker=None,
                requires_safety_checker=None,
            )

            scheduler_class = get_scheduler_class(self.args.scheduler)
            if self.args.attention == "xformers" and not shared.force_cpu:
                xformerify(s_pipeline)

            s_pipeline.scheduler = scheduler_class.from_config(
                s_pipeline.scheduler.config
            )
            if "UniPC" in self.args.scheduler:
                s_pipeline.scheduler.config.solver_type = "bh2"

            s_pipeline = s_pipeline.to(self.accelerator.device)

            with self.accelerator.autocast(), torch.inference_mode():
                if save_model:
                    # We are saving weights,
                    #  we need to ensure revision is saved
                    self.args.save()
                    try:
                        out_file = None
                        # Loras resume from pt
                        if not self.args.use_lora:
                            if save_snapshot:
                                pbar.set_description("Saving Snapshot")
                                status.textinfo = (
                                    "Saving snapshot at step"
                                    f" {self.args.revision}..."
                                )
                                self.accelerator.save_state(
                                    os.path.join(
                                        self.args.model_dir,
                                        "checkpoints",
                                        f"checkpoint-{self.args.revision}",
                                    )
                                )
                                pbar.update()

                            # We should save this regardless,
                            # because it's our fallback if no snapshot exists.
                            status.textinfo = (
                                "Saving diffusion model at step"
                                f" {self.args.revision}..."
                            )
                            pbar.set_description("Saving diffusion model")
                            s_pipeline.save_pretrained(
                                os.path.join(self.args.model_dir, "working"),
                                safe_serialization=True,
                            )
                            if self.ema_model is not None:
                                self.ema_model.save_pretrained(
                                    os.path.join(
                                        self.args.pretrained_model_name_or_path,
                                        "ema_unet",
                                    ),
                                    safe_serialization=True,
                                )
                            pbar.update()

                        elif save_lora:
                            pbar.set_description("Saving Lora Weights...")
                            # setup directory
                            loras_dir = os.path.join(
                                self.args.model_dir, "loras")
                            os.makedirs(loras_dir, exist_ok=True)
                            # setup pt path
                            if self.args.custom_model_name == "":
                                lora_model_name = self.args.model_name
                            else:
                                lora_model_name = self.args.custom_model_name
                            lora_file_prefix = (
                                f"{lora_model_name}_{self.args.revision}")
                            out_file = os.path.join(
                                loras_dir, f"{lora_file_prefix}.pt"
                            )
                            # create pt
                            tgt_module = get_target_module(
                                "module", self.args.use_lora_extended
                            )
                            save_lora_weight(
                                s_pipeline.unet, out_file, tgt_module)

                            modelmap = {"unet": (s_pipeline.unet, tgt_module)}
                            # save text_encoder
                            if self._stop_text_percentage != 0:
                                out_txt = out_file.replace(".pt", "_txt.pt")
                                modelmap["text_encoder"] = (
                                    s_pipeline.text_encoder,
                                    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
                                )
                                save_lora_weight(
                                    s_pipeline.text_encoder,
                                    out_txt,
                                    target_replace_module=(
                                        TEXT_ENCODER_DEFAULT_TARGET_REPLACE),
                                )
                                pbar.update()
                            # save extra_net
                            if self.args.save_lora_for_extra_net:
                                os.makedirs(
                                    shared.ui_lora_models_path, exist_ok=True
                                )
                                out_safe = os.path.join(
                                    shared.ui_lora_models_path,
                                    f"{lora_file_prefix}.safetensors",
                                )
                                save_extra_networks(modelmap, out_safe)
                        # package pt into checkpoint
                        if save_checkpoint:
                            pbar.set_description("Compiling Checkpoint")
                            snap_rev = (str(self.args.revision)
                                        if save_snapshot else "")
                            if export_diffusers:
                                copy_diffusion_model(
                                    self.args.model_name, diffusers_dir)
                            else:
                                compile_checkpoint(
                                    self.args.model_name,
                                    reload_models=False,
                                    lora_file_name=out_file,
                                    log=False, snap_rev=snap_rev, pbar=pbar)
                            printm("Restored, moved to acc.device.")
                    except Exception as ex:
                        print(f"Exception saving checkpoint/model: {ex}")
                        traceback.print_exc()
                        pass

            printm("Starting cleanup.")
            del s_pipeline
            if save_image:
                if "generator" in locals():
                    del generator
                try:
                    printm("Parse logs.")
                    log_parser = LogParser()
                    log_images, log_names = log_parser.parse_logs(
                        model_name=self.args.model_name
                    )
                    pbar.update()
                    for log_image in log_images:
                        last_samples.append(log_image)
                    for log_name in log_names:
                        last_prompts.append(log_name)

                    del log_images
                    del log_names
                except Exception as l:
                    traceback.print_exc()
                    print(f"Exception parsing logz: {l}")
                    pass
                status.sample_prompts = last_prompts
                status.current_image = last_samples
                pbar.update()

            printm("Unloading vae.")
            del vae

            status.current_image = last_samples
            printm("Cleanup.")

            # optim_to(profiler, optimizer, accelerator.device)

            # Restore all random states
            # to avoid having sampling impact training.
            if shared.device.type == 'cuda':
                torch.set_rng_state(torch_rng_state)
                torch.cuda.set_rng_state(cuda_cpu_rng_state, device="cpu")
                torch.cuda.set_rng_state(cuda_gpu_rng_state, device="cuda")

            cleanup()
            printm("Completed saving weights.")


# for compatibility
def build_embd_from_prompt_original(
        train_tenc,
        args,
        text_encoder,
        tokenizer,
        prompt,
        device):
    pad_tokens = args.pad_tokens if train_tenc else False

    def tokenize(caption):
        # strict_tokens:False -> add_special_tokens=True
        return tokenizer(caption, padding='max_length',
                         truncation=True,
                         add_special_tokens=True,
                         return_tensors='pt').input_ids
    encoder_hidden_states = encode_hidden_state(
        text_encoder,
        tokenize(prompt).unsqueeze(0).to(device),
        pad_tokens,
        1,
        args.max_token_length,
        tokenizer.model_max_length,
        args.clip_skip,
    )
    return encoder_hidden_states.squeeze(0)


def main() -> TrainResult:
    """
    @return: TrainResult
    """
    args = shared.db_model_config

    assert args.cache_latents
    logging_dir = Path(args.model_dir, "logging")

    result = TrainResult
    result.config = args

    set_seed(args.deterministic)

    @find_executable_batch_size(
        starting_batch_size=args.train_batch_size,
        starting_grad_size=args.gradient_accumulation_steps,
        logging_dir=logging_dir,
    )
    def inner_loop(train_batch_size: int,
                   gradient_accumulation_steps: int,
                   profiler: profile):
        try:
            OptimizerLoop(
                args,
                logging_dir,
                result=result,
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                profiler=profiler)
        finally:
            cleanup()
        return result
    
    return inner_loop()
