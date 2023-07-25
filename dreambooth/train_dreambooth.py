# Borrowed heavily from https://github.com/bmaltais/kohya_ss/blob/master/train_db.py and
# https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth
# With some custom bits sprinkled in and some stuff from OG diffusers as well.

import contextlib
import io
import gc
import itertools
import json
import logging
import math
import os
import random
import time
import traceback
import numpy as np
from decimal import Decimal
from pathlib import Path
from typing import Callable

import importlib_metadata
import torch
import torch.backends.cuda
import torch.backends.cudnn
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.nn as nn
import accelerate.utils
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.local_sgd import LocalSGD
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
from dreambooth.dataset.db_dataset import VAEEncoder
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
from kimuraya.kimuraya_utils import on_first_call
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


def syncronize_python_rng_state(accelerator):
    # syncronize python random state
    random_seed = random.randint(0, 2 ** 32)
    random_seed = int(accelerate.utils.broadcast(
        torch.tensor(random_seed).to(accelerator.device)))
    random.seed(random_seed)


def extract_optimizer_parameters(optimizer, is_check_grad=False):
    parameters = itertools.chain(
        *(p['params']
            for p
            in optimizer.param_groups))
    if is_check_grad:
        parameters = [p for p in parameters if p.grad is not None]
    return parameters


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


class ParamStatSave:
    def __init__(self, epoch, logging_dir):
        self._weights_sum = []
        self._grad2_sum = []
        self._count = 0
        self._logging_dir = logging_dir
        self._epoch = epoch

    @torch.no_grad()
    def step(self, params, grads):
        self._count += 1
        params_cpu = [p.detach().cpu() for p in params]
        grads_cpu = [g.detach().cpu() for g in grads]
        grads2 = [g * g for g in grads_cpu]
        if self._weights_sum == []:
            self._weights_sum = params_cpu
            self._grad2_sum = grads2
        else:
            self._weights_sum = list(map(lambda x: x[0] + x[1],
                                         zip(self._weights_sum,
                                             params_cpu)))
            self._grad2_sum = list(map(lambda x: x[0] + x[1],
                                       zip(self._grad2_sum, grads2)))

    @torch.no_grad()
    def epoch_end(self):
        weight_means = [weight * (1.0 / self._count)
                        for weight in self._weights_sum]
        grad2_means = [grad2 * (1.0 / self._count)
                       for grad2 in self._grad2_sum]
        param_path = os.path.join(self._logging_dir, f'stats_{self._epoch}.pt')
        torch.save({'weight_means': weight_means,
                    'grad2_means': grad2_means},
                   param_path)


class L2Regularization:
    @torch.no_grad()
    def __init__(self, args):
        self._enabled = args.l2_regularization
        self._ema_beta = args.l2_regularization_ema_beta
        self._adaptive = args.l2_regularization_adaptive
        self._lambda = args.l2_regularization_lambda
        self._target_grad_norm = torch.tensor(args.clip_grad_norm)
        self._ema_grad_norm = None
        if self._enabled:
            print('l2_regularization is enabled:',
                  self._lambda, self._adaptive)

    def calc_loss(self, parameters, logs):
        if not self._enabled:
            return torch.tensor(0.0)

        param_l2_norm2 = sum(
            [torch.sum(p.pow(2.0))
                for p in parameters])
        loss = self._lambda * param_l2_norm2
        if self._adaptive and self._ema_grad_norm is not None:
            loss *= torch.maximum(torch.tensor(1.0),
                                  (self._ema_grad_norm
                                   / self._target_grad_norm).pow(2.0))
        logs.update({
                "l2_regularization/param_l2_norm2":
                float(param_l2_norm2.detach().item()),
            })
        if self._ema_grad_norm is not None:
            logs.update({
                    "l2_regularization/ema_grad_norm":
                    float(self._ema_grad_norm.item()),
                })
        return loss

    def update_grad_norm(self, grad_norm):
        if grad_norm is None:
            self._ema_grad_norm = None
        elif self._ema_grad_norm is None:
            self._ema_grad_norm = grad_norm
        else:
            self._ema_grad_norm = (self._ema_beta * grad_norm
                                   + (1.0 - self._ema_beta)
                                   * self._ema_grad_norm)


class EWC:
    @torch.no_grad()
    def __init__(self, args, device, ref_optimizer_parameters=None):
        if args.ewc_params_path:
            model_dir = os.path.split(
                os.path.split(args.ewc_params_path)[0])[0]
            if ref_optimizer_parameters:
                with open(os.path.join(model_dir,
                                       'optimizer_parameters.json'),
                          'r', encoding='utf-8') as f:
                    optimizer_parameters = json.load(f)
                for name, ref_name in zip(optimizer_parameters,
                                          ref_optimizer_parameters):
                    name = name.replace('module.', '')
                    ref_name = ref_name.replace('module.', '')
                    if name != ref_name:
                        print(name, ref_name)

            stats = torch.load(args.ewc_params_path)
            self._weight_means = stats['weight_means']
            self._grad2_means = stats['grad2_means']
            grad2_mean_sum = sum(
                [torch.sum(g2m)
                    for g2m in self._grad2_means])
            grad2_mean_len = sum(
                [torch.numel(g2m)
                    for g2m in self._grad2_means])
            self._grad2_coef = (args.ewc_lambda
                                * grad2_mean_sum
                                * grad2_mean_len)
            print('ewc_lambda:', args.ewc_lambda)
            print('ewc._grad2_coef:', self._grad2_coef)

            self._weight_means = [w.to(device) for w in self._weight_means]
            self._grad2_means = [g.to(device) for g in self._grad2_means]
            self._grad2_coef = self._grad2_coef.to(device)
        else:
            self._weight_means = None

    def calc_loss(self, parameters, logs):
        if self._weight_means is None:
            return torch.tensor(0.0)

        loss = sum([torch.sum(self._grad2_coef * g2 * (p - w).pow(2.0))
                    for p, w, g2
                    in zip(parameters,
                           self._weight_means,
                           self._grad2_means)])
        logs.update({
                "ewc_loss":
                float(loss.detach().item()),
            })
        return loss


def get_vram_used():
    allocated = round(torch.cuda.memory_allocated(0)
                      / 1024 ** 3, 1)
    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    return allocated, cached


@torch.no_grad()
def dot_all_tensors(xs, ys):
    return sum([torch.sum(x * y) for x, y in zip(xs, ys)])


@torch.no_grad()
def norm_all_tensors(xs):
    return torch.sqrt(dot_all_tensors(xs, xs))


@torch.no_grad()
def offset_scaled_(dst_ts, src_ts, scale):
    for d, s in zip(dst_ts, src_ts):
        new_d = d + s * scale
        d.copy_(new_d)


@torch.no_grad()
def try_copy(src_ts, dst_ts):
    if len(dst_ts) == 0:
        dst_ts.extend([t.detach().clone() for t in src_ts])
    else:
        for src_t, dst_t in zip(src_ts, dst_ts):
            dst_t.copy_(src_t)


@torch.no_grad()
def offset_by_grad_(param_grads, prior_grads, logs, is_dry_run=False):
    with torch.no_grad():
        grads_dot = dot_all_tensors(param_grads, prior_grads)
        param_grads_norm = norm_all_tensors(param_grads)
        prior_grads_norm = norm_all_tensors(prior_grads)
        grads_cos = torch.minimum(grads_dot
                                  / (param_grads_norm * prior_grads_norm),
                                  torch.tensor(0.0))
        if not is_dry_run:
            offset_coef = -grads_cos * param_grads_norm / prior_grads_norm
            offset_scaled_(param_grads, prior_grads, offset_coef)
            offseted_param_grads_norm = norm_all_tensors(param_grads)
            offseted_grads_dot = dot_all_tensors(param_grads, prior_grads)
        logs.update({
                "grads_dot":
                float(grads_dot.item()),
                "grads_cos": float(grads_cos.item()),
                "inst/grads_norm":
                float(param_grads_norm.item()),
                "prior/grads_norm":
                float(prior_grads_norm.item()),
            })
        if not is_dry_run:
            logs.update({
                    "offseted_grads_dot":
                    float(offseted_grads_dot.item()),
                    "inst/offseted_grads_norm":
                    float(offseted_param_grads_norm.item()),
                })


def sequential_do(accelerator: Accelerator):
    def wrapper(function: Callable):
        def _inner_f(*args, **kargs):
            for i in accelerator.num_processes:
                accelerator.on_process(function, i)(*args, **kargs)
        return _inner_f
    return wrapper


def sequential_print(accelerator, *args, **kwargs):
    sequential_do(accelerator)(print)(*args, **kwargs)


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
        sync_steps = max(gradient_accumulation_steps,
                         args.local_sgd_steps)
        assert sync_steps % gradient_accumulation_steps == 0
        assert (args.local_sgd_steps == 0
                or sync_steps % args.local_sgd_steps == 0)
        self._profiler = profiler

        self._is_use_ref = 'reference' in args.noise_pred_target_method
        self._is_use_uncond_null = ('uncond_cancelled_by_null'
                                    in args.noise_pred_target_method)
        if self._is_use_uncond_null:
            self._unet_input_count = 3
        else:
            self._unet_input_count = 2

        self._stop_text_percentage = args.stop_text_encoder
        if not args.train_unet:
            self._stop_text_percentage = 1
        self._any_train_tenc = 0 != self._stop_text_percentage
        # verify_locon_installed(args)

        precision = args.mixed_precision if not shared.force_cpu else "no"
        self._build_accelerator(precision, logging_dir)
        print = self.accelerator.print

        args.max_token_length = int(args.max_token_length)
        if not args.pad_tokens and args.max_token_length > 75:
            print("Cannot raise token length limit ",
                  "above 75 when pad_tokens=False")

        self.accelerator.wait_for_everyone()
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

        self._optimizer_parameter_names = self._lookup_parameter_name(
            self._extract_optimizer_parameters())
        if self.accelerator.is_main_process:
            with open(os.path.join(args.model_dir, 
                                   'optimizer_parameters.json'),
                      'w', encoding='utf-8') as f:
                json.dump(self._optimizer_parameter_names, f, indent=4)

        # Train!
        max_train_steps = (self.args.num_train_epochs
                           * len(self.train_dataloader)
                           * self._total_batch_size)

        max_train_epochs = self.args.num_train_epochs
        # we calculate our number of tenc training epochs
        text_encoder_epochs = round(max_train_epochs
                                    * self._stop_text_percentage)

        print("  ***** Running training *****")
        if shared.force_cpu:
            print("  TRAINING WITH CPU ONLY")
        print(f"  Num batches each epoch = {len(self.train_dataloader)}")
        print(f"  Num Epochs = {max_train_epochs}")
        print(f"  Batch Size Per Device = {self._train_batch_size}")
        print("  Gradient Accumulation steps = "
              f"{self._gradient_accumulation_steps}")
        print("  Total train batch size (w. parallel, distributed) = "
              f"{self._total_batch_size}")
        print(f"  Text Encoder Epochs: {text_encoder_epochs}")
        print(f"  Total optimization steps = {sched_train_steps}")
        print(f"  Total training steps = {max_train_steps}")
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
        if self.args.use_lora and self._any_train_tenc:
            print("  LoRA Text Encoder LR: "
                  f"{self.args.lora_txt_learning_rate}")
        print(f"  V2: {self.args.v2}")

        self.accelerator.wait_for_everyone()

        self._inner_loop(max_train_steps,
                         max_train_epochs,
                         text_encoder_epochs)

    def _stop_profiler(self):
        if self._profiler is not None:
            try:
                print("Stopping profiler.")
                self._profiler.stop()
            except:
                pass

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
        self.text_encoder = torch2ify(self.text_encoder)

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
            if self._any_train_tenc:
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
            if self._any_train_tenc:
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

            print = self.accelerator.print
            scaling_factor = (self.accelerator.num_processes
                              * self._gradient_accumulation_steps)

            print('lr is scaled by num_processes '
                  '* gradient_accumulation_steps')
            print('lora_leerning_rate: ',
                  self.args.lora_learning_rate)
            print('lora_txt_learning_rate: ',
                  self.args.lora_txt_learning_rate)
            lora_learning_rate = (
                self.args.lora_learning_rate * scaling_factor)
            lora_txt_learning_rate = (
                self.args.lora_txt_learning_rate * scaling_factor)
            print('-> lora_leerning_rate: ',
                  lora_learning_rate)
            print('-> lora_txt_learning_rate: ',
                  lora_txt_learning_rate)

            learning_rate = lora_learning_rate

            print('Adam betas are scaled by num_processes '
                  '* gradient_accumulation_steps')
            print('adam_beta1: ', self.args.adam_beta1)
            print('adam_beta2: ', self.args.adam_beta2)
            scaling_factor *= self._train_batch_size
            adam_betas = (
                self.args.adam_beta1 ** scaling_factor,
                self.args.adam_beta2 ** scaling_factor)
            print('-> adam_beta1: ', adam_betas[0])
            print('-> adam_beta2: ', adam_betas[1])
        else:
            learning_rate = self.args.learning_rate
            adam_betas = (self.args.adam_beta1, self.args.adam_beta2)

        def build_params_to_optimize(lr_scale=1.0):
            if self.args.use_lora:
                if self._any_train_tenc:
                    return [
                        {
                            "params": unet_lora_params,
                            "lr": lora_learning_rate * lr_scale,
                            "betas": adam_betas,
                        },
                        {
                            "params": text_encoder_lora_params,
                            "lr": lora_txt_learning_rate * lr_scale,
                            "betas": adam_betas,
                        },
                    ]
                else:
                    params_to_optimize = unet_lora_params
            elif self._any_train_tenc:
                if self.args.train_unet:
                    params_to_optimize = itertools.chain(
                        self.unet.parameters(), self.text_encoder.parameters())
                else:
                    params_to_optimize = itertools.chain(
                        self.text_encoder.parameters())
            else:
                params_to_optimize = self.unet.parameters()
            return [
                {
                    "params": params_to_optimize,
                    "lr": learning_rate * lr_scale,
                    "betas": adam_betas,
                }
            ]

        self.optimizer, self._optimizer_kwargs = get_optimizer(
            self.args, build_params_to_optimize())
        if self.args.split_optimizer:
            self.instance_optimizer, self._instance_optimizer_kwargs =\
                get_optimizer(
                    self.args, build_params_to_optimize(
                        np.abs(self.args.instance_loss_weight)))

        self.noise_scheduler = get_noise_scheduler(self.args)

    def _lookup_parameter_name(self, parameters):
        unet_param_map = {p: name 
                          for name, p
                          in self.unet.named_parameters()}
        text_encoder_param_map = {p: name
                                  for name, p
                                  in self.text_encoder.named_parameters()}

        def lookup_name_of_parameter(p):
            if p in unet_param_map:
                return 'unet.' + unet_param_map[p]
            elif p in text_encoder_param_map:
                return 'te.' + text_encoder_param_map[p]
            else:
                return None
        return [lookup_name_of_parameter(p) for p in parameters]

    def _build_dataloader(self):
        instance_prompts, class_prompts = generate_classifiers(
            self.args, ui=False
        )
        n_workers = 0 if os.name == 'nt' else 1

        assert self.args.cache_latents
        printm("Created tenc")
        vae = create_vae(self.args)
        printm("Created vae")
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
            vae=vae,
            interleave_size=self.dataset_interleave_size
        )

        vae = torch2ify(vae)
        self._vae_encoder = VAEEncoder(vae)
        printm("Dataset loaded.")

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
            images = [example["image"] for example in examples]
            types = [example["is_class"] for example in examples]

            batch_data = {
                "prompt": prompts,
                "negative_prompt": negative_prompts,
                "guidance_scale": guidance_scales,
                "images": images,
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
        if self.args.split_optimizer:
            self.instance_lr_scheduler = UniversalScheduler(
                name=self.args.lr_scheduler,
                optimizer=self.instance_optimizer,
                num_warmup_steps=self.args.lr_warmup_steps,
                total_training_steps=sched_train_steps,
                min_lr=(self.args.learning_rate_min
                        * np.abs(self.args.instance_loss_weight)),
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
                              self.lr_scheduler]
        if self.args.use_ema:
            to_prepare_objects.append(self.ema_model.model)
        if self._any_train_tenc:
            to_prepare_objects.append(self.text_encoder)
        if self._is_use_ref:
            to_prepare_objects.append(self.unet_reference)
        if self.args.split_optimizer:
            to_prepare_objects.extend([self.instance_optimizer,
                                       self.instance_lr_scheduler])
        optimizer_type = type(self.optimizer)

        prepared_objects = self.accelerator.prepare(*to_prepare_objects)

        (self.unet,
         self.optimizer,
         self.lr_scheduler,
         *rs) = prepared_objects

        def sam_optimizer(optimizer):
            if 'SAM' in self.args.optimizer:
                from sam import SAM
                base_optimizer = optimizer
                self.accelerator.print('optimizer_type:', optimizer_type)
                optimizer = SAM(base_optimizer.param_groups,
                                optimizer_type,
                                rho=self.args.sam_rho,
                                adaptive='AdaptiveSAM' in self.args.optimizer,
                                **self._optimizer_kwargs)
                optimizer.base_optimizer = base_optimizer
                return optimizer
            else:
                return optimizer
        self.optimizer = sam_optimizer(self.optimizer)
        if self.args.use_ema:
            self.ema_model.model, *rs = rs
        if self._any_train_tenc:
            self.text_encoder, *rs = rs
        if self._is_use_ref:
            self.unet_reference, *rs = rs
        if self.args.split_optimizer:
            self.instance_optimizer, self.instance_lr_scheduler, *rs = rs
            self.instance_optimizer = sam_optimizer(self.instance_optimizer)
        assert rs == []

        self.train_dataloader = self.accelerator.prepare_data_loader(
            self.train_dataloader, device_placement=False)

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

    def _loss_fn(self, xs, ys, prod, is_instance,
                 kl_loss_gain=1.0, is_use_noise_prod_loss=False):
        prod = (prod if is_use_noise_prod_loss
                else torch.ones_like(prod))

        def loss_fun_batch(x, y, p):
            if ('instance_sqrt_mse' in self.args.loss_function_method
                    and is_instance):
                epsilon = 1e-4
                if on_first_call():
                    print('instance_sqrt_mse in',
                          self.args.loss_function_method,
                          is_instance)
                return p * torch.sqrt(torch.nn.functional.mse_loss(
                    xs.float(), ys.float(), reduction="mean"
                ) + epsilon)
            if 'instance_l1' in self.args.loss_function_method and is_instance:
                if on_first_call():
                    print('instance_l1 in',
                          self.args.loss_function_method,
                          is_instance)
                return p * torch.nn.functional.l1_loss(
                    xs.float(), ys.float(), reduction="mean"
                )
            elif ('instance_smoothl1sqrtmse' in self.args.loss_function_method
                  and is_instance):
                if on_first_call():
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
                if on_first_call():
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
                if on_first_call():
                    print('class_sqrt_mse in',
                          self.args.loss_function_method, is_instance)
                return p * torch.sqrt(torch.nn.functional.mse_loss(
                    xs.float(), ys.float(), reduction="mean"
                ) + epsilon)
            elif ('class_l1' in self.args.loss_function_method
                  and not is_instance):
                epsilon = 1e-4
                if on_first_call():
                    print('class_l1 in',
                          self.args.loss_function_method, is_instance)
                return p * torch.nn.functional.l1_loss(
                    xs.float(), ys.float(), reduction="mean")
            if on_first_call():
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

    def _extract_optimizer_parameters(self, is_check_grad=False):
        extract_optimizer_parameters(self.optimizer, is_check_grad)

    class NoSyncParametersPseudoModel:
        def __init__(self, parameters, no_sync_models):
            self._parameters = list(parameters)
            self._no_sync_models = no_sync_models

        def parameters(self):
            return self._parameters

        @contextlib.contextmanager
        def no_sync(self):
            with contextlib.ExitStack() as stack:
                for model in self._no_sync_models:
                    if hasattr(model, "no_sync"):
                        stack.enter_context(model.no_sync())
                yield

    def _clip_grad_norm(self, train_tenc, logs):
        if self.args.clip_grad_norm == 0.0:
            return None

        if self.args.use_lora:
            params_to_clip = self._extract_optimizer_parameters()
            cliped_grad_norm = self.accelerator.clip_grad_norm_(
                params_to_clip,
                self.args.clip_grad_norm)
        else:
            if train_tenc:
                params_to_clip = itertools.chain(
                    self.unet.parameters(), self.text_encoder.parameters())
            else:
                params_to_clip = self.unet.parameters()
            cliped_grad_norm = self.accelerator.clip_grad_norm_(
                params_to_clip, self.args.clip_grad_norm)
        logs.update({
                "cliped_grad_norm":
                float(cliped_grad_norm.item()),
            })
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
            if on_first_call():
                print('guidance_scale_batch:',
                      guidance_scale_batch)
                print('guidance_scale_at_step_maximum:',
                      guidance_scale_at_step_maximum)
                print('guidance_scale_at_step:',
                      guidance_scale_at_step)
        return guidance_scale_at_step

    def _calc_embeds(self, batch, train_tenc):
        if on_first_call():
            for prompt, negative_prompt in zip(
                    batch['prompt'], batch['negative_prompt']):
                print('prompt:', prompt)
                print('negative prompt:', negative_prompt)

        def build_embed(prompt):
            if not self._any_train_tenc:
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
            if not self._any_train_tenc:
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

        return encoder_hidden_states_input

    def _calc_unet(self,
                   noisy_latents,
                   timesteps,
                   encoder_hidden_states_input):
        timesteps_input = torch.concat(
            [timesteps] * self._unet_input_count)
        noisy_latents_input = torch.cat(
            [noisy_latents] * self._unet_input_count)
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
                       is_priors):
        def filter_chunk(is_prior):
            xss = list(zip(*[xs for xs
                             in zip(pred,
                                    target,
                                    noise_prods,
                                    is_priors)
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

    class SaveChecker:
        def __init__(self, args, parent, max_train_epochs):
            self.args = args
            self._parent = parent
            self._last_model_save = 0
            self._last_image_save = 0
            self._max_train_epochs = max_train_epochs

        def checked_save(self, epoch, is_epoch_check=False):
            save_model_interval = self.args.save_embedding_every
            save_image_interval = self.args.save_preview_every
            save_completed = epoch >= self._max_train_epochs
            save_canceled = status.interrupted
            save_image = False
            save_model = False
            if not save_canceled and not save_completed:
                # Check to see if the number of epochs
                # since last save is gt the interval
                if (0 < save_model_interval
                        <= epoch - self._last_model_save):
                    save_model = True
                    self._last_model_save = epoch

                # Repeat for sample images
                if (0 < save_image_interval
                        <= epoch - self._last_image_save):
                    save_image = True
                    self._last_image_save = epoch

            else:
                print("\nSave completed/canceled.")
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
                    print("Canceled, enabling saves.")
                    save_lora = self.args.save_lora_cancel
                    save_snapshot = self.args.save_state_cancel
                    save_checkpoint = self.args.save_ckpt_cancel
                elif save_completed:
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
                self._parent.save_weights(
                    save_image,
                    save_model,
                    save_snapshot,
                    save_checkpoint,
                    save_lora,
                )
            return save_model

    def _bulid_noise(self, latents):
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
        return noise

    def _optimizer_step(self,
                        is_instance_batch,
                        is_first_step, is_second_step):
        optimizer = (self.instance_optimizer
                     if self.args.split_optimizer and is_instance_batch
                     else self.optimizer)
        lr_scheduler = (self.instance_lr_scheduler
                     if self.args.split_optimizer and is_instance_batch
                     else self.lr_scheduler)
        if is_first_step:
            optimizer.first_step()
        elif is_second_step:
            optimizer.second_step()
            lr_scheduler.step()
        else:
            optimizer.step()
            lr_scheduler.step()

    class GradModifier:
        def __init__(self, args, accelerator):
            self.accelerator = accelerator
            self.args = args
            self._prior_grads = []
            self._instance_grads = []
            self._ema_prior_grad_norms = []
            self._ema_instance_grad_norms = []
            self._ema_beta = 0.01

        @torch.no_grad()
        def _save_params(self, filename, params, param_grads):
            if self.accelerator.is_main_process:
                torch.save(
                    (params, param_grads),
                    os.path.join(
                        self.args.model_dir,
                        'logging',
                        filename))

        def modify_grads(self,
                         params,
                         param_grads,
                         is_instance_batch,
                         counter,
                         logs):
            def save_params(filename):
                self._save_params(filename, params, param_grads)

            if self.args.split_optimizer:
                file_name_prefix = ('instance_'
                                    if is_instance_batch
                                    else 'prior_')
            else:
                file_name_prefix = ''
            if self.args.save_params_step:
                save_params(
                    f'{file_name_prefix}params_{counter.epoch}'
                    f'_{counter.step}.pt')

            if self.args.split_optimizer:
                if is_instance_batch:
                    pass
                    # try_copy(param_grads, self._instance_grads)
                else:
                    try_copy(param_grads, self._prior_grads)

                if (is_instance_batch
                        and len(param_grads) > 0
                        and len(self._prior_grads) > 0):
                    if self.args.split_optimizer_without_offset:
                        if on_first_call():
                            print('split_optimizer/offset_by_grad_ is dry_run')
                    offset_by_grad_(
                        param_grads,
                        self._prior_grads,
                        logs,
                        is_dry_run=self.args.split_optimizer_without_offset)

    class LoopCounter:
        def __init__(self, args, max_epochs, step_increment, global_step=0):
            self.args = args
            self._max_epochs = max_epochs
            self._step_increment = step_increment

            self.epoch = 0
            self.global_step = global_step
            self.step = 0

        def __iter__(self):
            init_args_epoch = self.args.epoch
            for epoch in range(self._max_epochs):
                self.step = 0
                self.epoch = epoch
                self.args.epoch = init_args_epoch + epoch
                yield epoch

        def __len__(self):
            return self._max_epochs

        @contextlib.contextmanager
        def step_context(self):
            self.args.revision += self._step_increment
            try:
                yield
            finally:
                self.step += self._step_increment
                self.global_step += self._step_increment

    def _fetch_latents(self, batch):
        def maybe_encode(maybe_latent):
            # is_latent, tensor = maybe_latent
            is_latent = maybe_latent.shape[0] == 4
            tensor = maybe_latent
            if is_latent:
                return tensor.to(self.accelerator.device)
            else:
                return self._vae_encoder.encode(tensor.unsqueeze(0)).squeeze(0)
        latents = torch.stack([maybe_encode(maybe_latent)
                               for maybe_latent
                               in batch["images"]])
        return latents

    class TrainingCompleteException(Exception):
        def __init__(self):
            pass

    def _inner_loop(self,
                    max_train_steps,
                    max_train_epochs,
                    text_encoder_epochs,
                    null_guidance_scale=0.0,
                    fft_method=None,
                    train_steps=100):

        @contextlib.contextmanager
        def slave_sync(model):
            if self.accelerator.sync_gradients:
                context = contextlib.nullcontext
            else:
                context = self.accelerator.no_sync

            with context(model):
                yield

        if not self._any_train_tenc:
            self._embeds_cache = {}
            self.text_encoder.to(self.accelerator.device,
                                 dtype=self._weight_dtype)
        # Afterwards we recalculate our number of training epochs
        # We need to initialize the trackers we use, and also store our
        # configuration.
        # The trackers will initialize automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth")

        save_checker = self.SaveChecker(self.args, self, max_train_epochs)

        # for debug?
        # os.environ.__setattr__("CUDA_LAUNCH_BLOCKING", 1)

        # Only show the progress bar once on each machine.
        progress_bar = mytqdm(
            range(max_train_steps),
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
        status.job_count = max_train_steps
        status.job_no = 0
        msg = ""

        last_tenc = 0 < text_encoder_epochs
        if not self._any_train_tenc:
            last_tenc = False

        self.noise_scheduler.set_timesteps(
            train_steps, device=self.accelerator.device)

        if self.accelerator.num_processes == 1:
            self._build_hist()

        l2_regularization = L2Regularization(self.args)
        ewc = EWC(self.args, self.accelerator.device)

        counter = self.LoopCounter(
            self.args, max_train_epochs, self._total_batch_size)
        grad_modifier = self.GradModifier(self.args, self.accelerator)
        for epoch in counter:
            if self.args.num_train_epochs > 1:
                if epoch >= max_train_epochs:
                    raise self.TrainingCompleteException()

            if self.accelerator.is_main_process:
                save_checker.checked_save(
                    counter.epoch,
                    is_epoch_check=True)

            if self.args.train_unet:
                self.unet.train()

            train_tenc = epoch < text_encoder_epochs
            if not self._any_train_tenc:
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
                self.args, current_epoch=epoch
            )

            param_stat_save = ParamStatSave(
                os.path.join(self.args.model_dir, 'logging'),
                epoch) if self.args.save_params_stat else None

            def loop_step(counter, batch,
                          latents, timesteps, noise,
                          is_first_step=False,
                          is_second_step=False):
                logs = {}
                # Add noise to the latents according
                # to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps)

                encoder_hidden_states_input =\
                    self._calc_embeds(batch, train_tenc)

                (noise_pred_uncond,
                    noise_pred_text,
                    noise_pred_uncond_null,
                    noise_pred_output_reference) =\
                    self._calc_unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states_input
                    )

                guidance_scale_at_step = self._calc_guidance_scale(
                    counter.global_step, batch)

                noise_pred = (noise_pred_uncond
                              + guidance_scale_at_step
                              * (noise_pred_text - noise_pred_uncond))

                reference_stats = self._calc_reference_stat(
                    noise,
                    noise_pred_output_reference,
                    guidance_scale_at_step)

                # Get the target for loss depending on the prediction type
                assert (self.noise_scheduler.config.prediction_type
                        != "v_prediction")
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

                assert self.args.split_loss
                (instance_outputs,
                 prior_outputs) = self._split_outputs(
                  pred,
                  target,
                  noise_prods,
                  batch['types'])
                is_instance_batch = instance_outputs[0] is not None
                is_prior_batch = prior_outputs[0] is not None
                # Concatenate the chunks in instance_chunks
                # to form the model_pred_instance tensor
                if is_instance_batch:
                    instance_loss = self._loss_fn(
                        *instance_outputs,
                        True)
                    logs.update({
                            "inst/loss":
                            float(instance_loss.detach().item()),
                        })
                else:
                    instance_loss = torch.tensor(0.0)

                if is_prior_batch:
                    prior_loss = self._loss_fn(
                        *prior_outputs,
                        False)
                    logs.update({
                            "prior/loss":
                            float(prior_loss.detach().item()),
                        })
                else:
                    prior_loss = torch.tensor(0.0)

                instance_loss_weight = (np.sign(self.args.instance_loss_weight)
                                        if self.args.split_optimizer
                                        else self.args.instance_loss_weight)
                loss = (instance_loss_weight * instance_loss
                        + current_prior_loss_weight * prior_loss)

                loss += l2_regularization.calc_loss(
                    self._extract_optimizer_parameters(),
                    logs)

                loss += ewc.calc_loss(
                    self._extract_optimizer_parameters(),
                    logs)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    if (self.args.split_optimizer
                            or self.args.save_params_step
                            or self.args.save_params_stat):
                        params = self._extract_optimizer_parameters(True)
                        param_grads = [p.grad for p in params]
                        if param_stat_save is not None:
                            param_stat_save.step(params, param_grads)
                    else:
                        params = None
                        param_grads = None

                    grad_modifier.modify_grads(
                        params=params,
                        param_grads=param_grads,
                        is_instance_batch=is_instance_batch,
                        counter=counter,
                        logs=logs,
                    )

                    l2_regularization.update_grad_norm(
                        self._clip_grad_norm(train_tenc, logs))

                    self._optimizer_step(is_instance_batch,
                                         is_first_step,
                                         is_second_step)

                    if self.args.use_ema and self.ema_model is not None:
                        self.ema_model.step(self.unet)
                    if self._profiler is not None:
                        self._profiler.step()
                    self.optimizer.zero_grad(
                        set_to_none=self.args.gradient_set_to_none)

                allocated, cached = get_vram_used()
                last_lr = self.lr_scheduler.get_last_lr()[0]

                status.job_no += self._total_batch_size

                # logs_to_histogram = {
                #     'pred': pred.detach().cpu(),
                #     'target': target.detach().cpu(),
                #     'target_pred_dif': (target.detach().cpu()
                #                         - pred.detach().cpu())}

                loss_step = loss.detach().item()
                assert self.args.split_loss
                logs.update({
                    "epoch": epoch,
                    "lr": float(last_lr),
                    "loss": float(loss_step),
                    "vram": float(cached),
                })
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
                status.job_no = counter.global_step
                status.textinfo = (
                    f"Steps: {counter.global_step}"
                    f"/{max_train_steps} (Current),"
                    f" {self.args.revision}/{max_train_steps}"
                    f" (Lifetime), Epoch: {epoch}"
                )

                if math.isnan(loss_step):
                    self.accelerator.print(
                        "Loss is NaN, your model is dead."
                        " Cancelling training.")
                    raise InterruptedError()

                if status.interrupted:
                    raise InterruptedError()

                return loss_step
            param_model = self.NoSyncParametersPseudoModel(
                self._extract_optimizer_parameters(),
                [self.unet, self.text_encoder])
            is_local_sgd_enabled = self.args.local_sgd_steps > 0
            local_sgd_context = LocalSGD(
                accelerator=self.accelerator,
                model=param_model,
                local_sgd_steps=self.args.local_sgd_steps,
                enabled=is_local_sgd_enabled,
            )
            if is_local_sgd_enabled:
                self.accelerator.print(
                    'Local SGD is enabled:',
                    self.args.local_sgd_steps)
            try:
                if 'SAM' in self.args.optimizer:
                    repeat_count = self._gradient_accumulation_steps
                    if on_first_call():
                        self.accelerator.print('SAM rho:', self.args.sam_rho)
                        self.accelerator.print(
                            'SAM repeat_count:', repeat_count)
                        self.accelerator.wait_for_everyone()
                    # for synchronize sampler
                    batches = []
                    timestepss = []
                    latentss = []
                    noises = []

                    def accum_step(is_first_step, is_second_step, local_sgd):
                        nonlocal loss_total
                        for batch, latents, timesteps, noise in zip(
                                batches, latentss, timestepss, noises):
                            with (self.accelerator.accumulate(None),
                                    slave_sync(self.unet),
                                    slave_sync(self.text_encoder),
                                    counter.step_context()):
                                loss_total += loop_step(
                                    counter, batch,
                                    latents=latents,
                                    timesteps=timesteps,
                                    noise=noise,
                                    is_first_step=is_first_step,
                                    is_second_step=is_second_step)
                                if is_second_step:
                                    local_sgd.step()
                    with local_sgd_context as local_sgd:
                        syncronize_python_rng_state(self.accelerator)
                        for step, batch in enumerate(self.train_dataloader):
                            # flush step
                            if self.accelerator.gradient_state.end_of_dataloader:
                                with self.accelerator.accumulate(None):
                                    pass
                                break
                            # collect batches for accumulation
                            # Sample a random timestep for each image
                            timesteps = torch.randint(
                                0,
                                self.noise_scheduler.config.num_train_timesteps,
                                (self._train_batch_size,),
                                device=self.accelerator.device,
                            ).long()
                            latents = self._fetch_latents(batch)
                            # Sample noise that we'll add to the latents
                            noise = self._bulid_noise(latents)

                            batches.append(batch)
                            timestepss.append(timesteps)
                            latentss.append(latents)
                            noises.append(noise)

                            if step % repeat_count == repeat_count - 1:
                                if accum_step(True, False, local_sgd):
                                    break
                                if accum_step(False, True, local_sgd):
                                    break
                                batches = []
                                timestepss = []
                                latentss = []
                                noises = []
                else:
                    with local_sgd_context as local_sgd:
                        # for synchronize sampler
                        syncronize_python_rng_state(self.accelerator)
                        for batch in self.train_dataloader:
                            # Sample a random timestep for each image
                            timesteps = torch.randint(
                                0,
                                self.noise_scheduler.config.num_train_timesteps,
                                (self._train_batch_size,),
                                device=self.accelerator.device,
                            ).long()
                            latents = self._fetch_latents(batch)
                            # Sample noise that we'll add to the latents
                            noise = self._bulid_noise(latents)

                            with (self.accelerator.accumulate(None),
                                  slave_sync(self.unet),
                                  slave_sync(self.text_encoder),
                                  counter.step_context()):
                                loss_total += loop_step(
                                    counter, batch,
                                    latents=latents,
                                    timesteps=timesteps,
                                    noise=noise)
                                local_sgd.step()

                if param_stat_save is not None:
                    param_stat_save.epoch_end()

                logs = {"epoch_loss": loss_total / counter.step}
                self.accelerator.log(logs, step=counter.global_step)

                # self.accelerator.wait_for_everyone()

                self.lr_scheduler.step(is_epoch=True)
                status.job_count = max_train_steps
                status.job_no = counter.global_step

            except (self.TrainingCompleteException, InterruptedError) as e:
                print("  Training complete (step check).")
                if isinstance(e, InterruptedError):
                    state = "cancelled"
                else:
                    state = "complete"

                status.textinfo = (
                    f"Training {state} {counter.global_step}/{max_train_steps},"
                    f" {self.args.revision}"
                    f" total."
                )
                break

        self.accelerator.end_training()
        self.result.msg = msg
        self.result.config = self.args
        self.result.samples = last_samples
        self._stop_profiler()

    def save_weights(
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
                            if self._any_train_tenc:
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
