# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict

import torch
from omegaconf import DictConfig

from torch import nn

from torchtune import config, utils

from torchtune.custom_generate_func import *  # UPDATED

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        from torchtune.models.convert_weights import _FROM_META  # UPDATED
        if 'biomedclip' in cfg.get('vision', ''):  # UPDATED
            import open_clip  # UPDATED
            model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device=self._device)  # UPDATED
            for k, _ in model_clip.visual.named_parameters():  # UPDATED
                _FROM_META[f'visual.{k}'] = f'visual.{k}'  # UPDATED
            _FROM_META['projector.0.weight'] = 'projector.0.weight'  # UPDATED
            _FROM_META['projector.0.bias'] = 'projector.0.bias'  # UPDATED
            _FROM_META['projector.2.weight'] = 'projector.2.weight'  # UPDATED
            _FROM_META['projector.2.bias'] = 'projector.2.bias'  # UPDATED
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
            cfg=cfg,  # UPDATED
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        cfg  # UPDATED
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)
            if 'biomedclip' in cfg.get('vision', ''):  # Before the state_dict is loaded  # UPDATED
                import open_clip  # UPDATED
                model.visual = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', device=self._device)[0].visual  # UPDATED
                model.projector = nn.Sequential(nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 4096 * 32), )  # UPDATED

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    @torch.no_grad()
    def generate(self, cfg: DictConfig, feat=None):  # UPDATED
        tokens = self._tokenizer.encode(cfg.prompt, add_bos=True, add_eos=False)
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                utils.generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = utils.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                eos_id=[self._tokenizer._encode_special_token(t) for t in self._tokenizer.all_special_tokens],  # UPDATED
                custom_generate_next_token=custom_generate_next_token,
                feat=feat,  # UPDATED
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            eos_id=[self._tokenizer._encode_special_token(t) for t in self._tokenizer.all_special_tokens],  # UPDATED
            custom_generate_next_token=custom_generate_next_token,
            feat=feat,  # UPDATED
        )

        # ... UPDATED

        return self._tokenizer.decode(generated_tokens)


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    ##################
    # UPDATED
    if cfg.get('test_metrics', False):
        test_metrics(recipe, cfg)
    if cfg.get('test_vqarad', False):
        test_vqarad(recipe, cfg)
    ##################


if __name__ == "__main__":
    sys.exit(main())
