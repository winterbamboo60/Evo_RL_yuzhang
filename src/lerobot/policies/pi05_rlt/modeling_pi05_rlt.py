from __future__ import annotations

import logging
from typing import Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    PI05Pytorch,
    _build_flow_matching_inputs,
    _prepare_trained_rtc_prefix,
    _reduce_training_rtc_loss,
    _sample_training_rtc_prefix_mask,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_pi05_rlt import PI05RLTConfig
from .rlt_token_transformer import RLTTokenTransformer


class PI05RLTPytorch(PI05Pytorch):
    """The current PI0.5 core plus an RL-token encoder/decoder."""

    def __init__(self, config: PI05RLTConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        rlt_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.rlt_module = RLTTokenTransformer(
            input_dim=config.rlt_input_dim,
            embed_dim=config.rlt_embed_dim,
            num_rl_tokens=config.rlt_num_rl_tokens,
            prefix_seq_len=config.rlt_prefix_seq_len,
            num_layers=config.rlt_num_layers,
            num_heads=config.rlt_num_heads,
            mlp_ratio=config.rlt_mlp_ratio,
            dropout_rate=config.rlt_dropout_rate,
        ).to(dtype=rlt_dtype)

    def _select_rlt_prefix_embeddings(
        self,
        prefix_output: Tensor,
        prefix_pad_masks: Tensor,
        num_image_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        if self.config.rlt_image_only:
            prefix_output = prefix_output[:, :num_image_tokens]
            prefix_pad_masks = prefix_pad_masks[:, :num_image_tokens]
        return prefix_output, prefix_pad_masks

    def encode_rlt(
        self,
        prefix_output: Tensor,
        prefix_pad_masks: Tensor,
        num_image_tokens: int,
    ) -> Tensor:
        prefix_output, prefix_pad_masks = self._select_rlt_prefix_embeddings(
            prefix_output, prefix_pad_masks, num_image_tokens
        )
        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = prefix_pad_masks if self.config.rlt_use_mask else None
        return self.rlt_module.encode_flat(prefix_output, rlt_mask).to(dtype=torch.float32)

    def forward_with_rlt_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        prefix_mask: Tensor | None = None,
        states=None,
        state_masks=None,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Run the current PI0.5 training path and return its contextual prefix once."""
        x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, model_time)

        backbone_dtype = (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )
        if backbone_dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix, suffix, attention, positions, condition):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attention,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, condition],
            )
            return prefix_out, suffix_out

        prefix_output, suffix_output = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_output = suffix_output[:, -self.config.chunk_size :].to(dtype=torch.float32)
        velocity = self._apply_checkpoint(self.action_out_proj, suffix_output)

        proprio_tokens = 0
        if self.proprio_history_proj is not None and states is not None:
            proprio_tokens = states.shape[1]
        num_image_tokens = prefix_embs.shape[1] - tokens.shape[1] - proprio_tokens
        losses = F.mse_loss(u_t, velocity, reduction="none")
        return losses, prefix_output, prefix_pad_masks, num_image_tokens

    @torch.no_grad()
    def sample_actions_and_rlt(
        self,
        images,
        img_masks,
        tokens,
        masks,
        states=None,
        state_masks=None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> dict[str, Tensor]:
        """Produce the action chunk and z_rl from one shared current-PI0.5 prefix pass."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        batch_size = tokens.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim), tokens.device
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        backbone_dtype = (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )
        prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        prefix_attention = prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=prefix_positions,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        proprio_tokens = 0
        if self.proprio_history_proj is not None and states is not None:
            proprio_tokens = states.shape[1]
        num_image_tokens = prefix_embs.shape[1] - tokens.shape[1] - proprio_tokens
        z_rl = self.encode_rlt(prefix_output, prefix_pad_masks, num_image_tokens)

        rtc_mode = "guided"
        trained_prefix = trained_prefix_mask = None
        if self._rtc_enabled():
            rtc_mode = self.rtc_processor.rtc_config.mode
            if rtc_mode == "trained":
                trained_prefix, trained_prefix_mask = _prepare_trained_rtc_prefix(
                    noise,
                    kwargs.get("prev_chunk_left_over"),
                    int(kwargs.get("inference_delay") or 0),
                    int(self.config.rtc_training_max_delay),
                )

        actions = euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled() and rtc_mode == "guided",
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
            hard_prefix=trained_prefix,
            hard_prefix_mask=trained_prefix_mask,
        )
        return {"actions": actions, "z_rl": z_rl}


class PI05RLTPolicy(PI05Policy):
    """A separately registered, native LeRobot PI0.5+RLT policy."""

    config_class = PI05RLTConfig
    name = "pi05_rlt"

    def __init__(self, config: PI05RLTConfig, **kwargs):
        del kwargs
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05RLTPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Permit a plain PI0.5 checkpoint to initialize only the new RLT parameters."""
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        disallowed_missing = [key for key in result.missing_keys if not key.startswith("model.rlt_module.")]
        if strict and (disallowed_missing or result.unexpected_keys):
            raise RuntimeError(
                "PI05-RLT checkpoint mismatch: "
                f"missing non-RLT keys={disallowed_missing}, unexpected keys={result.unexpected_keys}"
            )
        if result.missing_keys and not disallowed_missing:
            logging.warning("Initialized %d RLT checkpoint tensors from scratch.", len(result.missing_keys))
        return result

    @torch.no_grad()
    def predict_action_chunk_with_rlt(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> dict[str, Tensor]:
        self.eval()
        has_temporal_input = any(
            key in batch and batch[key].ndim == 5 for key in self.config.image_features
        ) or (OBS_STATE in batch and batch[OBS_STATE].ndim == 3)
        if (self.config.use_visual_memory or self.config.use_proprioceptive_memory) and not has_temporal_input:
            batch = self._stack_inference_memory(batch)

        images, img_masks = self._preprocess_images(batch)
        states, state_masks = self._prepare_memory_states(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        outputs = self.model.sample_actions_and_rlt(
            images,
            img_masks,
            tokens,
            masks,
            states=states,
            state_masks=state_masks,
            **kwargs,
        )
        action_dim = self.config.output_features[ACTION].shape[0]
        outputs["actions"] = outputs["actions"][:, :, :action_dim]
        return outputs

    @torch.no_grad()
    def extract_rlt_features(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = self.predict_action_chunk_with_rlt(batch)
        proprio = batch[OBS_STATE]
        if proprio.ndim == 3:
            proprio = proprio[:, -1]
        return {"z_rl": outputs["z_rl"], "ref_action": outputs["actions"], "proprio": proprio}

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        images, img_masks = self._preprocess_images(batch)
        states, state_masks = self._prepare_memory_states(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        prefix_mask = _sample_training_rtc_prefix_mask(
            actions.shape[0],
            actions.shape[1],
            self.config.rtc_training_max_delay,
            actions.device,
        )
        losses, prefix_output, rlt_mask, num_image_tokens = self.model.forward_with_rlt_prefix(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            prefix_mask=prefix_mask,
            states=states,
            state_masks=state_masks,
        )
        action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :action_dim]
        per_sample_vla = _reduce_training_rtc_loss(losses, prefix_mask, reduction="none")
        vla_loss = per_sample_vla.mean()

        prefix_output, rlt_mask = self.model._select_rlt_prefix_embeddings(
            prefix_output, rlt_mask, num_image_tokens
        )
        rlt_param = next(self.model.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        mask_for_rlt = rlt_mask if self.config.rlt_use_mask else None
        per_sample_rlt, rlt_info = self.model.rlt_module(prefix_output, mask_for_rlt, reduction="none")
        per_sample_loss = per_sample_rlt + self.config.rlt_alpha * per_sample_vla

        if prefix_mask is None:
            loss_per_dim = losses.mean(dim=(0, 1))
        else:
            postfix_mask = (~prefix_mask).unsqueeze(-1).expand_as(losses)
            loss_per_dim = (losses * postfix_mask).sum(dim=(0, 1)) / postfix_mask.sum(
                dim=(0, 1)
            ).clamp(min=1)
        loss_dict = {
            "loss": per_sample_loss.mean().item(),
            "vla_loss": vla_loss.item(),
            "rlt_loss": rlt_info["mse"].item(),
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
        }
        if reduction == "none":
            return per_sample_loss, loss_dict
        return per_sample_loss.mean(), loss_dict

    def _get_default_peft_targets(self) -> dict[str, object]:
        targets = super()._get_default_peft_targets()
        modules_to_save = list(targets.get("modules_to_save", []))
        if "model.rlt_module" not in modules_to_save:
            modules_to_save.append("model.rlt_module")
        targets["modules_to_save"] = modules_to_save
        return targets
