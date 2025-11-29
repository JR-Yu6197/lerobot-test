#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Any, TypedDict

import torch
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
# [수정] PolicyFeature 추가
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
# [수정] utils import 제거! (함수를 직접 정의함으로 해결)
# from lerobot.policies.utils import validate_visual_features_consistency 
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


# -----------------------------------------------------------------------------
# [추가] 순환 참조 방지를 위해 검증 함수를 여기에 직접 정의합니다.
# -----------------------------------------------------------------------------
def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    Raises a standardized ValueError for feature mismatches between dataset/environment and policy config.
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument."
    )

def validate_visual_features_consistency(
    cfg: PreTrainedConfig,
    features: dict[str, PolicyFeature],
) -> None:
    """
    Validates visual feature consistency between a policy config and provided dataset/environment features.
    """
    expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}
    if not provided_visuals.issubset(expected_visuals):
        raise_feature_mismatch_error(provided_visuals, expected_visuals)
# -----------------------------------------------------------------------------


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        return PI0Policy
    elif name == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy
        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
        return Classifier
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAPolicy
    elif name == "groot":
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        return GrootPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi05":
        return PI05Config(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    elif policy_type == "groot":
        return GrootConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


class ProcessorConfigKwargs(TypedDict, total=False):
    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if pretrained_path:
        if isinstance(policy_cfg, GrootConfig):
            preprocessor_overrides = {}
            postprocessor_overrides = {}
            preprocessor_overrides["groot_pack_inputs_v3"] = {
                "stats": kwargs.get("dataset_stats"),
                "normalize_min_max": True,
            }
            env_action_dim = policy_cfg.output_features["action"].shape[0]
            postprocessor_overrides["groot_action_unpack_unnormalize_v1"] = {
                "stats": kwargs.get("dataset_stats"),
                "normalize_min_max": True,
                "env_action_dim": env_action_dim,
            }
            kwargs["preprocessor_overrides"] = preprocessor_overrides
            kwargs["postprocessor_overrides"] = postprocessor_overrides

        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # ... (기존 프로세서 로직 유지) ...
    # Create a new processor based on policy type
    if isinstance(policy_cfg, TDMPCConfig):
        from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_pre_post_processors
        processors = make_tdmpc_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, DiffusionConfig):
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
        processors = make_diffusion_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, ACTConfig):
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
        processors = make_act_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, VQBeTConfig):
        from lerobot.policies.vqbet.processor_vqbet import make_vqbet_pre_post_processors
        processors = make_vqbet_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, PI0Config):
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
        processors = make_pi0_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, PI05Config):
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        processors = make_pi05_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, SACConfig):
        from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors
        processors = make_sac_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, RewardClassifierConfig):
        from lerobot.policies.sac.reward_model.processor_classifier import make_classifier_processor
        processors = make_classifier_processor(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, SmolVLAConfig):
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        processors = make_smolvla_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    elif isinstance(policy_cfg, GrootConfig):
        from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
        processors = make_groot_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
    else:
        raise NotImplementedError(f"Processor for policy type '{policy_cfg.type}' is not implemented.")

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.
    """
    # [수정] 내부 import도 필요 없음 (위에서 정의했으므로)
    
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    if not cfg.output_features:
        cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    if not rename_map:
        # [수정] 여기서 파일 내부의 함수를 호출 (외부 의존성 제거)
        validate_visual_features_consistency(cfg, features)

    return policy