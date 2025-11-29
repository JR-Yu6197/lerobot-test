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

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor import PolicyAction, RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR

# ⚠️ [수정] 순환 참조를 일으키는 factory import 제거됨

def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """Log missing and unexpected keys when loading a model.

    Args:
        missing_keys (list[str]): Keys that were expected but not found.
        unexpected_keys (list[str]): Keys that were found but not expected.
    """
    if missing_keys:
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")


# TODO(Steven): Move this function to a proper preprocessor step
def prepare_observation_for_inference(
    observation: dict[str, np.ndarray],
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """Converts observation data to model-ready PyTorch tensors.

    This function takes a dictionary of NumPy arrays, performs necessary
    preprocessing, and prepares it for model inference. The steps include:
    1. Converting NumPy arrays to PyTorch tensors.
    2. Normalizing and permuting image data (if any).
    3. Adding a batch dimension to each tensor.
    4. Moving all tensors to the specified compute device.
    5. Adding task and robot type information to the dictionary.

    Args:
        observation: A dictionary mapping observation names (str) to NumPy
            array data. For images, the format is expected to be (H, W, C).
        device: The PyTorch device (e.g., 'cpu' or 'cuda') to which the
            tensors will be moved.
        task: An optional string identifier for the current task.
        robot_type: An optional string identifier for the robot being used.

    Returns:
        A dictionary where values are PyTorch tensors preprocessed for
        inference, residing on the target device. Image tensors are reshaped
        to (C, H, W) and normalized to a [0, 1] range.
    """
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name]