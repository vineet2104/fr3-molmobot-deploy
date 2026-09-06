"""Standalone bridge policy for serving a MolmoBot SynthManip checkpoint.

Wraps `SynthManipMolmoInferenceWrapper` with action-chunk buffering so it can be
served over the existing `WebsocketPolicyServer`. Identical behavior whether the
client is a simulator or a real robot -- both just send the same observation dict.

Design goals:
  * No `molmo_spaces` / sim dependency (unlike olmo.eval.configure_real_robot).
  * Correct settings for the single-camera, absolute-joint (`joint_pos`),
    1-dim-gripper pilot model.
  * Predict `action_horizon` (16) actions, execute `execute_horizon` (8) open-loop,
    then re-query -- the server calls `get_action(obs)` once per control step.

Observation dict (sent by client each control step), msgpack-numpy encoded:
  {
    "<camera_name>": np.ndarray (H, W, 3) uint8 RGB,   # one key per camera_name
    "qpos": {"arm": np.ndarray (7,), "gripper": np.ndarray (>=1,)},
    "task": "Pick up the pineapple slices can",         # optional; falls back to default_task
  }

Returned action dict (per control step):
  {
    "arm": np.ndarray (7,),       # absolute joint position targets (radians)
    "gripper": np.ndarray (1,),   # gripper target
    "server_timing": {...},       # added by the server
  }
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BridgePolicyConfig:
    checkpoint_path: str
    camera_names: List[str] = field(default_factory=lambda: ["exo_front"])
    action_move_group_names: List[str] = field(default_factory=lambda: ["arm", "gripper"])
    action_spec: Dict[str, int] = field(default_factory=lambda: {"arm": 7, "gripper": 1})
    action_type: str = "joint_pos"            # ABSOLUTE joint positions (NOT joint_pos_rel)
    action_horizon: int = 16                   # actions predicted per chunk
    execute_horizon: int = 8                   # actions executed open-loop before re-query
    gripper_representation_count: int = 1      # gripper state dims fed to the model
    clamp_gripper: bool = False                # if True, binarize gripper output (>128 -> 255 else 0)
    default_task: str = "Pick up the pineapple slices can"
    device: str = "cuda"
    num_flow_steps: Optional[int] = None       # None -> use checkpoint default (10)
    # Only used when action_type == "joint_pos_rel"; left for parity, unused for our model.
    relative_max_joint_delta: Optional[List[float]] = None


class BridgePolicy:
    """Duck-typed InferencePolicy: implements prepare_model / reset / get_action."""

    def __init__(self, config: BridgePolicyConfig):
        self.config = config
        self.camera_names = config.camera_names
        self.action_move_group_names = config.action_move_group_names
        self.action_spec = config.action_spec
        self.action_horizon = config.action_horizon
        self.execute_horizon = config.execute_horizon
        self.action_type = config.action_type
        self.clamp_gripper = config.clamp_gripper
        self.gripper_representation_count = config.gripper_representation_count
        self.relative_max_joint_delta = (
            np.array(config.relative_max_joint_delta)
            if config.relative_max_joint_delta is not None else None
        )

        self.action_buffer: List[Dict[str, np.ndarray]] = []
        self.buffer_index = 0
        self.step_count = 0
        self._prepared = False
        self.agent = None

    # ---- lifecycle -------------------------------------------------------
    def prepare_model(self):
        """Load the model once (called by the server before serving)."""
        if self._prepared:
            return
        from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper

        logger.info(f"Loading checkpoint: {self.config.checkpoint_path}")
        self.agent = SynthManipMolmoInferenceWrapper(
            checkpoint_path=self.config.checkpoint_path,
            device=self.config.device,
            num_flow_steps=self.config.num_flow_steps,
            use_bfloat16=True,
        )
        self.input_window_size = getattr(self.agent.model_config, "n_obs_steps", 1)
        logger.info(
            f"Model loaded. action_horizon={getattr(self.agent, 'action_horizon', '?')}, "
            f"action_dim={getattr(self.agent, 'action_dim', '?')}, "
            f"n_obs_steps={self.input_window_size}, num_flow_steps={self.agent.num_flow_steps}"
        )
        self._prepared = True

    def reset(self):
        """Called on each new client connection."""
        self.action_buffer = []
        self.buffer_index = 0
        self.step_count = 0
        logger.info("Policy reset for new connection")

    # ---- inference -------------------------------------------------------
    def _extract_images(self, obs: dict) -> List[np.ndarray]:
        images = []
        for cam in self.camera_names:
            if cam not in obs:
                raise KeyError(
                    f"Camera '{cam}' not in observation. Available keys: {list(obs.keys())}"
                )
            images.append(np.asarray(obs[cam]))
        return images

    def _extract_state(self, obs: dict) -> np.ndarray:
        if "qpos" not in obs:
            raise KeyError(f"'qpos' missing from observation. Keys: {list(obs.keys())}")
        qpos = obs["qpos"]
        parts = []
        for g in self.action_move_group_names:
            if g not in qpos:
                raise KeyError(f"qpos['{g}'] missing. qpos keys: {list(qpos.keys())}")
            v = np.asarray(qpos[g], dtype=np.float32).reshape(-1)
            if g == "gripper":
                v = v[: self.gripper_representation_count]
            parts.append(v)
        return np.concatenate(parts).astype(np.float32)

    def _populate_action_buffer(self, obs: dict) -> None:
        images = self._extract_images(obs)
        state = self._extract_state(obs)
        task = obs.get("task") or self.config.default_task

        pred = self.agent.get_action_chunk(
            images=images,                 # single-cam -> wrapper routes 1 img correctly
            task_description=task,
            state=state,
        )  # (action_horizon, action_dim), un-normalized

        self.action_buffer = []
        for t in range(pred.shape[0]):
            action, start = {}, 0
            for g in self.action_move_group_names:
                dim = self.action_spec[g]
                sel = pred[t, start:start + dim]
                if g == "gripper" and self.clamp_gripper:
                    sel = np.where(sel > 128, 255, 0).astype(sel.dtype)
                action[g] = sel
                start += dim
            self.action_buffer.append(action)
        self.buffer_index = 0

    def get_action(self, observation) -> Dict[str, np.ndarray]:
        obs = observation[0] if isinstance(observation, list) else observation

        if self.buffer_index >= self.execute_horizon or not self.action_buffer:
            self._populate_action_buffer(obs)

        action = {k: v.copy() for k, v in self.action_buffer[self.buffer_index].items()}
        self.buffer_index += 1
        self.step_count += 1

        # For relative-action models only (our pilot is absolute, so this is a no-op).
        if self.action_type == "joint_pos_rel":
            action["arm"][:7] += np.asarray(obs["qpos"]["arm"], dtype=np.float32)
            if self.relative_max_joint_delta is not None:
                deltas = action["arm"][:7] - np.asarray(obs["qpos"]["arm"], dtype=np.float32)
                scale = np.abs(deltas) / self.relative_max_joint_delta
                if np.max(scale) > 1:
                    action["arm"][:7] = np.asarray(obs["qpos"]["arm"], dtype=np.float32) + deltas / np.max(scale)

        return action
