import os
from typing import Any, Dict

import numpy as np
import torch

from .umi_adapter_core import (
    UmiObsRunner,
    build_tx_matrix,
    load_umi_policy,
    robotwin_single_step_to_env_obs,
    umi_env14_to_robotwin_ee16,
)

_CURRENT_TASK_ENV = None


def _cfg_to_plain(obj: Any) -> Any:
    """将 Hydra/OmegaConf 的 DictConfig/ListConfig 转为原生 dict/list（避免顶层依赖 omegaconf）。"""
    if obj is None or isinstance(obj, (bool, int, float, str, np.ndarray)):
        return obj
    if isinstance(obj, dict):
        return {k: _cfg_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cfg_to_plain(x) for x in obj)
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__") and not isinstance(obj, (str, bytes)):
        try:
            return {k: _cfg_to_plain(obj[k]) for k in obj.keys()}
        except Exception:
            pass
    return obj


def encode_obs(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    assert _CURRENT_TASK_ENV is not None, "eval() 须在 encode_obs 之前设置 TASK_ENV"
    te = _CURRENT_TASK_ENV
    usr = getattr(encode_obs, "_usr_args", None)
    assert usr is not None, "get_model() 须在首次 encode_obs 前完成以注入 usr_args"
    return robotwin_single_step_to_env_obs(
        observation,
        lambda: te.get_arm_pose("left"),
        lambda: te.get_arm_pose("right"),
        lambda: te.robot.get_left_gripper_val(),
        lambda: te.robot.get_right_gripper_val(),
        usr["camera0_rgb_source"],
        usr["camera1_rgb_source"],
        float(usr.get("gripper_scale", 0.09)),
    )


class UmiBimanualDP:
    def __init__(self, usr_args: Dict[str, Any]):
        self.usr_args = usr_args
        umi_root = os.path.abspath(os.path.expanduser(usr_args["umi_root"]))
        os.environ["UMI_ROOT"] = umi_root

        from .umi_adapter_core import ensure_umi_imports

        ensure_umi_imports(umi_root)

        ckpt = usr_args["ckpt_path"]
        device = usr_args.get("device", "cuda:0")
        self.policy, self.cfg = load_umi_policy(ckpt, device, umi_root=umi_root)
        self.device = torch.device(device)
        self.dtype = next(self.policy.parameters()).dtype
        ni = usr_args.get("num_inference_steps")
        if ni is not None and hasattr(self.policy, "num_inference_steps"):
            self.policy.num_inference_steps = int(ni)

        task = self.cfg.task
        self.shape_meta = _cfg_to_plain(task.shape_meta)
        pr = getattr(task, "pose_repr", None)
        self.pose_repr = _cfg_to_plain(pr) if pr is not None else {}

        n_obs = int(
            usr_args.get("n_obs_steps")
            or getattr(task, "img_obs_horizon", None)
            or 2
        )
        tx = build_tx_matrix(usr_args.get("tx_robot1_robot0"))
        self.runner = UmiObsRunner(
            n_obs_steps=n_obs,
            shape_meta=self.shape_meta,
            pose_repr=self.pose_repr,
            tx_robot1_robot0=tx,
        )
        self.n_action_steps = int(self.policy.action_horizon)
        self.steps_per_inference = int(usr_args.get("steps_per_inference", 6))
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "relative")

    def reset(self) -> None:
        self.runner.reset()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def update_obs(self, env_obs_one: Dict[str, np.ndarray]) -> None:
        self.runner.push(env_obs_one)

    def get_action_chunk(self) -> np.ndarray:
        from diffusion_policy.common.pytorch_util import dict_apply
        from umi.real_world.real_inference_util import get_real_umi_action

        obs_t = self.runner.get_policy_obs(dict_apply, self.device, self.dtype)
        with torch.no_grad():
            out = self.policy.predict_action(obs_t)
        pred = out["action_pred"][0].detach().cpu().numpy()
        raw = self.runner._stack()
        if pred.ndim == 1:
            pred = pred[None, :]
        actions_rw = []
        n_exec = min(self.n_action_steps, pred.shape[0], self.steps_per_inference)
        for i in range(n_exec):
            a = pred[i]
            env14 = get_real_umi_action(a, raw, self.action_pose_repr)
            actions_rw.append(umi_env14_to_robotwin_ee16(env14))
        return np.stack(actions_rw, axis=0)


def get_model(usr_args: Dict[str, Any]):
    encode_obs._usr_args = usr_args
    return UmiBimanualDP(usr_args)


def eval(TASK_ENV, model: UmiBimanualDP, observation: dict):
    global _CURRENT_TASK_ENV
    _CURRENT_TASK_ENV = TASK_ENV

    obs = encode_obs(observation)
    if len(model.runner._buf) == 0:
        model.update_obs(obs)

    chunk = model.get_action_chunk()
    for action in chunk:
        TASK_ENV.take_action(action, action_type="ee")
        observation = TASK_ENV.get_obs()
        model.update_obs(encode_obs(observation))


def reset_model(model: UmiBimanualDP):
    model.reset()
