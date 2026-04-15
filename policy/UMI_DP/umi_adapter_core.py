import os
import sys
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial.transform as st
import torch
import transforms3d as t3d

import dill
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def ensure_umi_imports(umi_root: str) -> None:
    root = os.path.abspath(os.path.expanduser(umi_root))
    if root not in sys.path:
        sys.path.insert(0, root)


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = t3d.quaternions.quat2mat(quat_wxyz)
    return st.Rotation.from_matrix(mat).as_rotvec()


def pose6_to_xyz_quat_wxyz(pose6: np.ndarray) -> np.ndarray:
    pos = pose6[:3]
    rotvec = pose6[3:6]
    mat = st.Rotation.from_rotvec(rotvec).as_matrix()
    quat_wxyz = t3d.quaternions.mat2quat(mat)
    return np.concatenate([pos, quat_wxyz])


def umi_env14_to_robotwin_ee16(env14: np.ndarray) -> np.ndarray:
    """UMI get_real_umi_action 输出约 14 维 (两臂 6+1)，转为 RoboTwin ee: 左7+左夹爪+右7+右夹爪=16。"""
    assert env14.shape[-1] == 14, env14.shape
    p0 = env14[..., :6]
    g0 = env14[..., 6:7]
    p1 = env14[..., 7:13]
    g1 = env14[..., 13:14]
    e0 = pose6_to_xyz_quat_wxyz(p0)
    e1 = pose6_to_xyz_quat_wxyz(p1)
    return np.concatenate([e0, g0, e1, g1], axis=-1)


def load_umi_policy(ckpt_path: str, device: str, umi_root: Optional[str] = None):
    root = umi_root or os.environ.get("UMI_ROOT")
    if not root:
        raise ValueError("需要提供 umi_root 或设置环境变量 UMI_ROOT（指向 universal_manipulation_interface 仓库根目录）")
    ensure_umi_imports(root)
    import hydra
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    path = ckpt_path
    if not path.endswith(".ckpt"):
        path = os.path.join(path, "checkpoints", "latest.ckpt")
    payload = torch.load(open(path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def build_tx_matrix(arr: Optional[List[List[float]]]) -> np.ndarray:
    if arr is None:
        return np.eye(4, dtype=np.float64)
    return np.array(arr, dtype=np.float64)


class UmiObsRunner:
    """缓存逐步 env_obs（与 UMI 真机一致的单步 dict），再堆成带 T 维供 get_real_umi_obs_dict。"""

    def __init__(
        self,
        n_obs_steps: int,
        shape_meta: dict,
        pose_repr: dict,
        tx_robot1_robot0: np.ndarray,
    ):
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.pose_repr = pose_repr
        self.tx_robot1_robot0 = tx_robot1_robot0
        self._buf: Deque[Dict[str, np.ndarray]] = deque(maxlen=max(n_obs_steps * 2, n_obs_steps + 2))
        self.episode_start_pose: Optional[List[np.ndarray]] = None

    def reset(self) -> None:
        self._buf.clear()
        self.episode_start_pose = None

    def push(self, env_obs_one: Dict[str, np.ndarray]) -> None:
        if self.episode_start_pose is None:
            self.episode_start_pose = []
            for rid in (0, 1):
                pos = env_obs_one[f"robot{rid}_eef_pos"]
                rot = env_obs_one[f"robot{rid}_eef_rot_axis_angle"]
                if pos.ndim == 2:
                    pos = pos[-1]
                if rot.ndim == 2:
                    rot = rot[-1]
                p = np.concatenate([pos, rot])
                self.episode_start_pose.append(p.astype(np.float64))
        self._buf.append(env_obs_one)

    def _stack(self) -> Dict[str, np.ndarray]:
        buf = list(self._buf)
        assert len(buf) > 0
        keys = buf[0].keys()
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            arrs = [b[k] for b in buf]
            if arrs[0].ndim == 3:
                x = np.stack(arrs, axis=0)
            elif arrs[0].ndim == 1:
                x = np.stack(arrs, axis=0)
            elif arrs[0].ndim == 2:
                x = np.stack(arrs, axis=0)
            else:
                x = np.stack(arrs, axis=0)
            T_need = self.n_obs_steps
            if x.shape[0] < T_need:
                pad_n = T_need - x.shape[0]
                rep = np.repeat(x[:1], pad_n, axis=0)
                x = np.concatenate([rep, x], axis=0)
            elif x.shape[0] > T_need:
                x = x[-T_need:]
            out[k] = x
        return out

    def get_policy_obs(
        self,
        dict_apply,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        from umi.real_world.real_inference_util import get_real_umi_obs_dict

        raw = self._stack()
        obs_pose_repr = self.pose_repr.get("obs_pose_repr", "relative")
        obs_dict_np = get_real_umi_obs_dict(
            raw,
            self.shape_meta,
            obs_pose_repr=obs_pose_repr,
            tx_robot1_robot0=self.tx_robot1_robot0,
            episode_start_pose=self.episode_start_pose,
        )
        obs_t = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).to(device=device, dtype=dtype))
        out = {k: v.unsqueeze(0) for k, v in obs_t.items()}
        return out


def robotwin_single_step_to_env_obs(
    observation: dict,
    get_left_pose_fn,
    get_right_pose_fn,
    get_left_grip_fn,
    get_right_grip_fn,
    cam0_key: str,
    cam1_key: str,
    gripper_scale: float,
) -> Dict[str, np.ndarray]:
    """单步 RoboTwin 观测 -> UMI env_obs（各键最后一维为时间长度 1 的数组，便于堆叠）。"""
    obs = observation["observation"]
    im0 = obs[cam0_key]["rgb"].astype(np.uint8)
    im1 = obs[cam1_key]["rgb"].astype(np.uint8)
    if im0.ndim == 4:
        im0 = im0[0]
    if im1.ndim == 4:
        im1 = im1[0]

    lp = np.asarray(get_left_pose_fn(), dtype=np.float64)
    rp = np.asarray(get_right_pose_fn(), dtype=np.float64)
    pos0, q0 = np.array(lp[:3]), np.array(lp[3:7])
    pos1, q1 = np.array(rp[:3]), np.array(rp[3:7])
    rv0 = quat_wxyz_to_rotvec(q0)
    rv1 = quat_wxyz_to_rotvec(q1)
    g0 = np.array([float(get_left_grip_fn()) * gripper_scale], dtype=np.float32)
    g1 = np.array([float(get_right_grip_fn()) * gripper_scale], dtype=np.float32)

    return {
        "camera0_rgb": im0,
        "camera1_rgb": im1,
        "robot0_eef_pos": pos0.astype(np.float32),
        "robot0_eef_rot_axis_angle": rv0.astype(np.float32),
        "robot0_gripper_width": g0.astype(np.float32),
        "robot1_eef_pos": pos1.astype(np.float32),
        "robot1_eef_rot_axis_angle": rv1.astype(np.float32),
        "robot1_gripper_width": g1.astype(np.float32),
    }
