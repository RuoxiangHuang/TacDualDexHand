"""ShadowHand dual-arm folder manipulation demo.

This script uses ShadowHand dual-arm environment to manipulate a folder object.
Based on run_shadowhand_dual_arm_gui_pose_new.py.

Usage:
    isaaclab -p ./scripts/shadowhand_pick_up_object/run_shadowhand_dual_arm_folder.py --num_envs 1 --config ./scripts/shadowhand_pick_up_object/dual_arm_folder_config.yaml
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="ShadowHand dual-arm folder manipulation demo"
)
parser.add_argument(
    "--config",
    type=str,
    default="./scripts/shadowhand_pick_up_object/dual_arm_folder_config.yaml",
    help="Path to YAML config for dual-arm folder demo.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the GUI",
)
parser.add_argument(
    "--folder_usd_path",
    type=str,
    default="/media/isaac/e2fa93f5-512e-4f9a-9023-40cf92b0ad74/TacEx/source/tacex_tasks/tacex_tasks/bottle_cap/stationery/folder/file_7/model_file_7.usd",
    help="Path to folder USD file",
)
parser.add_argument(
    "--folder_pos",
    type=float,
    nargs=3,
    default=[1.1, -0.1, 0.01],
    help="Folder initial position (x, y, z). Default: 1.1 -0.1 0.01",
)
parser.add_argument(
    "--folder_scale",
    type=float,
    nargs=3,
    default=[1.0, 1.0, 1.0],
    help="Folder scale (x, y, z). Default: 1.0 1.0 1.0",
)
parser.add_argument(
    "--disable_fixed_joints",
    type=bool,
    default=True,
    help="Whether to disable FixedJoint in folder (default: True)",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# YAML config loading (optional)
# -----------------------------------------------------------------------------
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _load_yaml_config(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required for --config. Please install `pyyaml` in your IsaacLab python env.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


CFG: dict = {}
try:
    if args_cli.config:
        CFG = _load_yaml_config(args_cli.config)
except FileNotFoundError:
    print(f"[WARNING] Config file not found: {args_cli.config}. Using script defaults.")
    CFG = {}
except Exception as e:
    print(f"[WARNING] Failed to load config '{args_cli.config}': {e}. Using script defaults.")
    CFG = {}

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
from contextlib import suppress

import numpy as np
import torch
import datetime
from pathlib import Path

import carb
import pynvml
from isaacsim.core.api.objects import VisualCuboid

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.envs import DirectRLEnv, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaacsim.core.prims import XFormPrim
import omni.usd
from pxr import UsdGeom

from tacex import GelSightSensor

# Import from folder.py instead of shadowhand_pick_up_object_env.py
from tacex_tasks.shadowhand_pick_up_object.folder_new import (
    ShadowHandDualArmPickUpObjectScriptedEnv,
    ShadowHandDualArmPickUpObjectScriptedEnvCfg,
)

# -----------------------------------------------------------------------------
# YAML helpers (same as run_shadowhand_dual_arm_gui_pose_new.py)
# -----------------------------------------------------------------------------
def _get_cfg(dct: dict, path: list[str], default=None):
    cur = dct
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_tuple3(x, default=(0.0, 0.0, 0.0)):
    if x is None:
        return default
    return (float(x[0]), float(x[1]), float(x[2]))


def _as_tuple4_wxyz(x, default=(1.0, 0.0, 0.0, 0.0)):
    if x is None:
        return default
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]))


def _wxyz_to_xyzw(q_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert quaternion (w,x,y,z) -> (x,y,z,w). IsaacLab Articulation init_state rot uses (x,y,z,w)."""
    return (q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0])


def _apply_sensor_cfg_overrides(sensor_cfg, overrides: dict, defaults: dict):
    """Apply tactile sensor cfg overrides to a GelSightHandCfg-like object."""
    cam_res = overrides.get("camera_resolution", defaults.get("camera_resolution"))
    clip = overrides.get("clipping_range", defaults.get("clipping_range"))
    if cam_res is not None:
        sensor_cfg.sensor_camera_cfg.resolution = (int(cam_res[0]), int(cam_res[1]))
    if clip is not None:
        sensor_cfg.sensor_camera_cfg.clipping_range = (float(clip[0]), float(clip[1]))

    gel_min = overrides.get("gelpad_to_camera_min_distance", defaults.get("gelpad_to_camera_min_distance"))
    tactile_res = overrides.get("tactile_img_resolution", defaults.get("tactile_img_resolution"))
    if hasattr(sensor_cfg, "optical_sim_cfg") and sensor_cfg.optical_sim_cfg is not None:
        replace_kwargs = {}
        if gel_min is not None:
            replace_kwargs["gelpad_to_camera_min_distance"] = float(gel_min)
        if tactile_res is not None:
            replace_kwargs["tactile_img_res"] = (int(tactile_res[0]), int(tactile_res[1]))
        if replace_kwargs:
            sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(**replace_kwargs)


def _expand_env_vars(path: str) -> str:
    """Expand ${VAR}/$VAR and ~ in a string path.

    Special-case: if TACEX_ASSETS_DATA_DIR is not set as an env var, fall back to
    the python constant `tacex_assets.TACEX_ASSETS_DATA_DIR`.
    """
    import os
    raw = str(path)
    # Fall back to python constant if the env var isn't set.
    if ("TACEX_ASSETS_DATA_DIR" in raw) and ("TACEX_ASSETS_DATA_DIR" not in os.environ):
        try:
            from tacex_assets import TACEX_ASSETS_DATA_DIR as _BASE  # type: ignore
            raw = raw.replace("${TACEX_ASSETS_DATA_DIR}", str(_BASE))
            raw = raw.replace("$TACEX_ASSETS_DATA_DIR", str(_BASE))
        except Exception:
            # If import fails, we'll leave it to expandvars and the caller will likely error.
            pass
    return os.path.expanduser(os.path.expandvars(raw))


def _apply_camera_local_pose(stage, prim_path: str, pos_xyz: tuple[float, float, float], quat_wxyz: tuple[float, float, float, float]):
    """Set local pose on a prim using translate + orient ops (wxyz)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()
    translate_op.Set(pos_xyz)
    from pxr import Gf
    orient_op.Set(Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])))


def _read_local_pose_wxyz(stage, prim_path: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
    """Read local pose (translate + orient) from a prim. Returns (pos, quat_wxyz) or None if prim missing."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xformable = UsdGeom.Xformable(prim)
    pos = (0.0, 0.0, 0.0)
    quat = (1.0, 0.0, 0.0, 0.0)
    from pxr import Gf
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            if v is not None:
                pos = (float(v[0]), float(v[1]), float(v[2]))
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            q = op.Get()
            if q is not None:
                # q is Gf.Quat* -> real + imaginary vec3
                if isinstance(q, Gf.Quatd) or isinstance(q, Gf.Quatf):
                    quat = (float(q.GetReal()), float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2]))
                else:
                    # best effort
                    quat = (float(q.real), float(q.imaginary[0]), float(q.imaginary[1]), float(q.imaginary[2]))
    return pos, quat


# Fixed grasp pose (same as run_shadowhand_dual_arm_gui_pose_new.py)
FIXED_GRASP_POSE = {
    "WRJ2": -0.5236,
    "WRJ1": -0.1981,
    "FFJ4": -0.3491,
    "LFJ5": 0.4854,
    "MFJ4": -0.2009,
    "RFJ4": -0.1491,
    "THJ5": -0.1472,
    "FFJ3": 1.5708,
    "LFJ4": -0.3491,
    "MFJ3": 1.5708,
    "RFJ3": 1.5708,
    "THJ4": 1.2217,
    "FFJ2": 0.6000,
    "LFJ3": 1.5708,
    "MFJ2": 0.8000,
    "RFJ2": 1.5708,
    "THJ3": 0.2094,
    "FFJ1": 0.4000,
    "LFJ2": 0.2000,
    "MFJ1": 0.3500,
    "RFJ1": 0.4000,
    "THJ2": 0.6519,
    "LFJ1": 1.5708,
    "THJ1": -0.1118,
}

def _apply_robot_joint_pos_overrides(env, arm_name: str, joint_pos_cfg: dict, arm_joint_preset: list[float] | None):
    """Apply initial joint positions to a robot (left/right) by matching joint names and optional 6-DoF arm preset."""
    if arm_name == "left":
        robot = env._robot_left
        arm_ids = getattr(env, "_arm_joint_ids_left", None)
    else:
        robot = env._robot_right
        arm_ids = getattr(env, "_arm_joint_ids_right", None)

    joint_names = robot.joint_names
    joint_pos = robot.data.default_joint_pos.clone()

    if arm_joint_preset is not None and arm_ids is not None and len(arm_ids) >= 6:
        for i in range(6):
            j_idx = int(arm_ids[i].item())
            if j_idx < joint_pos.shape[1]:
                joint_pos[:, j_idx] = float(arm_joint_preset[i])

    if isinstance(joint_pos_cfg, dict) and len(joint_pos_cfg) > 0:
        for j in range(len(joint_names)):
            jn = joint_names[j]
            val = None
            if jn in joint_pos_cfg:
                val = joint_pos_cfg[jn]
            else:
                for pat, v in joint_pos_cfg.items():
                    if str(pat).upper() in jn.upper() or jn.upper() in str(pat).upper():
                        val = v
                        break
            if val is not None:
                joint_pos[:, j] = float(val)

    limits = robot.data.soft_joint_pos_limits
    joint_pos = torch.clamp(joint_pos, limits[:, :, 0], limits[:, :, 1])
    joint_vel = torch.zeros_like(joint_pos)
    robot.set_joint_position_target(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


# Reuse CustomEnvWindow from run_shadowhand_dual_arm_gui_pose_new.py
# For simplicity, we'll import the window class logic inline or create a simplified version
# Since the window class is quite large, we'll create a simplified version focused on folder manipulation

class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the ShadowHand dual-arm pick-up environment."""

    def __init__(self, env: DirectRLEnv, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        self.reset = False
        self.env = env
        
        # Default sensor parameters
        # Prefer YAML defaults if present
        y_defaults = _get_cfg(getattr(env, "_cfg_yaml", {}), ["tactile_sensors", "defaults"], {}) or {}
        y_clip = y_defaults.get("clipping_range", None) if isinstance(y_defaults, dict) else None
        y_min = y_defaults.get("gelpad_to_camera_min_distance", None) if isinstance(y_defaults, dict) else None
        if y_clip is not None:
            self.clipping_range_near = float(y_clip[0])
            self.clipping_range_far = float(y_clip[1])
        else:
            self.clipping_range_near = 0.017
            self.clipping_range_far = 0.024
        self.gelpad_to_camera_min_distance = float(y_min) if y_min is not None else 0.017

        # End-effector offset defaults (prefer YAML; fallback to env.cfg)
        ee_off = _get_cfg(getattr(env, "_cfg_yaml", {}), ["end_effector_offset"], {}) or {}
        # Backward compatible: allow either {pos,quat_wxyz} or {left:{...}, right:{...}}
        if isinstance(ee_off, dict) and ("left" in ee_off or "right" in ee_off):
            ee_left = ee_off.get("left", {}) if isinstance(ee_off.get("left", {}), dict) else {}
            ee_right = ee_off.get("right", {}) if isinstance(ee_off.get("right", {}), dict) else {}
            ee_pos_l = ee_left.get("pos", None)
            ee_quat_l = ee_left.get("quat_wxyz", None)
            ee_pos_r = ee_right.get("pos", None)
            ee_quat_r = ee_right.get("quat_wxyz", None)
        else:
            ee_pos_l = ee_off.get("pos", None) if isinstance(ee_off, dict) else None
            ee_quat_l = ee_off.get("quat_wxyz", None) if isinstance(ee_off, dict) else None
            ee_pos_r = ee_pos_l
            ee_quat_r = ee_quat_l

        # Prefer current env.cfg values if available (left defaults)
        cfg_pos = getattr(getattr(env, "cfg", None), "body_offset_pos", (0.0, 0.0, 0.0))
        cfg_quat = getattr(getattr(env, "cfg", None), "body_offset_rot", (1.0, 0.0, 0.0, 0.0))
        self.ee_offset_pos_left = _as_tuple3(ee_pos_l, tuple(cfg_pos))
        self.ee_offset_quat_wxyz_left = _as_tuple4_wxyz(ee_quat_l, tuple(cfg_quat))
        self.ee_offset_pos_right = _as_tuple3(ee_pos_r, tuple(cfg_pos))
        self.ee_offset_quat_wxyz_right = _as_tuple4_wxyz(ee_quat_r, tuple(cfg_quat))

        def _quat_to_rpy_deg(q_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float]:
            try:
                q = torch.tensor(q_wxyz, dtype=torch.float32).unsqueeze(0)
                r, p, y = euler_xyz_from_quat(q)
                return (float(r[0].item() * 180.0 / np.pi), float(p[0].item() * 180.0 / np.pi), float(y[0].item() * 180.0 / np.pi))
            except Exception:
                return (0.0, 0.0, 0.0)

        self.ee_offset_rpy_deg_left = _quat_to_rpy_deg(self.ee_offset_quat_wxyz_left)
        self.ee_offset_rpy_deg_right = _quat_to_rpy_deg(self.ee_offset_quat_wxyz_right)
        
        # Finger joint UI controls (will be initialized after env is ready)
        self.finger_joint_controls_left = {}
        self.finger_joint_controls_right = {}
        
        if ui_utils is not None:
            with self.ui_window_elements["main_vstack"]:
                with self.ui_window_elements["debug_frame"]:
                    with self.ui_window_elements["debug_vstack"]:
                        self.ui_window_elements["reset_button"] = ui_utils.btn_builder(
                            type="button",
                            text="Reset Env",
                            tooltip="Resets the environment.",
                            on_clicked_fn=self._reset_env,
                        )
            
            # Add sensor configuration frame
            with self.ui_window_elements["main_vstack"]:
                self._build_sensor_config_frame()
            # Add end-effector offset frame
            with self.ui_window_elements["main_vstack"]:
                self._build_ee_offset_frame()
            
            # Add finger joint control frames (will be built after env initialization)
            with self.ui_window_elements["main_vstack"]:
                self._build_finger_joint_control_frame("left")
            with self.ui_window_elements["main_vstack"]:
                self._build_finger_joint_control_frame("right")

    def _build_sensor_config_frame(self):
        """Build sensor configuration UI frame."""
        if ui_utils is None:
            return
            
        import omni.ui
        
        self.ui_window_elements["sensor_config_frame"] = omni.ui.CollapsableFrame(
            title="Sensor Configuration",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        
        with self.ui_window_elements["sensor_config_frame"]:
            self.ui_window_elements["sensor_config_vstack"] = omni.ui.VStack(spacing=5, height=50)
            with self.ui_window_elements["sensor_config_vstack"]:
                # Clipping range near
                self.ui_window_elements["clipping_range_near"] = ui_utils.combo_floatfield_slider_builder(
                    label="Clipping Range Near (m)",
                    default_val=self.clipping_range_near,
                    min=0.001,
                    max=0.050,
                    step=0.001,
                    tooltip="Near clipping plane distance for sensor cameras (applies to all 10 sensors).",
                )[0]
                
                # Clipping range far
                self.ui_window_elements["clipping_range_far"] = ui_utils.combo_floatfield_slider_builder(
                    label="Clipping Range Far (m)",
                    default_val=self.clipping_range_far,
                    min=0.010,
                    max=0.100,
                    step=0.001,
                    tooltip="Far clipping plane distance for sensor cameras (applies to all 10 sensors).",
                )[0]
                
                # Gelpad to camera min distance
                self.ui_window_elements["gelpad_to_camera_min_distance"] = ui_utils.combo_floatfield_slider_builder(
                    label="Gelpad to Camera Min Distance (m)",
                    default_val=self.gelpad_to_camera_min_distance,
                    min=0.010,
                    max=0.050,
                    step=0.001,
                    tooltip="Minimum distance from gelpad to camera for optical simulation (applies to all 10 sensors).",
                )[0]
                
                # Apply button
                self.ui_window_elements["apply_sensor_config_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Apply Sensor Config",
                    tooltip="Applies the above sensor configuration to all 10 finger sensors.",
                    on_clicked_fn=self._apply_sensor_config,
                )

                # Save YAML config snapshot
                self.ui_window_elements["save_yaml_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Save YAML Config Snapshot",
                    tooltip="Saves current runtime values into a new YAML file (next to --config).",
                    on_clicked_fn=self._save_yaml_snapshot,
                )

    def _build_ee_offset_frame(self):
        """Build end-effector offset UI frame."""
        if ui_utils is None:
            return
        import omni.ui

        self.ui_window_elements["ee_offset_frame"] = omni.ui.CollapsableFrame(
            title="End-Effector Offset (task frame)",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=True,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )

        with self.ui_window_elements["ee_offset_frame"]:
            v = omni.ui.VStack(spacing=5, height=50)
            with v:
                self._build_ee_offset_controls_for_arm("left")
                self._build_ee_offset_controls_for_arm("right")

    def _build_ee_offset_controls_for_arm(self, arm_name: str):
        """Build EE offset controls for a specific arm."""
        if ui_utils is None:
            return
        import omni.ui

        if arm_name == "left":
            pos = self.ee_offset_pos_left
            rpy = self.ee_offset_rpy_deg_left
        else:
            pos = self.ee_offset_pos_right
            rpy = self.ee_offset_rpy_deg_right

        frame = omni.ui.CollapsableFrame(
            title=f"EE Offset - {arm_name.upper()} Arm",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=(arm_name != "left"),
            style=ui_utils.get_style(),
        )
        with frame:
            vv = omni.ui.VStack(spacing=5, height=50)
            with vv:
                # Position (m)
                self.ui_window_elements[f"ee_off_{arm_name}_x"] = ui_utils.combo_floatfield_slider_builder(
                    label="Offset X (m)", default_val=float(pos[0]), min=-0.2, max=0.2, step=0.001,
                    tooltip=f"EE task frame offset position X ({arm_name} arm).",
                )[0]
                self.ui_window_elements[f"ee_off_{arm_name}_y"] = ui_utils.combo_floatfield_slider_builder(
                    label="Offset Y (m)", default_val=float(pos[1]), min=-0.2, max=0.2, step=0.001,
                    tooltip=f"EE task frame offset position Y ({arm_name} arm).",
                )[0]
                self.ui_window_elements[f"ee_off_{arm_name}_z"] = ui_utils.combo_floatfield_slider_builder(
                    label="Offset Z (m)", default_val=float(pos[2]), min=-0.2, max=0.2, step=0.001,
                    tooltip=f"EE task frame offset position Z ({arm_name} arm).",
                )[0]

                rr, pp, yy = rpy
                self.ui_window_elements[f"ee_off_{arm_name}_roll"] = ui_utils.combo_floatfield_slider_builder(
                    label="Offset Roll (deg)", default_val=float(rr), min=-180.0, max=180.0, step=1.0,
                    tooltip=f"EE task frame offset roll ({arm_name} arm).",
                )[0]
                self.ui_window_elements[f"ee_off_{arm_name}_pitch"] = ui_utils.combo_floatfield_slider_builder(
                    label="Offset Pitch (deg)", default_val=float(pp), min=-180.0, max=180.0, step=1.0,
                    tooltip=f"EE task frame offset pitch ({arm_name} arm).",
                )[0]
                self.ui_window_elements[f"ee_off_{arm_name}_yaw"] = ui_utils.combo_floatfield_slider_builder(
                    label="Offset Yaw (deg)", default_val=float(yy), min=-180.0, max=180.0, step=1.0,
                    tooltip=f"EE task frame offset yaw ({arm_name} arm).",
                )[0]

                self.ui_window_elements[f"apply_ee_offset_{arm_name}_button"] = ui_utils.btn_builder(
                    type="button",
                    text=f"Apply EE Offset ({arm_name.upper()})",
                    tooltip=f"Applies EE offset for {arm_name} arm immediately.",
                    on_clicked_fn=lambda a=arm_name: self._apply_ee_offset(a),
                )
                self.ui_window_elements[f"zero_ee_offset_{arm_name}_button"] = ui_utils.btn_builder(
                    type="button",
                    text=f"Set Offset = 0 ({arm_name.upper()})",
                    tooltip=f"Sets EE offset to identity for {arm_name} arm and applies it.",
                    on_clicked_fn=lambda a=arm_name: self._zero_ee_offset(a),
                )

    def _populate_command_menu(self, ui, width, height):
        """Add custom menu items."""
        pass

    def _reset_env(self):
        """Callback for the reset button."""
        self.reset = True
    
    def _apply_sensor_config(self):
        """Apply sensor configuration to all sensors."""
        if ui_utils is None:
            return
        
        # Get current values from UI
        self.clipping_range_near = self.ui_window_elements["clipping_range_near"].get_value_as_float()
        self.clipping_range_far = self.ui_window_elements["clipping_range_far"].get_value_as_float()
        self.gelpad_to_camera_min_distance = self.ui_window_elements["gelpad_to_camera_min_distance"].get_value_as_float()
        
        # Validate values
        if self.clipping_range_near >= self.clipping_range_far:
            print(f"[WARNING] Clipping range near ({self.clipping_range_near}) must be less than far ({self.clipping_range_far})")
            return
        
        print(f"\n[INFO] Applying sensor configuration:")
        print(f"  Clipping range: ({self.clipping_range_near:.4f}, {self.clipping_range_far:.4f}) m")
        print(f"  Gelpad to camera min distance: {self.gelpad_to_camera_min_distance:.4f} m")
        
        # Apply to all sensors (10 sensors: 5 per arm)
        sensors = [
            ("ff_left", self.env.gelsighthand_ff_left),
            ("mf_left", self.env.gelsighthand_mf_left),
            ("rf_left", self.env.gelsighthand_rf_left),
            ("lf_left", self.env.gelsighthand_lf_left),
            ("th_left", self.env.gelsighthand_th_left),
            ("ff_right", self.env.gelsighthand_ff_right),
            ("mf_right", self.env.gelsighthand_mf_right),
            ("rf_right", self.env.gelsighthand_rf_right),
            ("lf_right", self.env.gelsighthand_lf_right),
            ("th_right", self.env.gelsighthand_th_right),
        ]
        
        for sensor_name, sensor in sensors:
            try:
                # Update clipping range in config
                if hasattr(sensor, 'cfg') and hasattr(sensor.cfg, 'sensor_camera_cfg'):
                    sensor.cfg.sensor_camera_cfg.clipping_range = (
                        self.clipping_range_near,
                        self.clipping_range_far
                    )
                    
                    # Update camera clipping range if camera exists
                    if hasattr(sensor, 'camera') and sensor.camera is not None:
                        if hasattr(sensor.camera, 'cfg'):
                            sensor.camera.cfg.clipping_range = (
                                self.clipping_range_near,
                                self.clipping_range_far
                            )
                        
                        if hasattr(sensor.camera, '_prim_view') and sensor.camera._prim_view is not None:
                            try:
                                import omni.usd
                                from pxr import UsdGeom
                                
                                for prim in sensor.camera._prim_view.prims:
                                    camera_prim = prim
                                    camera_api = UsdGeom.Camera(camera_prim)
                                    if camera_api:
                                        camera_api.GetClippingRangeAttr().Set(
                                            (self.clipping_range_near, self.clipping_range_far)
                                        )
                            except Exception as e:
                                print(f"    [WARNING] Could not update USD camera clipping range: {e}")
                
                # Update gelpad to camera min distance
                if hasattr(sensor, 'cfg') and hasattr(sensor.cfg, 'optical_sim_cfg'):
                    if sensor.cfg.optical_sim_cfg is not None:
                        sensor.cfg.optical_sim_cfg.gelpad_to_camera_min_distance = self.gelpad_to_camera_min_distance
                        
                        if hasattr(sensor, 'optical_simulator') and sensor.optical_simulator is not None:
                            if hasattr(sensor.optical_simulator, 'gelpad_to_camera_min_distance'):
                                sensor.optical_simulator.gelpad_to_camera_min_distance = self.gelpad_to_camera_min_distance
                            elif hasattr(sensor.optical_simulator, 'cfg'):
                                sensor.optical_simulator.cfg.gelpad_to_camera_min_distance = self.gelpad_to_camera_min_distance
                
                print(f"  ✓ Updated {sensor_name} sensor")
            except Exception as e:
                import traceback
                print(f"  ✗ Failed to update {sensor_name} sensor: {e}")
                traceback.print_exc()
        
        print("[INFO] Sensor configuration applied successfully!")

    def _zero_ee_offset(self, arm_name: str):
        if ui_utils is None:
            return
        self.ui_window_elements[f"ee_off_{arm_name}_x"].set_value(0.0)
        self.ui_window_elements[f"ee_off_{arm_name}_y"].set_value(0.0)
        self.ui_window_elements[f"ee_off_{arm_name}_z"].set_value(0.0)
        self.ui_window_elements[f"ee_off_{arm_name}_roll"].set_value(0.0)
        self.ui_window_elements[f"ee_off_{arm_name}_pitch"].set_value(0.0)
        self.ui_window_elements[f"ee_off_{arm_name}_yaw"].set_value(0.0)
        self._apply_ee_offset(arm_name)

    def _apply_ee_offset(self, arm_name: str):
        """Apply EE offset to env tensors + cfg (per arm)."""
        env = self.env
        try:
            x = float(self.ui_window_elements[f"ee_off_{arm_name}_x"].get_value_as_float())
            y = float(self.ui_window_elements[f"ee_off_{arm_name}_y"].get_value_as_float())
            z = float(self.ui_window_elements[f"ee_off_{arm_name}_z"].get_value_as_float())
            roll = float(self.ui_window_elements[f"ee_off_{arm_name}_roll"].get_value_as_float()) * np.pi / 180.0
            pitch = float(self.ui_window_elements[f"ee_off_{arm_name}_pitch"].get_value_as_float()) * np.pi / 180.0
            yaw = float(self.ui_window_elements[f"ee_off_{arm_name}_yaw"].get_value_as_float()) * np.pi / 180.0

            r = torch.tensor([roll], device=env.device, dtype=torch.float32)
            p = torch.tensor([pitch], device=env.device, dtype=torch.float32)
            yw = torch.tensor([yaw], device=env.device, dtype=torch.float32)
            q = math_utils.quat_from_euler_xyz(r, p, yw)[0]  # wxyz

            # Update live tensors (used by pose/jacobian calculations)
            pos_t = torch.tensor([x, y, z], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
            quat_t = q.unsqueeze(0).repeat(env.num_envs, 1)

            if arm_name == "left":
                # left (also keep env._offset_pos/_offset_rot in sync for compatibility)
                if hasattr(env, "_offset_pos"):
                    env._offset_pos[:] = pos_t
                if hasattr(env, "_offset_rot"):
                    env._offset_rot[:] = quat_t
                if hasattr(env, "_offset_pos_left"):
                    env._offset_pos_left[:] = pos_t
                if hasattr(env, "_offset_rot_left"):
                    env._offset_rot_left[:] = quat_t
            else:
                # right
                if hasattr(env, "_offset_pos_right"):
                    env._offset_pos_right[:] = pos_t
                if hasattr(env, "_offset_rot_right"):
                    env._offset_rot_right[:] = quat_t

            # Store per-arm in cfg_yaml for snapshot (without mutating full config structure too much)
            if hasattr(env, "_cfg_yaml") and isinstance(env._cfg_yaml, dict):
                eo = env._cfg_yaml.get("end_effector_offset", {})
                if not isinstance(eo, dict):
                    eo = {}
                if "left" not in eo and "right" not in eo:
                    eo = {"left": {}, "right": {}}
                eo.setdefault("left", {})
                eo.setdefault("right", {})
                eo[arm_name] = {"pos": [x, y, z], "quat_wxyz": [float(q[0].item()), float(q[1].item()), float(q[2].item()), float(q[3].item())]}
                env._cfg_yaml["end_effector_offset"] = eo

            print(f"[INFO] Applied EE offset ({arm_name}): pos=({x:.4f},{y:.4f},{z:.4f}), rpy_deg=({roll*180/np.pi:.1f},{pitch*180/np.pi:.1f},{yaw*180/np.pi:.1f})")
        except Exception as e:
            import traceback as _tb
            print(f"[WARNING] Failed to apply EE offset: {e}")
            _tb.print_exc()

    def _save_yaml_snapshot(self):
        """Save current runtime parameters into a new YAML file."""
        if yaml is None:
            print("[WARNING] PyYAML is required to save config. Please install `pyyaml`.")
            return
        try:
            env = self.env
            cfg_in = getattr(env, "_cfg_yaml", {}) or {}

            stage = omni.usd.get_context().get_stage()
            num_envs = int(getattr(env, "num_envs", 1))

            # Helper: env_0 path
            def _env0(path_expr: str) -> str:
                return path_expr.replace("env_.*/", "env_0/") if "env_.*/" in path_expr else path_expr

            # Robot bases: save as env-relative (pos_w - env_origin), quat wxyz from root_link_quat_w
            origins = getattr(env.scene, "env_origins", None)
            origin0 = origins[0] if (origins is not None and origins.numel() > 0) else torch.zeros(3, device=env.device)

            def _xyzw_to_wxyz(q_xyzw):
                return (float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))

            def _pick_wxyz_from_unknown(q4, expected_wxyz=None):
                """Pick the most plausible wxyz from a quaternion that might be wxyz or xyzw."""
                q = [float(q4[0]), float(q4[1]), float(q4[2]), float(q4[3])]
                # Candidate A: interpret as wxyz
                a = q
                # Candidate B: interpret as xyzw and convert -> wxyz
                b = [q[3], q[0], q[1], q[2]]
                if expected_wxyz is None:
                    # Prefer the candidate with larger |w| (identity-like)
                    return a if abs(a[0]) >= abs(b[0]) else b
                ew = [float(expected_wxyz[0]), float(expected_wxyz[1]), float(expected_wxyz[2]), float(expected_wxyz[3])]
                # Compare by quaternion dot-product magnitude (handles q and -q equivalence)
                dot_a = abs(a[0] * ew[0] + a[1] * ew[1] + a[2] * ew[2] + a[3] * ew[3])
                dot_b = abs(b[0] * ew[0] + b[1] * ew[1] + b[2] * ew[2] + b[3] * ew[3])
                return a if dot_a >= dot_b else b

            def _robot_base(robot, expected_rot_xyzw=None):
                pos_w = robot.data.root_link_pos_w[0].detach().cpu()
                quat_raw = robot.data.root_link_quat_w[0].detach().cpu()
                pos_rel = (pos_w - origin0.detach().cpu())
                expected_wxyz = _xyzw_to_wxyz(expected_rot_xyzw) if expected_rot_xyzw is not None else None
                quat_wxyz = _pick_wxyz_from_unknown(quat_raw, expected_wxyz=expected_wxyz)
                return {
                    "pos": [float(pos_rel[0]), float(pos_rel[1]), float(pos_rel[2])],
                    "quat_wxyz": [float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])],
                }

            robot_bases = {
                "left": {**_robot_base(env._robot_left, getattr(env.cfg.robot_left.init_state, "rot", None))},
                "right": {**_robot_base(env._robot_right, getattr(env.cfg.robot_right.init_state, "rot", None))},
            }

            # Joint positions: explicit per joint name
            def _joint_pos_dict(robot):
                names = list(robot.joint_names)
                vals = robot.data.joint_pos[0].detach().cpu().tolist()
                return {names[i]: float(vals[i]) for i in range(min(len(names), len(vals)))}

            robot_bases["left"]["joint_pos"] = _joint_pos_dict(env._robot_left)
            robot_bases["right"]["joint_pos"] = _joint_pos_dict(env._robot_right)

            # End objects world pose
            end_objects = {}
            try:
                p, q = env.end_prim_view_left.get_world_poses()
                end_objects["left"] = {
                    "prim_path": getattr(env, "end_prim_view_left").prim_paths_expr if hasattr(getattr(env, "end_prim_view_left"), "prim_paths_expr") else "/End_Left",
                    "pos_w": [float(x) for x in p[0].detach().cpu().tolist()],
                    "quat_wxyz": [float(x) for x in q[0].detach().cpu().tolist()],
                }
            except Exception:
                pass
            try:
                p, q = env.end_prim_view_right.get_world_poses()
                end_objects["right"] = {
                    "prim_path": getattr(env, "end_prim_view_right").prim_paths_expr if hasattr(getattr(env, "end_prim_view_right"), "prim_paths_expr") else "/End_Right",
                    "pos_w": [float(x) for x in p[0].detach().cpu().tolist()],
                    "quat_wxyz": [float(x) for x in q[0].detach().cpu().tolist()],
                }
            except Exception:
                pass

            # Tactile sensors: read from live sensor cfg (after GUI apply)
            sensors_live = {
                "ff_left": env.gelsighthand_ff_left,
                "mf_left": env.gelsighthand_mf_left,
                "rf_left": env.gelsighthand_rf_left,
                "lf_left": env.gelsighthand_lf_left,
                "th_left": env.gelsighthand_th_left,
                "ff_right": env.gelsighthand_ff_right,
                "mf_right": env.gelsighthand_mf_right,
                "rf_right": env.gelsighthand_rf_right,
                "lf_right": env.gelsighthand_lf_right,
                "th_right": env.gelsighthand_th_right,
            }

            # Defaults from first sensor
            any_sensor = next(iter(sensors_live.values()))
            defaults = {}
            if hasattr(any_sensor, "cfg") and hasattr(any_sensor.cfg, "sensor_camera_cfg"):
                defaults["camera_resolution"] = [int(any_sensor.cfg.sensor_camera_cfg.resolution[0]), int(any_sensor.cfg.sensor_camera_cfg.resolution[1])]
                cr = any_sensor.cfg.sensor_camera_cfg.clipping_range
                defaults["clipping_range"] = [float(cr[0]), float(cr[1])]
            if hasattr(any_sensor, "cfg") and hasattr(any_sensor.cfg, "optical_sim_cfg") and any_sensor.cfg.optical_sim_cfg is not None:
                defaults["gelpad_to_camera_min_distance"] = float(any_sensor.cfg.optical_sim_cfg.gelpad_to_camera_min_distance)
                tir = any_sensor.cfg.optical_sim_cfg.tactile_img_res
                defaults["tactile_img_resolution"] = [int(tir[0]), int(tir[1])]

            # Camera local pose snapshot (store as enabled=true)
            defaults["camera_pose_local"] = {"enabled": False, "pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}

            per_sensors = {}
            for key, s in sensors_live.items():
                scfg = getattr(s, "cfg", None)
                per = {}
                if scfg is not None and hasattr(scfg, "sensor_camera_cfg"):
                    per["camera_resolution"] = [int(scfg.sensor_camera_cfg.resolution[0]), int(scfg.sensor_camera_cfg.resolution[1])]
                    cr = scfg.sensor_camera_cfg.clipping_range
                    per["clipping_range"] = [float(cr[0]), float(cr[1])]
                if scfg is not None and hasattr(scfg, "optical_sim_cfg") and scfg.optical_sim_cfg is not None:
                    per["gelpad_to_camera_min_distance"] = float(scfg.optical_sim_cfg.gelpad_to_camera_min_distance)
                    tir = scfg.optical_sim_cfg.tactile_img_res
                    per["tactile_img_resolution"] = [int(tir[0]), int(tir[1])]

                # Read camera local pose from USD (only reliable for num_envs==1)
                cam_expr = f"{getattr(scfg, 'prim_path', '')}/Camera" if scfg is not None else ""
                if cam_expr:
                    cam_path = _env0(cam_expr) if num_envs == 1 else None
                    if cam_path is not None:
                        pose = _read_local_pose_wxyz(stage, cam_path)
                        if pose is not None:
                            pos, quat = pose
                            per["camera_pose_local"] = {
                                "enabled": True,
                                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                                "quat_wxyz": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                            }
                per_sensors[key] = per

            tactile_sensors = {"defaults": defaults, "sensors": per_sensors}

            # Env camera (viewer uses eye/lookat; arm_camera uses offset)
            env_camera = {
                "viewer": {
                    "eye": [float(x) for x in getattr(env.cfg.viewer, "eye", (1.5, 1.0, 0.5))],
                    "lookat": [float(x) for x in getattr(env.cfg.viewer, "lookat", (0.2, 0.0, 0.1))],
                },
                "arm_camera": {
                    "pos": [float(x) for x in env.cfg.arm_camera.offset.pos],
                    "quat_wxyz": [float(x) for x in env.cfg.arm_camera.offset.rot],
                    "resolution": [int(env.cfg.arm_camera.width), int(env.cfg.arm_camera.height)],
                    "clipping_range": [float(env.cfg.arm_camera.spawn.clipping_range[0]), float(env.cfg.arm_camera.spawn.clipping_range[1])],
                },
            }

            # Assets
            assets = {
                "ground_plate": {
                    "usd_path": str(getattr(env.cfg.plate.spawn, "usd_path", "")),
                    "pos": [float(x) for x in env.cfg.plate.init_state.pos],
                    "scale": [float(x) for x in getattr(env.cfg.plate.spawn, "scale", (1.0, 1.0, 1.0))],
                },
                "object": {
                    "usd_path": str(getattr(env.cfg.object.spawn, "usd_path", "")),
                    "pos": [float(x) for x in env.cfg.object.init_state.pos],
                    "scale": [float(x) for x in getattr(env.cfg.object.spawn, "scale", (1.0, 1.0, 1.0))],
                },
            }

            # EE offset (task frame) - per arm (read from live tensors)
            def _tensor3(t):
                return [float(x) for x in t[0].detach().cpu().tolist()]

            def _tensor4(t):
                return [float(x) for x in t[0].detach().cpu().tolist()]

            left_pos = _tensor3(getattr(env, "_offset_pos_left", getattr(env, "_offset_pos", torch.zeros((env.num_envs, 3), device=env.device))))
            left_quat = _tensor4(getattr(env, "_offset_rot_left", getattr(env, "_offset_rot", torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1))))
            right_pos = _tensor3(getattr(env, "_offset_pos_right", getattr(env, "_offset_pos_left", torch.zeros((env.num_envs, 3), device=env.device))))
            right_quat = _tensor4(getattr(env, "_offset_rot_right", getattr(env, "_offset_rot_left", torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1))))

            end_effector_offset = {
                "left": {"pos": left_pos, "quat_wxyz": left_quat},
                "right": {"pos": right_pos, "quat_wxyz": right_quat},
            }

            # Finger poses: save current finger_targets as closed per-arm (pattern keys from FIXED_GRASP_POSE)
            def _finger_pose_from_targets(robot, finger_joint_ids, finger_targets_row):
                names = list(robot.joint_names)
                out = {}
                for pat in FIXED_GRASP_POSE.keys():
                    v = None
                    for i, j_idx in enumerate(finger_joint_ids):
                        jn = names[int(j_idx)]
                        if pat.upper() in jn.upper() or jn.upper() in pat.upper():
                            v = float(finger_targets_row[i].item())
                            break
                    if v is not None:
                        out[pat] = v
                return out

            left_ids = getattr(env, "_finger_joint_ids_left", torch.tensor([], device=env.device))
            right_ids = getattr(env, "_finger_joint_ids_right", torch.tensor([], device=env.device))
            left_closed = _finger_pose_from_targets(env._robot_left, left_ids.tolist(), env.finger_targets_left[0])
            right_closed = _finger_pose_from_targets(env._robot_right, right_ids.tolist(), env.finger_targets_right[0])

            finger_poses = {
                "initial": "closed",
                "closed": {"joint_pos": FIXED_GRASP_POSE, "per_arm": {"left": left_closed, "right": right_closed}},
                "open": (cfg_in.get("finger_poses", {}) or {}).get("open", {"mode": "lower_limits", "joint_pos": {}, "per_arm": {"left": {}, "right": {}}}),
            }

            # Preserve arm_joint_presets from input config if present
            arm_joint_presets = cfg_in.get("arm_joint_presets", {}) if isinstance(cfg_in, dict) else {}

            # Preserve ik_controller overrides
            ik_controller = cfg_in.get("ik_controller", {}) if isinstance(cfg_in, dict) else {"overrides": {}}

            out_cfg = {
                "robot_bases": robot_bases,
                "arm_joint_presets": arm_joint_presets,
                "end_objects": end_objects,
                "end_effector_offset": end_effector_offset,
                "tactile_sensors": tactile_sensors,
                "env_camera": env_camera,
                "assets": assets,
                "finger_poses": finger_poses,
                "ik_controller": ik_controller,
            }

            # Determine output path
            base_cfg_path = (
                Path(args_cli.config).expanduser().resolve()
                if getattr(args_cli, "config", None)
                else Path.cwd() / "dual_arm_gui_pose_config.yaml"
            )

            save_cfg = (cfg_in.get("save_config", {}) if isinstance(cfg_in, dict) else {}) or {}
            output_dir = save_cfg.get("output_dir", "") if isinstance(save_cfg, dict) else ""
            prefix = save_cfg.get("filename_prefix", "dual_arm_gui_pose_config_snapshot_") if isinstance(save_cfg, dict) else "dual_arm_gui_pose_config_snapshot_"
            ts_fmt = save_cfg.get("timestamp_format", "%Y%m%d_%H%M%S") if isinstance(save_cfg, dict) else "%Y%m%d_%H%M%S"
            overwrite = bool(save_cfg.get("overwrite_config", False)) if isinstance(save_cfg, dict) else False

            if overwrite:
                out_path = base_cfg_path
            else:
                if output_dir and isinstance(output_dir, str):
                    out_dir = Path(_expand_env_vars(output_dir)).expanduser()
                else:
                    out_dir = base_cfg_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.datetime.now().strftime(str(ts_fmt))
                out_path = out_dir / f"{prefix}{ts}.yaml"

            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(out_cfg, f, sort_keys=False, allow_unicode=True)

            print(f"[INFO] Saved YAML snapshot to: {out_path}")
        except Exception as e:
            import traceback as _tb
            print(f"[WARNING] Failed to save YAML snapshot: {e}")
            _tb.print_exc()
    
    def _build_finger_joint_control_frame(self, arm_name: str):
        """Build finger joint control UI frame for specified arm."""
        if ui_utils is None:
            return
        
        import omni.ui
        
        # Get arm-specific attributes
        if arm_name == "left":
            finger_joint_names = self.env._finger_joint_names_left if hasattr(self.env, '_finger_joint_names_left') else []
            finger_joint_ids = self.env._finger_joint_ids_left if hasattr(self.env, '_finger_joint_ids_left') else []
            robot = self.env._robot_left
            finger_targets = self.env.finger_targets_left if hasattr(self.env, 'finger_targets_left') else None
            controls_dict = self.finger_joint_controls_left
        else:  # right
            finger_joint_names = self.env._finger_joint_names_right if hasattr(self.env, '_finger_joint_names_right') else []
            finger_joint_ids = self.env._finger_joint_ids_right if hasattr(self.env, '_finger_joint_ids_right') else []
            robot = self.env._robot_right
            finger_targets = self.env.finger_targets_right if hasattr(self.env, 'finger_targets_right') else None
            controls_dict = self.finger_joint_controls_right
        
        # Wait for env to be initialized
        if len(finger_joint_names) == 0:
            self.ui_window_elements[f"finger_joint_frame_{arm_name}"] = omni.ui.CollapsableFrame(
                title=f"Finger Joint Control - {arm_name.upper()} Arm (Not Available)",
                width=omni.ui.Fraction(1),
                height=0,
                collapsed=True,
                style=ui_utils.get_style(),
            )
            return
        
        self.ui_window_elements[f"finger_joint_frame_{arm_name}"] = omni.ui.CollapsableFrame(
            title=f"Finger Joint Control - {arm_name.upper()} Arm ({len(finger_joint_names)} joints)",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=arm_name != "left",  # Expand left arm by default
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        
        with self.ui_window_elements[f"finger_joint_frame_{arm_name}"]:
            self.ui_window_elements[f"finger_joint_vstack_{arm_name}"] = omni.ui.VStack(spacing=5, height=50)
            with self.ui_window_elements[f"finger_joint_vstack_{arm_name}"]:
                # Group joints by finger
                finger_groups = {
                    "WRJ": [],
                    "FFJ": [],
                    "MFJ": [],
                    "RFJ": [],
                    "LFJ": [],
                    "THJ": [],
                }
                
                for i, joint_name in enumerate(finger_joint_names):
                    for prefix in finger_groups.keys():
                        if prefix in joint_name.upper():
                            finger_groups[prefix].append((i, joint_name))
                            break
                    else:
                        if "WRJ" not in [g[1] for g in finger_groups["WRJ"]]:
                            finger_groups["WRJ"].append((i, joint_name))
                
                # Create controls for each joint
                for prefix, joints in finger_groups.items():
                    if len(joints) == 0:
                        continue
                    
                    finger_name_map = {
                        "WRJ": "Wrist",
                        "FFJ": "Index Finger (FF)",
                        "MFJ": "Middle Finger (MF)",
                        "RFJ": "Ring Finger (RF)",
                        "LFJ": "Little Finger (LF)",
                        "THJ": "Thumb (TH)",
                    }
                    
                    finger_frame = omni.ui.CollapsableFrame(
                        title=finger_name_map.get(prefix, prefix),
                        width=omni.ui.Fraction(1),
                        height=0,
                        collapsed=prefix != "FFJ",
                        style=ui_utils.get_style(),
                    )
                    
                    with finger_frame:
                        finger_vstack = omni.ui.VStack(spacing=3, height=50)
                        with finger_vstack:
                            for joint_idx, joint_name in joints:
                                if finger_targets is not None and finger_targets.shape[1] > joint_idx:
                                    current_val = finger_targets[0, joint_idx].item()
                                else:
                                    current_val = 0.0
                                
                                if robot is not None and hasattr(robot, 'data'):
                                    joint_id = finger_joint_ids[joint_idx].item()
                                    lower_limit = robot.data.soft_joint_pos_limits[0, joint_id, 0].item()
                                    upper_limit = robot.data.soft_joint_pos_limits[0, joint_id, 1].item()
                                else:
                                    lower_limit = -1.57
                                    upper_limit = 1.57
                                
                                joint_control = ui_utils.combo_floatfield_slider_builder(
                                    label=f"{joint_name}",
                                    default_val=current_val,
                                    min=lower_limit,
                                    max=upper_limit,
                                    step=0.01,
                                    tooltip=f"Control {joint_name} joint (range: [{lower_limit:.3f}, {upper_limit:.3f}] rad)",
                                )[0]
                                
                                controls_dict[joint_idx] = {
                                    'control': joint_control,
                                    'joint_name': joint_name,
                                    'joint_idx': joint_idx,
                                }
                                
                                def make_update_callback(env_ref, idx, arm):
                                    def update_joint(value_model):
                                        if arm == "left" and hasattr(env_ref, 'finger_targets_left'):
                                            new_value = value_model.get_value_as_float()
                                            env_ref.finger_targets_left[0, idx] = new_value
                                        elif arm == "right" and hasattr(env_ref, 'finger_targets_right'):
                                            new_value = value_model.get_value_as_float()
                                            env_ref.finger_targets_right[0, idx] = new_value
                                    return update_joint
                                
                                joint_control.add_value_changed_fn(make_update_callback(self.env, joint_idx, arm_name))
                
                # Add "Reset to Open" button
                self.ui_window_elements[f"reset_fingers_open_button_{arm_name}"] = ui_utils.btn_builder(
                    type="button",
                    text=f"Reset All Fingers to Open ({arm_name.upper()})",
                    tooltip=f"Resets all finger joints to their minimum (open) positions for {arm_name} arm.",
                    on_clicked_fn=lambda: self._reset_fingers_open(arm_name),
                )
                
                # Add "Reset to Fixed Grasp" button
                if hasattr(self.env, 'cfg') and hasattr(self.env.cfg, 'manual_finger_joint_positions_left') and self.env.cfg.manual_finger_joint_positions_left is not None:
                    self.ui_window_elements[f"reset_fingers_grasp_button_{arm_name}"] = ui_utils.btn_builder(
                        type="button",
                        text=f"Reset to Fixed Grasp Pose ({arm_name.upper()})",
                        tooltip=f"Resets all finger joints to the fixed grasp pose for {arm_name} arm.",
                        on_clicked_fn=lambda: self._reset_fingers_grasp(arm_name),
                    )
    
    def _reset_fingers_open(self, arm_name: str):
        """Reset all finger joints to open (minimum) position for specified arm."""
        if arm_name == "left":
            finger_joint_ids = self.env._finger_joint_ids_left if hasattr(self.env, '_finger_joint_ids_left') else []
            robot = self.env._robot_left
            controls_dict = self.finger_joint_controls_left
            finger_targets = self.env.finger_targets_left
            open_cfg = getattr(self.env, "_finger_pose_open_left", None)
        else:
            finger_joint_ids = self.env._finger_joint_ids_right if hasattr(self.env, '_finger_joint_ids_right') else []
            robot = self.env._robot_right
            controls_dict = self.finger_joint_controls_right
            finger_targets = self.env.finger_targets_right
            open_cfg = getattr(self.env, "_finger_pose_open_right", None)
        
        if len(finger_joint_ids) == 0:
            return
        
        try:
            mode = None
            explicit = None
            if isinstance(open_cfg, dict):
                mode = open_cfg.get("mode", None)
                explicit = open_cfg.get("joint_pos", None)
            if mode == "explicit" and isinstance(explicit, dict) and len(explicit) > 0:
                joint_names = robot.joint_names
                vals = torch.zeros((len(finger_joint_ids),), device=self.env.device)
                for i, j_idx in enumerate(finger_joint_ids):
                    jn = joint_names[j_idx]
                    v = None
                    if jn in explicit:
                        v = explicit[jn]
                    else:
                        for pat, pv in explicit.items():
                            if str(pat).upper() in jn.upper() or jn.upper() in str(pat).upper():
                                v = pv
                                break
                    if v is None:
                        vals[i] = robot.data.soft_joint_pos_limits[0, j_idx, 0]
                    else:
                        vals[i] = float(v)
                finger_targets[0] = vals
            else:
                finger_lower_limits = robot.data.soft_joint_pos_limits[0, finger_joint_ids, 0]
                finger_targets[0] = finger_lower_limits
            
            for joint_idx, control_info in controls_dict.items():
                if joint_idx < finger_targets.shape[1]:
                    control_info['control'].set_value(finger_targets[0, joint_idx].item())
            
            print(f"[INFO] Reset all fingers to open position ({arm_name} arm)")
        except Exception as e:
            print(f"[WARNING] Failed to reset fingers ({arm_name} arm): {e}")
    
    def _reset_fingers_grasp(self, arm_name: str):
        """Reset all finger joints to fixed grasp pose for specified arm."""
        if arm_name == "left":
            cfg_attr = 'manual_finger_joint_positions_left'
            finger_joint_ids = self.env._finger_joint_ids_left if hasattr(self.env, '_finger_joint_ids_left') else []
            robot = self.env._robot_left
            controls_dict = self.finger_joint_controls_left
            finger_targets = self.env.finger_targets_left
            grasp_pose = getattr(self.env, "_finger_pose_closed_left", None)
        else:
            cfg_attr = 'manual_finger_joint_positions_right'
            finger_joint_ids = self.env._finger_joint_ids_right if hasattr(self.env, '_finger_joint_ids_right') else []
            robot = self.env._robot_right
            controls_dict = self.finger_joint_controls_right
            finger_targets = self.env.finger_targets_right
            grasp_pose = getattr(self.env, "_finger_pose_closed_right", None)
        
        if not hasattr(self.env, 'cfg') or not hasattr(self.env.cfg, cfg_attr):
            return
        
        fixed_pose = grasp_pose if isinstance(grasp_pose, dict) else getattr(self.env.cfg, cfg_attr)
        if fixed_pose is None:
            return
        
        try:
            joint_names = robot.joint_names
            
            for i, joint_idx in enumerate(finger_joint_ids):
                joint_name = joint_names[joint_idx]
                matched = False
                for pattern, value in fixed_pose.items():
                    if pattern.upper() in joint_name.upper() or joint_name.upper() in pattern.upper():
                        finger_targets[0, i] = value
                        if i in controls_dict:
                            controls_dict[i]['control'].set_value(value)
                        matched = True
                        break
            
            print(f"[INFO] Reset all fingers to fixed grasp pose ({arm_name} arm)")
        except Exception as e:
            print(f"[WARNING] Failed to reset fingers to grasp pose ({arm_name} arm): {e}")


def run_simulator(env: ShadowHandDualArmPickUpObjectScriptedEnv):
    """Runs the simulation loop for folder manipulation."""
    print(f"Starting ShadowHand dual-arm folder manipulation simulation with {env.num_envs} envs")
    print(f"Folder USD path: {args_cli.folder_usd_path}")
    print(f"Folder position: {args_cli.folder_pos}")
    print(f"Folder scale: {args_cli.folder_scale}")
    
    try:
        env.reset()

        # Attach YAML to env
        if not hasattr(env, "_cfg_yaml"):
            env._cfg_yaml = CFG

        # Apply YAML initial joint poses
        try:
            left_joint_cfg = _get_cfg(getattr(env, "_cfg_yaml", {}), ["robot_bases", "left", "joint_pos"], {}) or {}
            right_joint_cfg = _get_cfg(getattr(env, "_cfg_yaml", {}), ["robot_bases", "right", "joint_pos"], {}) or {}
            left_arm_preset = None
            right_arm_preset = None
            if _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "left", "enabled"], False):
                left_arm_preset = _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "left", "joints_6"], None)
            if _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "right", "enabled"], False):
                right_arm_preset = _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "right", "joints_6"], None)
            _apply_robot_joint_pos_overrides(env, "left", left_joint_cfg, left_arm_preset)
            _apply_robot_joint_pos_overrides(env, "right", right_joint_cfg, right_arm_preset)
        except Exception as e:
            print(f"[WARNING] Failed to apply YAML joint_pos at startup: {e}")
        
        # Create End virtual objects for interactive control
        end_left_cfg = _get_cfg(getattr(env, "_cfg_yaml", {}), ["end_objects", "left"], {}) or {}
        end_right_cfg = _get_cfg(getattr(env, "_cfg_yaml", {}), ["end_objects", "right"], {}) or {}
        initial_end_pos_left = np.array(end_left_cfg.get("pos_w", [0.5, 0.0, 0.3]), dtype=np.float32)
        initial_end_quat_left = np.array(end_left_cfg.get("quat_wxyz", [0.70711, 0.70711, 0.0, 0.0]), dtype=np.float32)
        initial_end_pos_right = np.array(end_right_cfg.get("pos_w", [1.5, 0.0, 0.3]), dtype=np.float32)
        initial_end_quat_right = np.array(end_right_cfg.get("quat_wxyz", [0.70711, 0.70711, 0.0, 0.0]), dtype=np.float32)
        end_left_prim_path = end_left_cfg.get("prim_path", "/End_Left")
        end_right_prim_path = end_right_cfg.get("prim_path", "/End_Right")
        end_left_color = np.array(end_left_cfg.get("color_rgb", [0.0, 255.0, 0.0]), dtype=np.float32)
        end_right_color = np.array(end_right_cfg.get("color_rgb", [255.0, 0.0, 0.0]), dtype=np.float32)
        
        VisualCuboid(
            prim_path=end_left_prim_path,
            size=0.02,
            position=initial_end_pos_left,
            orientation=initial_end_quat_left,
            visible=True,
            color=end_left_color,
        )
        
        VisualCuboid(
            prim_path=end_right_prim_path,
            size=0.02,
            position=initial_end_pos_right,
            orientation=initial_end_quat_right,
            visible=True,
            color=end_right_color,
        )
        
        env.end_prim_view_left = XFormPrim(prim_paths_expr=end_left_prim_path, name="End_Left", usd=True)
        env.end_prim_view_right = XFormPrim(prim_paths_expr=end_right_prim_path, name="End_Right", usd=True)
        env.ik_commands_left = torch.zeros((env.num_envs, env._ik_controller_left.action_dim), device=env.device)
        env.ik_commands_right = torch.zeros((env.num_envs, env._ik_controller_right.action_dim), device=env.device)
        
        env.scene.update(dt=0.0)
        initial_end_pos_tensor_left = torch.tensor(initial_end_pos_left, device=env.device, dtype=torch.float32).unsqueeze(0)
        initial_end_quat_tensor_left = torch.tensor(initial_end_quat_left, device=env.device, dtype=torch.float32).unsqueeze(0)
        env.end_prim_view_left.set_world_poses(
            positions=initial_end_pos_tensor_left,
            orientations=initial_end_quat_tensor_left
        )
        
        initial_end_pos_tensor_right = torch.tensor(initial_end_pos_right, device=env.device, dtype=torch.float32).unsqueeze(0)
        initial_end_quat_tensor_right = torch.tensor(initial_end_quat_right, device=env.device, dtype=torch.float32).unsqueeze(0)
        env.end_prim_view_right.set_world_poses(
            positions=initial_end_pos_tensor_right,
            orientations=initial_end_quat_tensor_right
        )
        
        env.scene.update(dt=0.0)
        env.sim.step(render=False)
        env.scene.update(dt=0.0)
        
        # Initialize ik_commands with initial target position
        positions, orientations = env.end_prim_view_left.get_world_poses()
        root_pos_w = env._robot_left.data.root_link_pos_w
        root_quat_w = env._robot_left.data.root_link_quat_w
        end_pos_b, end_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, positions, orientations
        )
        env.ik_commands_left[:, :3] = end_pos_b
        env.ik_commands_left[:, 3:] = end_quat_b
        
        positions, orientations = env.end_prim_view_right.get_world_poses()
        root_pos_w = env._robot_right.data.root_link_pos_w
        root_quat_w = env._robot_right.data.root_link_quat_w
        end_pos_b, end_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, positions, orientations
        )
        env.ik_commands_right[:, :3] = end_pos_b
        env.ik_commands_right[:, 3:] = end_quat_b
        
        env.scene.update(dt=0.0)
        end_pos_w_left, end_quat_w_left = env.end_prim_view_left.get_world_poses()
        end_pos_w_right, end_quat_w_right = env.end_prim_view_right.get_world_poses()
        print(f"\n[INFO] Created '/End_Left' (green) at world position {end_pos_w_left[0].cpu().numpy()}.")
        print(f"[INFO] Created '/End_Right' (red) at world position {end_pos_w_right[0].cpu().numpy()}.")
        print("[INFO] Move them in Isaac Sim to control robot end-effectors.")
        
        # Print folder information
        if hasattr(env, 'folder'):
            folder_pos_w = env.folder.data.root_link_pos_w
            print(f"[INFO] Folder position: {folder_pos_w[0].cpu().numpy()}")
            print(f"[INFO] Folder is ready for manipulation.")
        
        # Simulation loop
        step_count = 0
        should_exit = False
        while simulation_app.is_running() and not should_exit:
            if env._window is not None and env._window.reset:
                print("\n" + "-" * 80)
                print("[INFO]: Resetting environment...")
                env._window.reset = False
                env.reset()
                step_count = 0
                
                # Apply YAML initial joint poses
                try:
                    left_joint_cfg = _get_cfg(getattr(env, "_cfg_yaml", {}), ["robot_bases", "left", "joint_pos"], {}) or {}
                    right_joint_cfg = _get_cfg(getattr(env, "_cfg_yaml", {}), ["robot_bases", "right", "joint_pos"], {}) or {}
                    left_arm_preset = None
                    right_arm_preset = None
                    if _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "left", "enabled"], False):
                        left_arm_preset = _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "left", "joints_6"], None)
                    if _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "right", "enabled"], False):
                        right_arm_preset = _get_cfg(getattr(env, "_cfg_yaml", {}), ["arm_joint_presets", "right", "joints_6"], None)
                    _apply_robot_joint_pos_overrides(env, "left", left_joint_cfg, left_arm_preset)
                    _apply_robot_joint_pos_overrides(env, "right", right_joint_cfg, right_arm_preset)
                except Exception as e:
                    print(f"[WARNING] Failed to apply YAML joint_pos on reset: {e}")
                
                # Reset End positions
                env.scene.update(dt=0.0)
                initial_end_pos_tensor_left = torch.tensor(initial_end_pos_left, device=env.device, dtype=torch.float32).unsqueeze(0)
                initial_end_quat_tensor_left = torch.tensor(initial_end_quat_left, device=env.device, dtype=torch.float32).unsqueeze(0)
                env.end_prim_view_left.set_world_poses(
                    positions=initial_end_pos_tensor_left,
                    orientations=initial_end_quat_tensor_left
                )
                
                initial_end_pos_tensor_right = torch.tensor(initial_end_pos_right, device=env.device, dtype=torch.float32).unsqueeze(0)
                initial_end_quat_tensor_right = torch.tensor(initial_end_quat_right, device=env.device, dtype=torch.float32).unsqueeze(0)
                env.end_prim_view_right.set_world_poses(
                    positions=initial_end_pos_tensor_right,
                    orientations=initial_end_quat_tensor_right
                )
                
                env.scene.update(dt=0.0)
                
                # Update ik_commands
                positions, orientations = env.end_prim_view_left.get_world_poses()
                root_pos_w = env._robot_left.data.root_link_pos_w
                root_quat_w = env._robot_left.data.root_link_quat_w
                end_pos_b, end_quat_b = math_utils.subtract_frame_transforms(
                    root_pos_w, root_quat_w, positions, orientations
                )
                env.ik_commands_left[:, :3] = end_pos_b
                env.ik_commands_left[:, 3:] = end_quat_b
                
                positions, orientations = env.end_prim_view_right.get_world_poses()
                root_pos_w = env._robot_right.data.root_link_pos_w
                root_quat_w = env._robot_right.data.root_link_quat_w
                end_pos_b, end_quat_b = math_utils.subtract_frame_transforms(
                    root_pos_w, root_quat_w, positions, orientations
                )
                env.ik_commands_right[:, :3] = end_pos_b
                env.ik_commands_right[:, 3:] = end_quat_b
            
            # Step simulation
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            
            # Update isaac buffers
            env.scene.update(dt=env.physics_dt)
            
            # Read End virtual object positions and update ik_commands
            positions, orientations = env.end_prim_view_left.get_world_poses()
            root_pos_w = env._robot_left.data.root_link_pos_w
            root_quat_w = env._robot_left.data.root_link_quat_w
            end_pos_b, end_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, positions, orientations
            )
            env.ik_commands_left[:, :3] = end_pos_b
            env.ik_commands_left[:, 3:] = end_quat_b
            
            positions, orientations = env.end_prim_view_right.get_world_poses()
            root_pos_w = env._robot_right.data.root_link_pos_w
            root_quat_w = env._robot_right.data.root_link_quat_w
            end_pos_b, end_quat_b = math_utils.subtract_frame_transforms(
                root_pos_w, root_quat_w, positions, orientations
            )
            env.ik_commands_right[:, :3] = end_pos_b
            env.ik_commands_right[:, 3:] = end_quat_b
            
            # Render scene
            env.sim.render()
            
            step_count += 1
            if step_count % 1000 == 0:
                obs = env._get_observations()
                print(f"\nStep {step_count}: Observation shape - policy: {obs['policy']['proprio_obs'].shape}, vision: {obs['policy']['vision_obs'].shape}")
                
                # Print folder position
                if hasattr(env, 'folder'):
                    folder_pos_w = env.folder.data.root_link_pos_w
                    avg_folder_height = folder_pos_w[:, 2].mean().item()
                    print(f"  Average folder height: {avg_folder_height:.4f}")
    finally:
        pass
    
    env.close()
    pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    env_cfg = ShadowHandDualArmPickUpObjectScriptedEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Apply viewer camera from YAML if present
    v_eye = _get_cfg(CFG, ["env_camera", "viewer", "eye"], None)
    v_look = _get_cfg(CFG, ["env_camera", "viewer", "lookat"], None)
    env_cfg.viewer.eye = _as_tuple3(v_eye, (1.5, 1.0, 0.5))
    env_cfg.viewer.lookat = _as_tuple3(v_look, (0.2, 0.0, 0.1))
    
    # Set simulation device
    env_cfg.sim.device = "cpu"
    
    # Override action_space to 0 for IK control
    env_cfg.action_space = 0

    # Apply YAML robot base poses
    rb_left = _get_cfg(CFG, ["robot_bases", "left"], {}) or {}
    rb_right = _get_cfg(CFG, ["robot_bases", "right"], {}) or {}
    if isinstance(rb_left, dict):
        pos = _as_tuple3(rb_left.get("pos"), env_cfg.robot_left.init_state.pos)
        quat_wxyz = _as_tuple4_wxyz(rb_left.get("quat_wxyz"), (1.0, 0.0, 0.0, 0.0))
        env_cfg.robot_left.init_state.pos = pos
        env_cfg.robot_left.init_state.rot = _wxyz_to_xyzw(quat_wxyz)
    if isinstance(rb_right, dict):
        pos = _as_tuple3(rb_right.get("pos"), env_cfg.robot_right.init_state.pos)
        quat_wxyz = _as_tuple4_wxyz(rb_right.get("quat_wxyz"), (1.0, 0.0, 0.0, 0.0))
        env_cfg.robot_right.init_state.pos = pos
        env_cfg.robot_right.init_state.rot = _wxyz_to_xyzw(quat_wxyz)

    # Apply end-effector offset
    ee_off = _get_cfg(CFG, ["end_effector_offset"], {}) or {}
    if isinstance(ee_off, dict) and ("left" in ee_off or "right" in ee_off):
        ee_left = ee_off.get("left", {}) if isinstance(ee_off.get("left", {}), dict) else {}
        ee_right = ee_off.get("right", {}) if isinstance(ee_off.get("right", {}), dict) else {}
        left_pos = _as_tuple3(ee_left.get("pos"), getattr(env_cfg, "body_offset_pos", (0.0, 0.0, 0.0)))
        left_quat = _as_tuple4_wxyz(ee_left.get("quat_wxyz"), getattr(env_cfg, "body_offset_rot", (1.0, 0.0, 0.0, 0.0)))
        right_pos = _as_tuple3(ee_right.get("pos"), left_pos)
        right_quat = _as_tuple4_wxyz(ee_right.get("quat_wxyz"), left_quat)
    else:
        left_pos = _as_tuple3(ee_off.get("pos") if isinstance(ee_off, dict) else None, getattr(env_cfg, "body_offset_pos", (0.0, 0.0, 0.0)))
        left_quat = _as_tuple4_wxyz(ee_off.get("quat_wxyz") if isinstance(ee_off, dict) else None, getattr(env_cfg, "body_offset_rot", (1.0, 0.0, 0.0, 0.0)))
        right_pos = left_pos
        right_quat = left_quat

    env_cfg.body_offset_pos = left_pos
    env_cfg.body_offset_rot = left_quat

    # Apply YAML tactile sensor defaults
    t_defaults = _get_cfg(CFG, ["tactile_sensors", "defaults"], {}) or {}
    t_per = _get_cfg(CFG, ["tactile_sensors", "sensors"], {}) or {}
    if not isinstance(t_defaults, dict):
        t_defaults = {}
    if not isinstance(t_per, dict):
        t_per = {}

    sensor_map = {
        "ff_left": env_cfg.gelsighthand_ff_left,
        "mf_left": env_cfg.gelsighthand_mf_left,
        "rf_left": env_cfg.gelsighthand_rf_left,
        "lf_left": env_cfg.gelsighthand_lf_left,
        "th_left": env_cfg.gelsighthand_th_left,
        "ff_right": env_cfg.gelsighthand_ff_right,
        "mf_right": env_cfg.gelsighthand_mf_right,
        "rf_right": env_cfg.gelsighthand_rf_right,
        "lf_right": env_cfg.gelsighthand_lf_right,
        "th_right": env_cfg.gelsighthand_th_right,
    }
    for k, scfg in sensor_map.items():
        per = t_per.get(k, {}) if isinstance(t_per.get(k, {}), dict) else {}
        _apply_sensor_cfg_overrides(scfg, per, t_defaults)

    # Apply folder configuration
    folder_cfg = _get_cfg(CFG, ["folder"], {}) or {}
    if isinstance(folder_cfg, dict):
        folder_usd = folder_cfg.get("usd_path", None)
        folder_pos = folder_cfg.get("pos", None)
        folder_scale = folder_cfg.get("scale", None)
        disable_fixed_joints = folder_cfg.get("disable_fixed_joints", None)
        
        # 质量/密度配置
        folder_mass_density = folder_cfg.get("folder_mass_density", None)
        folder_mass = folder_cfg.get("folder_mass", None)
        
        # 关节配置
        folder_joint_name = folder_cfg.get("folder_joint_name", None)
        set_folder_joint_limits_in_usd = folder_cfg.get("set_folder_joint_limits_in_usd", None)
        folder_joint_min_angle = folder_cfg.get("folder_joint_min_angle", None)
        folder_joint_max_angle = folder_cfg.get("folder_joint_max_angle", None)
        folder_joint_initial_angle = folder_cfg.get("folder_joint_initial_angle", None)
        folder_joint_drive_stiffness = folder_cfg.get("folder_joint_drive_stiffness", None)
        folder_joint_drive_damping = folder_cfg.get("folder_joint_drive_damping", None)
        
        if folder_usd is not None:
            env_cfg.folder.spawn.usd_path = _expand_env_vars(str(folder_usd))
        else:
            env_cfg.folder.spawn.usd_path = _expand_env_vars(args_cli.folder_usd_path)
        
        if folder_pos is not None:
            env_cfg.folder.init_state.pos = _as_tuple3(folder_pos, env_cfg.folder.init_state.pos)
        else:
            env_cfg.folder.init_state.pos = tuple(args_cli.folder_pos)
        
        if folder_scale is not None:
            env_cfg.folder.spawn.scale = _as_tuple3(folder_scale, env_cfg.folder.spawn.scale)
        else:
            env_cfg.folder.spawn.scale = tuple(args_cli.folder_scale)
        
        if disable_fixed_joints is not None:
            env_cfg.disable_fixed_joints = bool(disable_fixed_joints)
        else:
            env_cfg.disable_fixed_joints = args_cli.disable_fixed_joints
        
        # 应用质量/密度配置
        if folder_mass_density is not None:
            env_cfg.folder_mass_density = float(folder_mass_density) if folder_mass_density is not None else None
        if folder_mass is not None:
            env_cfg.folder_mass = float(folder_mass) if folder_mass is not None else None
        
        # 应用关节配置
        if folder_joint_name is not None:
            env_cfg.folder_joint_name = str(folder_joint_name)
        if set_folder_joint_limits_in_usd is not None:
            env_cfg.set_folder_joint_limits_in_usd = bool(set_folder_joint_limits_in_usd)
        if folder_joint_min_angle is not None:
            env_cfg.folder_joint_min_angle = float(folder_joint_min_angle)
        if folder_joint_max_angle is not None:
            env_cfg.folder_joint_max_angle = float(folder_joint_max_angle)
        if folder_joint_initial_angle is not None:
            env_cfg.folder_joint_initial_angle = float(folder_joint_initial_angle)
        if folder_joint_drive_stiffness is not None:
            env_cfg.folder_joint_drive_stiffness = float(folder_joint_drive_stiffness)
        if folder_joint_drive_damping is not None:
            env_cfg.folder_joint_drive_damping = float(folder_joint_drive_damping)
    else:
        # Use CLI arguments
        env_cfg.folder.spawn.usd_path = _expand_env_vars(args_cli.folder_usd_path)
        env_cfg.folder.init_state.pos = tuple(args_cli.folder_pos)
        env_cfg.folder.spawn.scale = tuple(args_cli.folder_scale)
        env_cfg.disable_fixed_joints = args_cli.disable_fixed_joints

    print(f"[INFO] Using folder: {env_cfg.folder.spawn.usd_path}")
    print(f"[INFO] Folder position: {env_cfg.folder.init_state.pos}")
    print(f"[INFO] Folder scale: {env_cfg.folder.spawn.scale}")
    print(f"[INFO] Disable fixed joints: {env_cfg.disable_fixed_joints}")
    if hasattr(env_cfg, 'folder_mass_density') and env_cfg.folder_mass_density is not None:
        print(f"[INFO] Folder mass density: {env_cfg.folder_mass_density} kg/m³")
    if hasattr(env_cfg, 'folder_mass') and env_cfg.folder_mass is not None:
        print(f"[INFO] Folder mass: {env_cfg.folder_mass} kg")
    if hasattr(env_cfg, 'folder_joint_name'):
        print(f"[INFO] Folder joint name: {env_cfg.folder_joint_name}")
    if hasattr(env_cfg, 'folder_joint_min_angle') and hasattr(env_cfg, 'folder_joint_max_angle'):
        min_deg = math.degrees(env_cfg.folder_joint_min_angle)
        max_deg = math.degrees(env_cfg.folder_joint_max_angle)
        print(f"[INFO] Folder joint limits: [{env_cfg.folder_joint_min_angle:.4f}, {env_cfg.folder_joint_max_angle:.4f}] rad ({min_deg:.2f}°, {max_deg:.2f}°)")
    if hasattr(env_cfg, 'folder_joint_initial_angle') and env_cfg.folder_joint_initial_angle is not None:
        initial_deg = math.degrees(env_cfg.folder_joint_initial_angle)
        print(f"[INFO] Folder joint initial angle: {env_cfg.folder_joint_initial_angle:.4f} rad ({initial_deg:.2f}°)")
    if hasattr(env_cfg, 'folder_joint_drive_stiffness') and hasattr(env_cfg, 'folder_joint_drive_damping'):
        print(f"[INFO] Folder joint drive: stiffness={env_cfg.folder_joint_drive_stiffness}, damping={env_cfg.folder_joint_drive_damping}")

    # Apply finger poses
    fp = _get_cfg(CFG, ["finger_poses"], {}) or {}
    fp_initial = fp.get("initial", "closed") if isinstance(fp, dict) else "closed"
    closed = fp.get("closed", {}) if isinstance(fp, dict) else {}
    closed_base = (closed.get("joint_pos", {}) if isinstance(closed, dict) else {}) or {}
    closed_per = (closed.get("per_arm", {}) if isinstance(closed, dict) else {}) or {}
    left_closed = {**(closed_base if isinstance(closed_base, dict) else {}), **(closed_per.get("left", {}) if isinstance(closed_per, dict) else {})}
    right_closed = {**(closed_base if isinstance(closed_base, dict) else {}), **(closed_per.get("right", {}) if isinstance(closed_per, dict) else {})}
    if len(left_closed) == 0:
        left_closed = FIXED_GRASP_POSE
    if len(right_closed) == 0:
        right_closed = FIXED_GRASP_POSE

    if str(fp_initial).lower() == "open":
        env_cfg.manual_finger_joint_positions_left = None
        env_cfg.manual_finger_joint_positions_right = None
    else:
        env_cfg.manual_finger_joint_positions_left = left_closed
        env_cfg.manual_finger_joint_positions_right = right_closed
    
    # Create environment
    env = ShadowHandDualArmPickUpObjectScriptedEnv(cfg=env_cfg)

    # Disable automatic reset
    env._disable_auto_reset = True

    # Attach YAML to env
    env._cfg_yaml = CFG

    # Apply right-arm EE offset
    try:
        pos_t = torch.tensor(right_pos, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        quat_t = torch.tensor(right_quat, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        if hasattr(env, "_offset_pos_right"):
            env._offset_pos_right[:] = pos_t
        if hasattr(env, "_offset_rot_right"):
            env._offset_rot_right[:] = quat_t
    except Exception as e:
        print(f"[WARNING] Failed to apply right-arm EE offset from YAML: {e}")
    
    # Disable automatic finger control
    env._auto_finger_control_left = False
    env._auto_finger_control_right = False
    
    # Create custom window
    env._window = CustomEnvWindow(env, window_name="ShadowHand Dual-Arm Folder Manipulation")
    
    # Run simulator
    try:
        run_simulator(env)
    except Exception:
        carb.log_error(traceback.format_exc())
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()

