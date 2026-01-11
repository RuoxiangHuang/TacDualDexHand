"""Single-arm ShadowHand pick-up task environment with GelSight Hand sensors."""

from __future__ import annotations

import torch
import math
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.ur10e_shadowhand.ur10e_shadowhand_gelsighthand import (
    UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG,
    UR10E_SHADOWHAND_RIGHT_GELSIGHTHAND_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_hand.gelsighthand_cfg import GelSightHandCfg
from tacex_tasks.utils import DirectLiveVisualizer


def _create_sensor_cfg_for_shadowhand(finger_name: str, robot_prim_path: str) -> GelSightHandCfg:
    """Create GelSight Hand sensor configuration for ShadowHand finger.
    
    Args:
        finger_name: Finger name (ff, mf, rf, lf, th)
        robot_prim_path: Robot prim path (e.g., "/World/envs/env_.*/Robot")
    
    Returns:
        GelSightHandCfg configuration
    """
    finger_name_map = {
        "ff": ("ffdistal", "ffdistal"),
        "mf": ("mfdistal", "mfdistal"),
        "rf": ("rfdistal", "rfdistal"),
        "lf": ("lfdistal", "lfdistal"),
        "th": ("thdistal", "thdistal"),
    }
    case_suffix, gelpad_suffix = finger_name_map[finger_name]
    case_name = f"left_{case_suffix}_case"
    gelpad_name = f"left_{gelpad_suffix}_gelpad"

    sensor_cfg = GelSightHandCfg(
        prim_path=f"{robot_prim_path}/{case_name}",
        sensor_camera_cfg=GelSightHandCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),
            data_types=["depth"],
            # Same as GelSightMiniCfg in run_shape_touch.py
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],
    )
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
        # Same as GelSightMiniCfg in run_shape_touch.py
        gelpad_to_camera_min_distance=0.024,
    )
    return sensor_cfg


@configclass
class ShadowHandPickUpObjectEnvCfg(DirectRLEnvCfg):
    """Config for ShadowHand pick-up task with 5 GelSight Hand sensors."""

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.5, 1.0, 0.5)
    viewer.lookat = (0.2, 0.0, 0.1)

    debug_vis = True
    ui_window_class_type = BaseEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,  # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
        render=RenderCfg(enable_translucency=True),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/plate.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0, 0.035)),
        spawn=sim_utils.UsdFileCfg(
            scale=(5, 5, 5),
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    robot: ArticulationCfg = UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # Create 5 GelSight Hand sensors (one per finger)
    gelsighthand_ff = _create_sensor_cfg_for_shadowhand("ff", "/World/envs/env_.*/Robot")
    gelsighthand_mf = _create_sensor_cfg_for_shadowhand("mf", "/World/envs/env_.*/Robot")
    gelsighthand_rf = _create_sensor_cfg_for_shadowhand("rf", "/World/envs/env_.*/Robot")
    gelsighthand_lf = _create_sensor_cfg_for_shadowhand("lf", "/World/envs/env_.*/Robot")
    gelsighthand_th = _create_sensor_cfg_for_shadowhand("th", "/World/envs/env_.*/Robot")

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    episode_length_s = 6.0
    action_space = 6  # 6 DoF IK (dx, dy, dz, droll, dpitch, dyaw)
    observation_space = {
        "proprio_obs": 20,  # ee pos(3) + euler(3) + object pos(3) + joint_pos(11) + actions(6)
        "vision_obs": [32, 32, 15],  # 5 tactile RGB (5*3=15 channels)
    }
    state_space = 0

    action_scale_translation = 0.05
    action_scale_rotation = 0.35
    min_object_height = 0.003
    success_height = 0.15

    # Scripted trajectory parameters
    approach_height: float = 0.035  # m above object (e.g., 0.07 - 0.035 = 0.035 for object at z=0.035)
    grasp_height: float = -0.01  # m relative to object center
    lift_height: float = 0.18  # target additional height above object
    
    # End-effector offset from palm body to desired end-effector frame
    # This compensates for the difference between palm body frame and actual grasping point
    body_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) offset in meters
    body_offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z) quaternion
    # Dual-arm can optionally override offsets per arm. If these stay None, we fall back to body_offset_* above.
    body_offset_pos_left: tuple[float, float, float] | None = None
    body_offset_rot_left: tuple[float, float, float, float] | None = None
    body_offset_pos_right: tuple[float, float, float] | None = None
    body_offset_rot_right: tuple[float, float, float, float] | None = None
    
    # Manual finger joint positions (for custom grasping pose)
    # If None, uses automatic closed position (lower limits)
    # Format: dict mapping joint name pattern to target position (radians)
    # Example: {"FFJ1": 0.5, "FFJ2": 0.3, "MFJ1": 0.5, ...}
    manual_finger_joint_positions: dict[str, float] | None = None

    use_images: bool = True

    arm_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/ArmCamera",
        offset=OffsetCfg(
            pos=(1.04358, 0.43, 0.47413),
            rot=(0.3946, 0.19595, 0.42067, 0.79305),  # (w, x, y, z)
        ),
        update_period=0.0,
        height=32,
        width=32,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=0.8,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 2.0),
        ),
    )


class ShadowHandPickUpObjectEnv(DirectRLEnv):
    """RL env for picking up an object with ShadowHand and 5 GelSight Hand sensors."""

    cfg: ShadowHandPickUpObjectEnvCfg

    def __init__(self, cfg: ShadowHandPickUpObjectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        
        # Find palm or end-effector body
        # For ShadowHand, we need to find the correct end-effector body
        # The structure is: UR10e arm -> base/mount -> palm/hand
        # We'll try multiple strategies to find the correct body
        self._body_idx = None
        self._body_name = None
        
        # Strategy 1: Try to find palm body (left_palm for left hand, right_palm for right hand)
        try:
            body_ids, body_names = self._robot.find_bodies(".*palm.*")
            if len(body_ids) > 0:
                self._body_idx = body_ids[0]
                self._body_name = body_names[0]
                print(f"[INFO] Found palm body: {self._body_name} (index {self._body_idx})")
        except:
            pass
        
        # Strategy 2: Try to find hand body
        if self._body_idx is None:
            try:
                body_ids, body_names = self._robot.find_bodies(".*hand.*")
                if len(body_ids) > 0:
                    self._body_idx = body_ids[0]
                    self._body_name = body_names[0]
                    print(f"[INFO] Found hand body: {self._body_name} (index {self._body_idx})")
            except:
                pass
        
        # Strategy 3: Try to find base/mount body (connection between arm and hand)
        if self._body_idx is None:
            try:
                body_ids, body_names = self._robot.find_bodies(".*(base|mount|adapter|link).*")
                # Prefer bodies that are not the root
                for body_id, body_name in zip(body_ids, body_names):
                    if body_id != 0:  # Not root
                        self._body_idx = body_id
                        self._body_name = body_name
                        print(f"[INFO] Found base/mount body: {self._body_name} (index {self._body_idx})")
                        break
            except:
                pass
        
        # Strategy 4: Fallback to root link (body index 0)
        if self._body_idx is None:
            self._body_idx = 0
            self._body_name = self._robot.body_names[0] if len(self._robot.body_names) > 0 else "root"
            print(f"[WARNING] Could not find palm/hand body. Using root link: {self._body_name} (index {self._body_idx})")
        
        # For fixed base robots, the Jacobian body index is one less than the body index
        # because the root body (index 0) is not included in the returned Jacobians
        # ShadowHand is a fixed base robot (base_link is fixed), so we use body_idx - 1
        # Example: if body_idx = 12 (palm), then jacobian index = 11
        self._jacobi_body_idx = self._body_idx - 1 if self._body_idx > 0 else 0
        
        # Offset from body frame to desired end-effector frame
        # This compensates for the difference between the body frame and the actual end-effector position
        # For ShadowHand, if we're using a base/mount body, we may need an offset to reach the palm center
        # Can be configured via cfg.body_offset_pos and cfg.body_offset_rot
        offset_pos = self.cfg.body_offset_pos
        offset_rot = self.cfg.body_offset_rot
        self._offset_pos = torch.tensor(offset_pos, device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor(offset_rot, device=self.device).repeat(self.num_envs, 1)
        
        print(f"[INFO] Using end-effector body: {self._body_name} (index {self._body_idx}, jacobian index {self._jacobi_body_idx})")
        print(f"[INFO] End-effector offset: pos={self._offset_pos[0].tolist()}, rot={self._offset_rot[0].tolist()}")

        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Find finger joints (ShadowHand has multiple fingers, each with multiple joints)
        # ShadowHand joint structure:
        # - UR10e arm: 6 joints (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
        # - ShadowHand fingers: typically 22 joints
        #   * Index finger (FF): FFJ1, FFJ2, FFJ3, FFJ4 (4 joints)
        #   * Middle finger (MF): MFJ1, MFJ2, MFJ3, MFJ4 (4 joints)
        #   * Ring finger (RF): RFJ1, RFJ2, RFJ3, RFJ4 (4 joints)
        #   * Little finger (LF): LFJ1, LFJ2, LFJ3, LFJ4, LFJ5 (5 joints)
        #   * Thumb (TH): THJ1, THJ2, THJ3, THJ4, THJ5 (5 joints)
        #   * Wrist (WR): WRJ1, WRJ2 (2 joints)
        # Note: Joint names are UPPERCASE in the USD file
        try:
            all_joint_names = self._robot.joint_names
            # Try to get joint types if available, otherwise use placeholder
            try:
                all_joint_types = self._robot.joint_types
            except AttributeError:
                all_joint_types = ["unknown"] * len(all_joint_names)
            
            # Find arm joints (UR10e: 6 DOF)
            arm_joint_ids = []
            arm_joint_names = []
            for i, joint_name in enumerate(all_joint_names):
                name_lower = joint_name.lower()
                # Match UR10e arm joints (lowercase: shoulder, elbow, wrist)
                if any(term in name_lower for term in ["shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3", "ur10", "base"]):
                    arm_joint_ids.append(i)
                    arm_joint_names.append(joint_name)
            
            # Find finger joints (ShadowHand: typically 22 DOF)
            finger_joint_ids = []
            finger_joint_names = []
            for i, joint_name in enumerate(all_joint_names):
                # Check if it's a finger joint (uppercase: FFJ, MFJ, RFJ, LFJ, THJ, WRJ)
                # Also check lowercase for compatibility
                name_upper = joint_name.upper()
                name_lower = joint_name.lower()
                if any(finger in name_upper for finger in ["FFJ", "MFJ", "RFJ", "LFJ", "THJ", "WRJ"]) or \
                   any(finger in name_lower for finger in ["ffj", "mfj", "rfj", "lfj", "thj", "wrj", "finger", "thumb"]):
                    finger_joint_ids.append(i)
                    finger_joint_names.append(joint_name)
            
            # Fallback: if no finger joints found by name, assume all joints after arm are fingers
            if len(finger_joint_ids) == 0 and len(arm_joint_ids) > 0:
                num_arm_joints = len(arm_joint_ids)
                finger_joint_ids = list(range(num_arm_joints, len(all_joint_names)))
                finger_joint_names = [all_joint_names[i] for i in finger_joint_ids]
                print(f"[WARNING] Could not identify finger joints by name. Assuming joints {num_arm_joints} to {len(all_joint_names)-1} are fingers.")
            
            self._arm_joint_ids = torch.tensor(arm_joint_ids, device=self.device) if arm_joint_ids else None
            self._arm_joint_names = arm_joint_names
            self._finger_joint_ids = torch.tensor(finger_joint_ids, device=self.device) if finger_joint_ids else torch.tensor([], device=self.device, dtype=torch.long)
            self._finger_joint_names = finger_joint_names
            
            print(f"[INFO] Robot joint structure:")
            print(f"  Total joints: {len(all_joint_names)}")
            print(f"  Arm joints ({len(arm_joint_ids)}): {arm_joint_names}")
            print(f"  Finger joints ({len(finger_joint_ids)}): {finger_joint_names[:5]}{'...' if len(finger_joint_names) > 5 else ''}")
            
        except Exception as e:
            print(f"[WARNING] Could not identify joints: {e}. Using fallback method.")
            # Fallback: assume first 6 are arm, rest are fingers
            all_joint_names = self._robot.joint_names
            self._arm_joint_ids = torch.tensor(list(range(min(6, len(all_joint_names)))), device=self.device)
            self._arm_joint_names = all_joint_names[:6] if len(all_joint_names) >= 6 else all_joint_names
            self._finger_joint_ids = torch.tensor(list(range(6, len(all_joint_names))), device=self.device) if len(all_joint_names) > 6 else torch.tensor([], device=self.device, dtype=torch.long)
            self._finger_joint_names = all_joint_names[6:] if len(all_joint_names) > 6 else []
        
        # Finger target positions (for scripted grasping)
        num_finger_joints = len(self._finger_joint_ids) if len(self._finger_joint_ids) > 0 else 0
        self.finger_targets = torch.zeros((self.num_envs, num_finger_joints), device=self.device)
        
        # Initialize finger targets
        if num_finger_joints > 0:
            if self.cfg.manual_finger_joint_positions is not None:
                # Use manual finger joint positions if specified
                self.finger_targets = self._get_manual_finger_targets()
                print(f"[INFO] Using manual finger joint positions: {self.cfg.manual_finger_joint_positions}")
            else:
                # Use automatic open position (lower limits) as default
                finger_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._finger_joint_ids, 0]
                self.finger_targets[:] = finger_lower_limits.unsqueeze(0)  # Set to open position by default
                print(f"[INFO] Using automatic finger joint positions (lower limits - open state)")

        # Initialize grasp target positions (will be set in reset)
        self._initial_grasp_targets_b = torch.zeros((self.num_envs, 3), device=self.device)
    
    def _get_manual_finger_targets(self) -> torch.Tensor:
        """Get manual finger joint targets based on cfg.manual_finger_joint_positions.
        
        Returns:
            Tensor of shape (num_envs, num_finger_joints) with target positions
        """
        if self.cfg.manual_finger_joint_positions is None:
            # Fallback to lower limits
            finger_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._finger_joint_ids, 0]
            return finger_lower_limits.unsqueeze(0).repeat(self.num_envs, 1)
        
        # Get joint names
        joint_names = self._robot.joint_names
        finger_targets = torch.zeros((self.num_envs, len(self._finger_joint_ids)), device=self.device)
        
        # Match joint names to manual positions
        for i, joint_idx in enumerate(self._finger_joint_ids):
            joint_name = joint_names[joint_idx]
            # Try exact match first
            if joint_name in self.cfg.manual_finger_joint_positions:
                finger_targets[:, i] = self.cfg.manual_finger_joint_positions[joint_name]
            else:
                # Try pattern matching (e.g., "FFJ1" matches joint name containing "FFJ1")
                matched = False
                for pattern, value in self.cfg.manual_finger_joint_positions.items():
                    if pattern.upper() in joint_name.upper() or joint_name.upper() in pattern.upper():
                        finger_targets[:, i] = value
                        matched = True
                        break
                if not matched:
                    # Fallback to lower limit
                    finger_targets[:, i] = self._robot.data.soft_joint_pos_limits[0, joint_idx, 0]
        
        return finger_targets

        # Debug visualizers
        if self.cfg.debug_vis:
            self.visualizers = {
                "Observations": DirectLiveVisualizer(
                    self.cfg.debug_vis, self.num_envs, self._window, visualizer_name="Observations"
                ),
            }
            self.visualizers["Observations"].terms["sensor_output"] = torch.zeros(
                (
                    self.num_envs,
                    self.cfg.observation_space["vision_obs"][0],
                    self.cfg.observation_space["vision_obs"][1],
                    self.cfg.observation_space["vision_obs"][2],
                )
            )
            for vis in self.visualizers.values():
                vis.create_visualizer()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        RigidObject(self.cfg.plate)

        self.scene.clone_environments(copy_from_source=False)

        # Create sensors
        self.gelsighthand_ff = GelSightSensor(self.cfg.gelsighthand_ff)
        self.gelsighthand_mf = GelSightSensor(self.cfg.gelsighthand_mf)
        self.gelsighthand_rf = GelSightSensor(self.cfg.gelsighthand_rf)
        self.gelsighthand_lf = GelSightSensor(self.cfg.gelsighthand_lf)
        self.gelsighthand_th = GelSightSensor(self.cfg.gelsighthand_th)
        
        self.scene.sensors["gelsighthand_ff"] = self.gelsighthand_ff
        self.scene.sensors["gelsighthand_mf"] = self.gelsighthand_mf
        self.scene.sensors["gelsighthand_rf"] = self.gelsighthand_rf
        self.scene.sensors["gelsighthand_lf"] = self.gelsighthand_lf
        self.scene.sensors["gelsighthand_th"] = self.gelsighthand_th

        # Global RGB camera
        if not hasattr(self.cfg.arm_camera.offset, "convention"):
            self.cfg.arm_camera.offset.convention = "usd"
        self.arm_camera = Camera(self.cfg.arm_camera)
        self.scene.sensors["arm_camera"] = self.arm_camera

        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

        # Rigid object
        self.object = RigidObject(self.cfg.object)
        self.scene.rigid_objects["object"] = self.object

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions for IK control."""
        actions = actions.clamp(-1.0, 1.0)
        self.actions[:] = actions
        ee_pos_b, ee_quat_b = self._compute_frame_pose()

        self.processed_actions.zero_()
        trans_scale = self.cfg.action_scale_translation
        rot_scale = self.cfg.action_scale_rotation
        self.processed_actions[:, :3] = actions[:, :3] * trans_scale
        self.processed_actions[:, 3:6] = actions[:, 3:6] * rot_scale
        self._ik_controller.set_command(self.processed_actions, ee_pos_b, ee_quat_b)

    def _apply_action(self):
        """Apply IK control to robot and finger joints."""
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        # Apply finger joint targets if available
        if len(self._finger_joint_ids) > 0:
            joint_pos_des[:, self._finger_joint_ids] = self.finger_targets

        self._robot.set_joint_position_target(joint_pos_des)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episode is done."""
        # For GUI control mode, disable automatic reset to prevent overwriting IK state
        # Only reset when manually triggered via GUI
        if hasattr(self, '_disable_auto_reset') and self._disable_auto_reset:
            # Return no reset conditions (all False)
            reset_cond = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return reset_cond, time_out
        
        object_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            self.object.data.root_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1),
        )
        min_height = object_pos_b[:, 2] < self.cfg.min_object_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reset_cond = min_height | time_out
        return reset_cond, time_out

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        ee_pos_b, _ = self._compute_frame_pose()
        object_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            self.object.data.root_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1),
        )

        height_reward = torch.clamp(object_pos_b[:, 2] - 0.02, min=0.0)
        dist = torch.norm(ee_pos_b - object_pos_b, dim=1)
        dist_reward = -dist
        success = (object_pos_b[:, 2] > self.cfg.success_height) & (dist < 0.05)
        success_bonus = success.float() * 5.0

        return 2.0 * height_reward + dist_reward + success_bonus

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environment."""
        super()._reset_idx(env_ids)
        
        # Reset rigid object first
        object_state = self.object.data.default_root_state.clone()[env_ids]
        object_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, 0:7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)
        self.object.reset(env_ids)
        
        # Update scene to get latest object position
        self.scene.update(dt=0.0)

        # Reset to default joint positions (original default)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset finger targets (use manual if specified, otherwise use upper limits - open state)
        if len(self._finger_joint_ids) > 0:
            if self.cfg.manual_finger_joint_positions is not None:
                manual_targets = self._get_manual_finger_targets()
                self.finger_targets[env_ids] = manual_targets[env_ids]
            else:
                finger_lower_limits = self._robot.data.soft_joint_pos_limits[:, self._finger_joint_ids, 0]
                self.finger_targets[env_ids] = finger_lower_limits[env_ids]

        self.processed_actions.zero_()
        self.actions[env_ids] = 0.0
        
        # Reset episode length
        self.episode_length_buf[env_ids] = 0

    def _get_observations(self) -> dict:
        """Get observations."""
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        ee_euler = euler_xyz_from_quat(ee_quat_b)
        ex = wrap_to_pi(ee_euler[0]).unsqueeze(1)
        ey = wrap_to_pi(ee_euler[1]).unsqueeze(1)
        ez = wrap_to_pi(ee_euler[2]).unsqueeze(1)

        object_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            self.object.data.root_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1),
        )

        joint_pos = self._robot.data.joint_pos[:, :]
        # Use first 11 joints for observation (adjust based on your robot)
        num_joints_obs = min(11, joint_pos.shape[1])
        joint_pos_obs = joint_pos[:, :num_joints_obs]

        proprio_obs = torch.cat(
            (ee_pos_b, ex, ey, ez, object_pos_b, joint_pos_obs, self.processed_actions),
            dim=-1,
        )

        # Stack 5 tactile RGB images: (num_envs, H, W, 15) = 5 * 3 channels
        imgs = []  # Initialize imgs list
        if self.cfg.use_images:
            for sensor in (self.gelsighthand_ff, self.gelsighthand_mf, self.gelsighthand_rf, 
                          self.gelsighthand_lf, self.gelsighthand_th):
                tactile = sensor.data.output.get("tactile_rgb", None)
                if tactile is not None:
                    tensor = torch.as_tensor(tactile, device=self.device, dtype=torch.float32)
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                    imgs.append(tensor)

            if len(imgs) == 0:
                vision_obs = torch.zeros((self.num_envs, 32, 32, 15), device=self.device)
            else:
                # Ensure 5 images
                while len(imgs) < 5:
                    imgs.append(torch.zeros_like(imgs[0]) if len(imgs) > 0 else torch.zeros((self.num_envs, 32, 32, 3), device=self.device))
                vision_obs = torch.cat(imgs, dim=-1)
        else:
            vision_obs = torch.zeros((self.num_envs, 32, 32, 15), device=self.device)

        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}

        if self.cfg.debug_vis and hasattr(self, "visualizers") and "Observations" in self.visualizers:
            for i, sensor_name in enumerate(["ff", "mf", "rf", "lf", "th"]):
                if i < len(imgs):
                    self.visualizers["Observations"].terms[f"sensor_output_tactile_{sensor_name}"] = imgs[i]

        return {"policy": obs}

    @property
    def jacobian_w(self) -> torch.Tensor:
        """Get world frame jacobian."""
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        """Get base frame jacobian."""
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute end-effector pose in base frame."""
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot)
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Compute jacobian for end-effector frame."""
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


@configclass
class ShadowHandPickUpObjectScriptedEnvCfg(ShadowHandPickUpObjectEnvCfg):
    """Variant where end-effector trajectory is scripted (hardcoded)."""

    action_space = 0  # No actions needed, fully scripted


class ShadowHandPickUpObjectScriptedEnv(ShadowHandPickUpObjectEnv):
    """Env: scripted end-effector motion for pick-up task."""

    cfg: ShadowHandPickUpObjectScriptedEnvCfg

    def __init__(self, cfg: ShadowHandPickUpObjectScriptedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Flag to enable/disable automatic finger control
        # When False, allows manual control via GUI or keyboard
        self._auto_finger_control = cfg.manual_finger_joint_positions is not None

    def _pre_physics_step(self, actions: torch.Tensor):
        """Control end-effector: use provided actions if non-zero, otherwise use scripted trajectory.
        
        If actions are provided (non-zero), use them for IK control.
        Otherwise, use scripted trajectory (lift upward).
        If ik_commands is available (e.g., from virtual object control), use it.
        """
        # Current EE pose (base frame)
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        
        # Check if ik_commands is available (for virtual object control, similar to run_shape_touch.py)
        if hasattr(self, 'ik_commands') and self.ik_commands is not None and torch.any(self.ik_commands != 0):
            # Use ik_commands directly (exactly as in run_shape_touch.py)
            # For use_relative_mode=False, ik_commands contains absolute poses in base frame
            self._ik_controller.set_command(self.ik_commands)
        # Check if processed_actions is set (alternative method)
        elif hasattr(self, 'processed_actions') and torch.any(self.processed_actions != 0):
            # Use processed_actions directly
            self._ik_controller.set_command(self.processed_actions, ee_pos_b, ee_quat_b)
        # Check if actions are provided (non-zero)
        elif actions is not None and torch.any(actions != 0):
            # Use provided actions for IK control
            self.processed_actions.zero_()
            trans_scale = self.cfg.action_scale_translation
            rot_scale = self.cfg.action_scale_rotation
            self.processed_actions[:, :3] = actions[:, :3] * trans_scale
            self.processed_actions[:, 3:6] = actions[:, 3:6] * rot_scale
            self._ik_controller.set_command(self.processed_actions, ee_pos_b, ee_quat_b)
        else:
            # Use scripted trajectory: Lift end-effector upward while maintaining grasp
            # Object position in world frame
            object_pos_w = self.object.data.root_pos_w
            
            # Normalized time [0, 1]
            frac = (self.episode_length_buf.to(torch.float32) / max(self.max_episode_length - 1, 1)).unsqueeze(1)

            # Target end-effector position in world frame: x,y aligned with object, z interpolates from grasp to lift
            z_grasp_w = object_pos_w[:, 2:3] + self.cfg.grasp_height
            z_lift_w = object_pos_w[:, 2:3] + self.cfg.lift_height
            z_target_w = z_grasp_w * (1.0 - frac) + z_lift_w * frac
            
            # Target position in world frame
            target_pos_w = torch.cat([object_pos_w[:, 0:2], z_target_w], dim=1)
            
            # Convert target position from world frame to base frame
            root_pos_w = self._robot.data.root_link_pos_w
            root_quat_w = self._robot.data.root_link_quat_w
            target_pos_b, _ = math_utils.subtract_frame_transforms(
                root_pos_w,
                root_quat_w,
                target_pos_w,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1),
            )

            # IK command: gentle upward movement
            delta_pos = target_pos_b - ee_pos_b
            trans_scale = max(self.cfg.action_scale_translation, 1e-6)
            self.processed_actions.zero_()
            self.processed_actions[:, :3] = torch.clamp(0.1 * delta_pos / trans_scale, -1.0, 1.0)
            self.processed_actions[:, 3:6] = 0.0  # No rotation change
            self._ik_controller.set_command(self.processed_actions, ee_pos_b, ee_quat_b)

        # Control finger joints: use manual positions if specified and auto control is enabled
        # NOTE: If _auto_finger_control is False, allow manual control (e.g., via GUI or keyboard)
        if len(self._finger_joint_ids) > 0:
            if self._auto_finger_control and self.cfg.manual_finger_joint_positions is not None:
                # Use manual finger positions from config
                finger_targets = self._get_manual_finger_targets()
                # Smooth transition using exponential moving average
                alpha = 0.1  # Smoothing factor
                self.finger_targets = alpha * finger_targets + (1.0 - alpha) * self.finger_targets
            # else: Don't modify finger_targets - allow manual control (e.g., GUI or keyboard)


##
# Dual-Arm Configuration and Environment
##


def _create_sensor_cfg_for_shadowhand_dual_arm(finger_name: str, robot_prim_path: str, arm_name: str) -> GelSightHandCfg:
    """Create GelSight Hand sensor configuration for ShadowHand finger (dual-arm version).
    
    Args:
        finger_name: Finger name (ff, mf, rf, lf, th)
        robot_prim_path: Robot prim path (e.g., "/World/envs/env_.*/Robot_Left")
        arm_name: Arm name ("left" or "right")
    
    Returns:
        GelSightHandCfg configuration
    """
    finger_name_map = {
        "ff": ("ffdistal", "ffdistal"),
        "mf": ("mfdistal", "mfdistal"),
        "rf": ("rfdistal", "rfdistal"),
        "lf": ("lfdistal", "lfdistal"),
        "th": ("thdistal", "thdistal"),
    }
    case_suffix, gelpad_suffix = finger_name_map[finger_name]
    case_name = f"{arm_name}_{case_suffix}_case"
    gelpad_name = f"{arm_name}_{gelpad_suffix}_gelpad"

    sensor_cfg = GelSightHandCfg(
        prim_path=f"{robot_prim_path}/{case_name}",
        sensor_camera_cfg=GelSightHandCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],
    )
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
        gelpad_to_camera_min_distance=0.024,
    )
    return sensor_cfg


@configclass
class ShadowHandDualArmPickUpObjectScriptedEnvCfg(ShadowHandPickUpObjectEnvCfg):
    """Config for dual-arm ShadowHand pick-up task with 10 GelSight Hand sensors (5 per arm)."""

    # Left arm robot
    robot_left: ArticulationCfg = UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_Left",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  # Relative to env_origins
            rot=(0.0, 0.0, 0.0, 1.0),
            # Joint positions will be set in _reset_idx to avoid straight arm pose
        ),
    )

    # Right arm robot
    robot_right: ArticulationCfg = UR10E_SHADOWHAND_RIGHT_GELSIGHTHAND_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_Right",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(3.0, 0.0, 0.0),  # Relative to env_origins, offset 3.0m from left arm (increased distance)
            rot=(0.0, 0.0, 0.0, 1.0),
            # Joint positions will be set in _reset_idx to avoid straight arm pose
        ),
    )

    # Left arm sensors
    gelsighthand_ff_left = _create_sensor_cfg_for_shadowhand_dual_arm("ff", "/World/envs/env_.*/Robot_Left", "left")
    gelsighthand_mf_left = _create_sensor_cfg_for_shadowhand_dual_arm("mf", "/World/envs/env_.*/Robot_Left", "left")
    gelsighthand_rf_left = _create_sensor_cfg_for_shadowhand_dual_arm("rf", "/World/envs/env_.*/Robot_Left", "left")
    gelsighthand_lf_left = _create_sensor_cfg_for_shadowhand_dual_arm("lf", "/World/envs/env_.*/Robot_Left", "left")
    gelsighthand_th_left = _create_sensor_cfg_for_shadowhand_dual_arm("th", "/World/envs/env_.*/Robot_Left", "left")

    # Right arm sensors
    gelsighthand_ff_right = _create_sensor_cfg_for_shadowhand_dual_arm("ff", "/World/envs/env_.*/Robot_Right", "right")
    gelsighthand_mf_right = _create_sensor_cfg_for_shadowhand_dual_arm("mf", "/World/envs/env_.*/Robot_Right", "right")
    gelsighthand_rf_right = _create_sensor_cfg_for_shadowhand_dual_arm("rf", "/World/envs/env_.*/Robot_Right", "right")
    gelsighthand_lf_right = _create_sensor_cfg_for_shadowhand_dual_arm("lf", "/World/envs/env_.*/Robot_Right", "right")
    gelsighthand_th_right = _create_sensor_cfg_for_shadowhand_dual_arm("th", "/World/envs/env_.*/Robot_Right", "right")

    # IK controllers for each arm
    ik_controller_cfg_left = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller_cfg_right = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    # Manual finger joint positions for each arm (optional)
    manual_finger_joint_positions_left: dict[str, float] | None = None
    manual_finger_joint_positions_right: dict[str, float] | None = None
    
    folder = ArticulationCfg(
        prim_path="/World/envs/env_.*/folder",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.1, -0.1, 0.01),
            rot=(1.0, 0.0, 0.0, 0.0),
            # joint_pos 留空，让Articulation自动处理所有joint
            # 注意：如果设置joint_pos={}，Articulation可能会尝试匹配所有joint，导致错误
            # 不设置joint_pos属性，让Articulation使用默认值
            # joint_pos={
            #     "RevoluteJoint_file_7_up": 0.5
            # }
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/media/isaac/e2fa93f5-512e-4f9a-9023-40cf92b0ad74/TacEx/source/tacex_tasks/tacex_tasks/bottle_cap/stationery/folder/file_7/model_file_7.usd",
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,  # 启用重力（False表示启用重力）
                kinematic_enabled=False,  # 重要：设置为False表示文件夹是可移动的（dynamic），不是静态的（kinematic）
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            scale=(1.0, 1.0, 1.0),  # 可以根据需要调整缩放
        ),
        actuators={},  # Empty actuators dict for passive object
    )
    
    # ============================================================================
    # Folder Fixed Joint Configuration
    # ============================================================================
    # 是否禁用FixedJoint（防止文件夹被固定）
    # True: 自动禁用所有找到的FixedJoint
    # False: 保留FixedJoint（文件夹可能被固定）
    disable_fixed_joints: bool = True  # 默认禁用FixedJoint，允许文件夹移动
    
    # ============================================================================
    # Folder Mass/Density Configuration
    # ============================================================================
    # 文件夹质量/密度设置
    # 减小质量可以减少重力对文件夹的影响，有助于防止掉落
    # 可以设置密度（kg/m³）或质量（kg），如果都设置则使用质量
    # 典型值：水=1000, 塑料≈900-1000, 金属≈2700-7800
    # 如果设置为None，则使用USD文件中的原始质量/密度
    folder_mass_density: float | None = None  # [kg/m³] 密度（推荐使用，会根据体积自动计算质量）
    folder_mass: float | None = None  # [kg] 直接设置质量（如果设置，会覆盖density）
    
    # ============================================================================
    # Folder Joint Configuration
    # ============================================================================
    # 旋转关节名称（从终端输出可以看到是 RevoluteJoint_file_7_up）
    folder_joint_name: str = "RevoluteJoint_file_7_up"  # 旋转关节名称
    
    # 是否在USD文件中设置关节的物理限制（lowerLimit/upperLimit）
    # True: 根据下面的参数设置关节限制，会覆盖USD文件中的原始限制
    # False: 使用USD文件中的原始限制
    set_folder_joint_limits_in_usd: bool = True  # 是否设置关节限制
    
    # 旋转关节参数 (RevoluteJoint_file_7_up)
    # 注意：在YAML配置文件中使用角度值（度），代码会自动转换为弧度
    # 从终端输出可以看到：下限: 0.0 rad (0.00 deg), 上限: 80.0 rad (4583.66 deg)
    folder_joint_min_angle: float = 0.0  # [rad] (旋转关节下限，内部使用弧度，YAML中用度)
    folder_joint_max_angle: float = math.pi / 2  # [rad] (旋转关节上限，内部使用弧度，YAML中用度，默认90度)
    
    # 文件夹关节初始角度（重置时的初始状态）
    # 注意：在YAML配置文件中使用角度值（度），代码会自动转换为弧度
    # 如果为None，则使用USD文件中的默认值
    folder_joint_initial_angle: float | None = None  # [rad] (初始角度，内部使用弧度，YAML中用度)
    
    # 关节驱动属性（用于平衡GUI拖动能力和重力抵抗能力）
    # 值越大，越能抵抗重力，但GUI拖动可能越困难
    # 建议范围：stiffness: 1000-10000, damping: 100-1000
    # 注意：如果设置为0，将完全移除恢复力，允许自由拖动，但可能无法抵抗重力
    folder_joint_drive_stiffness: float = 1000  # 驱动刚度（用于抵抗重力，0=无恢复力）
    folder_joint_drive_damping: float = 100  # 驱动阻尼（用于稳定运动，0=无恢复力）


class ShadowHandDualArmPickUpObjectScriptedEnv(ShadowHandPickUpObjectScriptedEnv):
    """Dual-arm ShadowHand environment with independent control for each arm."""

    cfg: ShadowHandDualArmPickUpObjectScriptedEnvCfg

    def __init__(self, cfg: ShadowHandDualArmPickUpObjectScriptedEnvCfg, render_mode: str | None = None, **kwargs):
        # Temporarily set robot to left for initialization (will be overridden)
        original_robot = cfg.robot
        cfg.robot = cfg.robot_left
        # Store right arm config for later use (don't set to None, just don't let parent process it)
        self._robot_right_cfg = cfg.robot_right
        super().__init__(cfg, render_mode, **kwargs)
        cfg.robot = original_robot  # Restore original
        
        # Store left robot reference
        self._robot_left = self._robot
        self._ik_controller_left = self._ik_controller
        self._body_idx_left = self._body_idx
        self._body_name_left = self._body_name
        self._jacobi_body_idx_left = self._jacobi_body_idx
        # Left arm offset (allow per-arm override; fallback to shared body_offset_*)
        if cfg.body_offset_pos_left is not None:
            self._offset_pos_left = torch.tensor(cfg.body_offset_pos_left, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos_left = self._offset_pos
        if cfg.body_offset_rot_left is not None:
            self._offset_rot_left = torch.tensor(cfg.body_offset_rot_left, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_rot_left = self._offset_rot
        self._arm_joint_ids_left = self._arm_joint_ids
        self._arm_joint_names_left = self._arm_joint_names
        self._finger_joint_ids_left = self._finger_joint_ids
        self._finger_joint_names_left = self._finger_joint_names
        self.finger_targets_left = self.finger_targets
        self.processed_actions_left = self.processed_actions
        self.actions_left = self.actions
        
        # Disable automatic finger control by default
        self._auto_finger_control_left = cfg.manual_finger_joint_positions_left is not None
        self._auto_finger_control_right = cfg.manual_finger_joint_positions_right is not None
        
        # Initialize right arm (after scene setup)
        self._setup_right_arm()
        
        # 在sim.reset()完成后，禁用之前找到的FixedJoint
        # 注意：必须在super().__init__()完成后，因为此时sim.reset()已经执行
        if hasattr(self, '_fixed_joints_to_disable') and self._fixed_joints_to_disable:
            if self.cfg.disable_fixed_joints:
                self._disable_fixed_joints()
            else:
                print(f"[INFO] FixedJoint禁用功能已关闭，保留所有FixedJoint")
        
        # 在sim.reset()之后，重新设置folder的位置和状态
        # sim.reset()会将所有物体重置到USD文件的初始状态，可能导致folder位置变为(0,0,0)
        # 因此需要在这里重新设置folder的位置，确保使用配置中的位置
        if hasattr(self, 'folder'):
            # 直接使用配置的位置，而不是default_root_state（因为default_root_state可能已被sim.reset()覆盖）
            folder_pos = torch.tensor(self.cfg.folder.init_state.pos, device=self.device)
            folder_rot = torch.tensor(self.cfg.folder.init_state.rot, device=self.device)
            
            # 构建root_state tensor (13维: pos(3) + rot(4) + lin_vel(3) + ang_vel(3))
            root_state = torch.zeros((self.num_envs, 13), device=self.device)
            root_state[:, :3] = folder_pos.unsqueeze(0) + self.scene.env_origins
            root_state[:, 3:7] = folder_rot.unsqueeze(0)
            root_state[:, 7:] = 0.0  # 速度设为0
            
            # 写入仿真，确保folder在正确位置
            self.folder.write_root_state_to_sim(root_state)
            print(f"[DEBUG] 在sim.reset()后重新设置folder位置为 {self.cfg.folder.init_state.pos}")
            
            # 设置文件夹关节的驱动目标位置（如果配置了初始角度）
            if hasattr(self.cfg, 'folder_joint_initial_angle') and self.cfg.folder_joint_initial_angle is not None:
                joint_names = self.folder.joint_names
                if self.cfg.folder_joint_name in joint_names:
                    joint_idx = joint_names.index(self.cfg.folder_joint_name)
                    # 获取当前关节位置
                    current_joint_pos = self.folder.data.joint_pos.clone()
                    # 设置初始角度
                    current_joint_pos[:, joint_idx] = self.cfg.folder_joint_initial_angle
                    # 设置驱动目标位置，确保驱动保持关节在目标位置
                    self.folder.set_joint_position_target(current_joint_pos)
                    print(f"[DEBUG] 在初始化完成后设置文件夹关节 '{self.cfg.folder_joint_name}' 驱动目标位置为 {self.cfg.folder_joint_initial_angle:.4f} rad ({self.cfg.folder_joint_initial_angle * 180 / math.pi:.2f}°)")

    def _setup_right_arm(self):
        """Setup right arm IK controller and finger joints."""
        # Right arm robot is already created in _setup_scene
        
        # Create IK controller for right arm
        self._ik_controller_right = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg_right, num_envs=self.num_envs, device=self.device
        )
        
        # Find end-effector body for right arm (same logic as left arm)
        self._body_idx_right = None
        self._body_name_right = None
        
        try:
            body_ids, body_names = self._robot_right.find_bodies(".*palm.*")
            if len(body_ids) > 0:
                self._body_idx_right = body_ids[0]
                self._body_name_right = body_names[0]
        except:
            pass
        
        if self._body_idx_right is None:
            try:
                body_ids, body_names = self._robot_right.find_bodies(".*hand.*")
                if len(body_ids) > 0:
                    self._body_idx_right = body_ids[0]
                    self._body_name_right = body_names[0]
            except:
                pass
        
        if self._body_idx_right is None:
            try:
                body_ids, body_names = self._robot_right.find_bodies(".*(base|mount|adapter|link).*")
                for body_id, body_name in zip(body_ids, body_names):
                    if body_id != 0:
                        self._body_idx_right = body_id
                        self._body_name_right = body_name
                        break
            except:
                pass
        
        if self._body_idx_right is None:
            self._body_idx_right = 0
            self._body_name_right = self._robot_right.body_names[0] if len(self._robot_right.body_names) > 0 else "root"
        
        self._jacobi_body_idx_right = self._body_idx_right - 1 if self._body_idx_right > 0 else 0
        
        # Offset for right arm (allow per-arm override; fallback to shared body_offset_*)
        offset_pos = self.cfg.body_offset_pos_right if self.cfg.body_offset_pos_right is not None else self.cfg.body_offset_pos
        offset_rot = self.cfg.body_offset_rot_right if self.cfg.body_offset_rot_right is not None else self.cfg.body_offset_rot
        self._offset_pos_right = torch.tensor(offset_pos, device=self.device).repeat(self.num_envs, 1)
        self._offset_rot_right = torch.tensor(offset_rot, device=self.device).repeat(self.num_envs, 1)
        
        # Find finger joints for right arm
        try:
            all_joint_names = self._robot_right.joint_names
            arm_joint_ids = []
            arm_joint_names = []
            finger_joint_ids = []
            finger_joint_names = []
            
            for i, joint_name in enumerate(all_joint_names):
                name_lower = joint_name.lower()
                if any(term in name_lower for term in ["shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3", "ur10", "base"]):
                    arm_joint_ids.append(i)
                    arm_joint_names.append(joint_name)
            
            for i, joint_name in enumerate(all_joint_names):
                name_upper = joint_name.upper()
                name_lower = joint_name.lower()
                if any(finger in name_upper for finger in ["FFJ", "MFJ", "RFJ", "LFJ", "THJ", "WRJ"]) or \
                   any(finger in name_lower for finger in ["ffj", "mfj", "rfj", "lfj", "thj", "wrj", "finger", "thumb"]):
                    finger_joint_ids.append(i)
                    finger_joint_names.append(joint_name)
            
            if len(finger_joint_ids) == 0 and len(arm_joint_ids) > 0:
                num_arm_joints = len(arm_joint_ids)
                finger_joint_ids = list(range(num_arm_joints, len(all_joint_names)))
                finger_joint_names = [all_joint_names[i] for i in finger_joint_ids]
            
            self._arm_joint_ids_right = torch.tensor(arm_joint_ids, device=self.device) if arm_joint_ids else None
            self._arm_joint_names_right = arm_joint_names
            self._finger_joint_ids_right = torch.tensor(finger_joint_ids, device=self.device) if finger_joint_ids else torch.tensor([], device=self.device, dtype=torch.long)
            self._finger_joint_names_right = finger_joint_names
        except Exception as e:
            print(f"[WARNING] Could not identify right arm joints: {e}. Using fallback method.")
            all_joint_names = self._robot_right.joint_names
            self._arm_joint_ids_right = torch.tensor(list(range(min(6, len(all_joint_names)))), device=self.device)
            self._arm_joint_names_right = all_joint_names[:6] if len(all_joint_names) >= 6 else all_joint_names
            self._finger_joint_ids_right = torch.tensor(list(range(6, len(all_joint_names))), device=self.device) if len(all_joint_names) > 6 else torch.tensor([], device=self.device, dtype=torch.long)
            self._finger_joint_names_right = all_joint_names[6:] if len(all_joint_names) > 6 else []
        
        # Finger targets for right arm
        num_finger_joints = len(self._finger_joint_ids_right) if len(self._finger_joint_ids_right) > 0 else 0
        self.finger_targets_right = torch.zeros((self.num_envs, num_finger_joints), device=self.device)
        
        if num_finger_joints > 0:
            if self.cfg.manual_finger_joint_positions_right is not None:
                self.finger_targets_right = self._get_manual_finger_targets_right()
            else:
                finger_lower_limits = self._robot_right.data.soft_joint_pos_limits[0, self._finger_joint_ids_right, 0]
                self.finger_targets_right[:] = finger_lower_limits.unsqueeze(0)
        
        # Action buffers for right arm
        self.processed_actions_right = torch.zeros((self.num_envs, self._ik_controller_right.action_dim), device=self.device)
        self.actions_right = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        print(f"[INFO] Right arm setup complete:")
        print(f"  Body: {self._body_name_right} (index {self._body_idx_right}, jacobian index {self._jacobi_body_idx_right})")
        print(f"  Arm joints: {len(self._arm_joint_names_right)}")
        print(f"  Finger joints: {len(self._finger_joint_names_right)}")

    def _get_manual_finger_targets_left(self) -> torch.Tensor:
        """Get manual finger joint targets for left arm."""
        if self.cfg.manual_finger_joint_positions_left is None:
            finger_lower_limits = self._robot_left.data.soft_joint_pos_limits[0, self._finger_joint_ids_left, 0]
            return finger_lower_limits.unsqueeze(0).repeat(self.num_envs, 1)
        
        joint_names = self._robot_left.joint_names
        finger_targets = torch.zeros((self.num_envs, len(self._finger_joint_ids_left)), device=self.device)
        
        for i, joint_idx in enumerate(self._finger_joint_ids_left):
            joint_name = joint_names[joint_idx]
            if joint_name in self.cfg.manual_finger_joint_positions_left:
                finger_targets[:, i] = self.cfg.manual_finger_joint_positions_left[joint_name]
            else:
                matched = False
                for pattern, value in self.cfg.manual_finger_joint_positions_left.items():
                    if pattern.upper() in joint_name.upper() or joint_name.upper() in pattern.upper():
                        finger_targets[:, i] = value
                        matched = True
                        break
                if not matched:
                    finger_targets[:, i] = self._robot_left.data.soft_joint_pos_limits[0, joint_idx, 0]
        
        return finger_targets

    def _get_manual_finger_targets_right(self) -> torch.Tensor:
        """Get manual finger joint targets for right arm."""
        if self.cfg.manual_finger_joint_positions_right is None:
            finger_lower_limits = self._robot_right.data.soft_joint_pos_limits[0, self._finger_joint_ids_right, 0]
            return finger_lower_limits.unsqueeze(0).repeat(self.num_envs, 1)
        
        joint_names = self._robot_right.joint_names
        finger_targets = torch.zeros((self.num_envs, len(self._finger_joint_ids_right)), device=self.device)
        
        for i, joint_idx in enumerate(self._finger_joint_ids_right):
            joint_name = joint_names[joint_idx]
            if joint_name in self.cfg.manual_finger_joint_positions_right:
                finger_targets[:, i] = self.cfg.manual_finger_joint_positions_right[joint_name]
            else:
                matched = False
                for pattern, value in self.cfg.manual_finger_joint_positions_right.items():
                    if pattern.upper() in joint_name.upper() or joint_name.upper() in pattern.upper():
                        finger_targets[:, i] = value
                        matched = True
                        break
                if not matched:
                    finger_targets[:, i] = self._robot_right.data.soft_joint_pos_limits[0, joint_idx, 0]
        
        return finger_targets

    def _setup_scene(self):
        """Setup scene with both robots and all sensors."""
        # Create left arm robot (cfg.robot was set to cfg.robot_left in __init__)
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        # Create right arm robot BEFORE cloning
        self._robot_right = Articulation(self._robot_right_cfg)
        self.scene.articulations["robot_right"] = self._robot_right
        
        RigidObject(self.cfg.plate)
        
        # Clone environments after both robots are created
        self.scene.clone_environments(copy_from_source=False)
        
        # Create sensors for left arm
        self.gelsighthand_ff_left = GelSightSensor(self.cfg.gelsighthand_ff_left)
        self.gelsighthand_mf_left = GelSightSensor(self.cfg.gelsighthand_mf_left)
        self.gelsighthand_rf_left = GelSightSensor(self.cfg.gelsighthand_rf_left)
        self.gelsighthand_lf_left = GelSightSensor(self.cfg.gelsighthand_lf_left)
        self.gelsighthand_th_left = GelSightSensor(self.cfg.gelsighthand_th_left)
        
        self.scene.sensors["gelsighthand_ff_left"] = self.gelsighthand_ff_left
        self.scene.sensors["gelsighthand_mf_left"] = self.gelsighthand_mf_left
        self.scene.sensors["gelsighthand_rf_left"] = self.gelsighthand_rf_left
        self.scene.sensors["gelsighthand_lf_left"] = self.gelsighthand_lf_left
        self.scene.sensors["gelsighthand_th_left"] = self.gelsighthand_th_left
        
        # Create sensors for right arm
        self.gelsighthand_ff_right = GelSightSensor(self.cfg.gelsighthand_ff_right)
        self.gelsighthand_mf_right = GelSightSensor(self.cfg.gelsighthand_mf_right)
        self.gelsighthand_rf_right = GelSightSensor(self.cfg.gelsighthand_rf_right)
        self.gelsighthand_lf_right = GelSightSensor(self.cfg.gelsighthand_lf_right)
        self.gelsighthand_th_right = GelSightSensor(self.cfg.gelsighthand_th_right)
        
        self.scene.sensors["gelsighthand_ff_right"] = self.gelsighthand_ff_right
        self.scene.sensors["gelsighthand_mf_right"] = self.gelsighthand_mf_right
        self.scene.sensors["gelsighthand_rf_right"] = self.gelsighthand_rf_right
        self.scene.sensors["gelsighthand_lf_right"] = self.gelsighthand_lf_right
        self.scene.sensors["gelsighthand_th_right"] = self.gelsighthand_th_right
        
        # Global RGB camera
        if not hasattr(self.cfg.arm_camera.offset, "convention"):
            self.cfg.arm_camera.offset.convention = "usd"
        self.arm_camera = Camera(self.cfg.arm_camera)
        self.scene.sensors["arm_camera"] = self.arm_camera
        
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)
        
        # Rigid object
        self.object = RigidObject(self.cfg.object)
        self.scene.rigid_objects["object"] = self.object
        
        # Create folder (articulation)
        self.folder = Articulation(self.cfg.folder)
        self.scene.articulations["folder"] = self.folder
        
        print(f"[INFO] Folder: {self.folder}")
        # 打印folder位置
        
        # 设置文件夹关节的驱动属性（用于平衡GUI拖动能力和重力抵抗能力）
        # self._set_folder_joint_drives()
        
        # 设置文件夹关节的物理限制（上限/下限）
        # self._set_folder_joint_limits()
        
        # 设置文件夹的质量/密度（如果配置了的话）
        # if self.cfg.folder_mass is not None or self.cfg.folder_mass_density is not None:
            # self._set_folder_mass_properties()
        
        # 检查文件夹的kinematic属性和joints（调试用）
        # self.check_folder_kinematic_properties(env_idx=0)
        
        # 递归遍历并打印所有joint的属性
        # self.inspect_all_folder_joints(env_idx=0)
        
        # 注意：不在_setup_scene中禁用FixedJoint，因为此时Articulation还未完全初始化
        # 将在__init__完成后，在第一次reset之前禁用FixedJoint

    def _compute_frame_pose_left(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute left arm end-effector pose in base frame."""
        return self._compute_frame_pose_impl(self._robot_left, self._body_idx_left, self._jacobi_body_idx_left, 
                                            self._offset_pos_left, self._offset_rot_left)

    def _compute_frame_pose_right(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute right arm end-effector pose in base frame."""
        return self._compute_frame_pose_impl(self._robot_right, self._body_idx_right, self._jacobi_body_idx_right,
                                            self._offset_pos_right, self._offset_rot_right)

    def _compute_frame_pose_impl(self, robot, body_idx, jacobi_body_idx, offset_pos, offset_rot) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper to compute end-effector pose for a given robot.
        
        This matches the single-arm version's _compute_frame_pose implementation.
        """
        # Get body pose in world frame (use body_link_pos_w and body_link_quat_w to match single-arm version)
        ee_pos_w = robot.data.body_link_pos_w[:, body_idx]
        ee_quat_w = robot.data.body_link_quat_w[:, body_idx]
        
        # Convert to base frame first
        root_pos_w = robot.data.root_link_pos_w
        root_quat_w = robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        
        # Apply offset (same as single-arm version)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(ee_pose_b, ee_quat_b, offset_pos, offset_rot)
        
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian_left(self) -> torch.Tensor:
        """Compute left arm end-effector Jacobian with offset."""
        # Get world frame jacobian (same as single-arm version)
        jacobian_w = self._robot_left.root_physx_view.get_jacobians()[:, self._jacobi_body_idx_left, :, :]
        # Convert to base frame (same as single-arm version's jacobian_b property)
        base_rot = self._robot_left.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian = jacobian_w.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        # Apply offset transformation (same as single-arm version)
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos_left), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot_left), jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_jacobian_right(self) -> torch.Tensor:
        """Compute right arm end-effector Jacobian with offset."""
        # Get world frame jacobian (same as single-arm version)
        jacobian_w = self._robot_right.root_physx_view.get_jacobians()[:, self._jacobi_body_idx_right, :, :]
        # Convert to base frame (same as single-arm version's jacobian_b property)
        base_rot = self._robot_right.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian = jacobian_w.clone()
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        # Apply offset transformation (same as single-arm version)
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos_right), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot_right), jacobian[:, 3:, :])
        return jacobian

    def _pre_physics_step(self, actions: torch.Tensor):
        """Control end-effector: use provided actions if non-zero, otherwise use scripted trajectory.
        
        If actions are provided (non-zero), use them for IK control.
        Otherwise, use scripted trajectory (lift upward).
        If ik_commands is available (e.g., from virtual object control), use it.
        """
        # Left arm - exactly match single-arm version (lines 703-774)
        # Current EE pose (base frame)
        ee_pos_b_left, ee_quat_b_left = self._compute_frame_pose_left()
        
        # Check if ik_commands is available (for virtual object control, similar to run_shape_touch.py)
        if hasattr(self, 'ik_commands_left') and self.ik_commands_left is not None and torch.any(self.ik_commands_left != 0):
            # Use ik_commands directly (exactly as in run_shape_touch.py)
            # For use_relative_mode=False, ik_commands contains absolute poses in base frame
            self._ik_controller_left.set_command(self.ik_commands_left)
        # Check if processed_actions is set (alternative method)
        elif hasattr(self, 'processed_actions_left') and torch.any(self.processed_actions_left != 0):
            # Use processed_actions directly
            self._ik_controller_left.set_command(self.processed_actions_left, ee_pos_b_left, ee_quat_b_left)
        # Check if actions are provided (non-zero)
        elif actions is not None and torch.any(actions != 0):
            # Use provided actions for IK control
            self.processed_actions_left.zero_()
            trans_scale = self.cfg.action_scale_translation
            rot_scale = self.cfg.action_scale_rotation
            self.processed_actions_left[:, :3] = actions[:, :3] * trans_scale
            self.processed_actions_left[:, 3:6] = actions[:, 3:6] * rot_scale
            self._ik_controller_left.set_command(self.processed_actions_left, ee_pos_b_left, ee_quat_b_left)
        else:
            # Use scripted trajectory: Lift end-effector upward while maintaining grasp
            # Object position in world frame
            object_pos_w = self.object.data.root_pos_w
            
            # Normalized time [0, 1]
            frac = (self.episode_length_buf.to(torch.float32) / max(self.max_episode_length - 1, 1)).unsqueeze(1)

            # Target end-effector position in world frame: x,y aligned with object, z interpolates from grasp to lift
            z_grasp_w = object_pos_w[:, 2:3] + self.cfg.grasp_height
            z_lift_w = object_pos_w[:, 2:3] + self.cfg.lift_height
            z_target_w = z_grasp_w * (1.0 - frac) + z_lift_w * frac
            
            # Target position in world frame
            target_pos_w = torch.cat([object_pos_w[:, 0:2], z_target_w], dim=1)
            
            # Convert target position from world frame to base frame
            root_pos_w = self._robot_left.data.root_link_pos_w
            root_quat_w = self._robot_left.data.root_link_quat_w
            target_pos_b, _ = math_utils.subtract_frame_transforms(
                root_pos_w,
                root_quat_w,
                target_pos_w,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1),
            )

            # IK command: gentle upward movement
            delta_pos = target_pos_b - ee_pos_b_left
            trans_scale = max(self.cfg.action_scale_translation, 1e-6)
            self.processed_actions_left.zero_()
            self.processed_actions_left[:, :3] = torch.clamp(0.1 * delta_pos / trans_scale, -1.0, 1.0)
            self.processed_actions_left[:, 3:6] = 0.0  # No rotation change
            self._ik_controller_left.set_command(self.processed_actions_left, ee_pos_b_left, ee_quat_b_left)

        # Control finger joints: use manual positions if specified and auto control is enabled
        # NOTE: If _auto_finger_control is False, allow manual control (e.g., via GUI or keyboard)
        if len(self._finger_joint_ids_left) > 0:
            if self._auto_finger_control_left and self.cfg.manual_finger_joint_positions_left is not None:
                # Use manual finger positions from config
                finger_targets = self._get_manual_finger_targets_left()
                # Smooth transition using exponential moving average
                alpha = 0.1  # Smoothing factor
                self.finger_targets_left = alpha * finger_targets + (1.0 - alpha) * self.finger_targets_left
            # else: Don't modify finger_targets - allow manual control (e.g., GUI or keyboard)
        
        # Right arm - exactly match single-arm version (lines 703-774), just variable names differ
        # Current EE pose (base frame)
        ee_pos_b_right, ee_quat_b_right = self._compute_frame_pose_right()
        
        # Check if ik_commands is available (for virtual object control, similar to run_shape_touch.py)
        if hasattr(self, 'ik_commands_right') and self.ik_commands_right is not None and torch.any(self.ik_commands_right != 0):
            # Use ik_commands directly (exactly as in run_shape_touch.py)
            # For use_relative_mode=False, ik_commands contains absolute poses in base frame
            self._ik_controller_right.set_command(self.ik_commands_right)
        # Check if processed_actions is set (alternative method)
        elif hasattr(self, 'processed_actions_right') and torch.any(self.processed_actions_right != 0):
            # Use processed_actions directly
            self._ik_controller_right.set_command(self.processed_actions_right, ee_pos_b_right, ee_quat_b_right)
        # Check if actions are provided (non-zero)
        elif actions is not None and torch.any(actions != 0):
            # Use provided actions for IK control
            self.processed_actions_right.zero_()
            trans_scale = self.cfg.action_scale_translation
            rot_scale = self.cfg.action_scale_rotation
            self.processed_actions_right[:, :3] = actions[:, :3] * trans_scale
            self.processed_actions_right[:, 3:6] = actions[:, 3:6] * rot_scale
            self._ik_controller_right.set_command(self.processed_actions_right, ee_pos_b_right, ee_quat_b_right)
        else:
            # Use scripted trajectory: Lift end-effector upward while maintaining grasp
            # Object position in world frame
            object_pos_w = self.object.data.root_pos_w
            
            # Normalized time [0, 1]
            frac = (self.episode_length_buf.to(torch.float32) / max(self.max_episode_length - 1, 1)).unsqueeze(1)

            # Target end-effector position in world frame: x,y aligned with object, z interpolates from grasp to lift
            z_grasp_w = object_pos_w[:, 2:3] + self.cfg.grasp_height
            z_lift_w = object_pos_w[:, 2:3] + self.cfg.lift_height
            z_target_w = z_grasp_w * (1.0 - frac) + z_lift_w * frac
            
            # Target position in world frame
            target_pos_w = torch.cat([object_pos_w[:, 0:2], z_target_w], dim=1)
            
            # Convert target position from world frame to base frame
            root_pos_w = self._robot_right.data.root_link_pos_w
            root_quat_w = self._robot_right.data.root_link_quat_w
            target_pos_b, _ = math_utils.subtract_frame_transforms(
                root_pos_w,
                root_quat_w,
                target_pos_w,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1),
            )

            # IK command: gentle upward movement
            delta_pos = target_pos_b - ee_pos_b_right
            trans_scale = max(self.cfg.action_scale_translation, 1e-6)
            self.processed_actions_right.zero_()
            self.processed_actions_right[:, :3] = torch.clamp(0.1 * delta_pos / trans_scale, -1.0, 1.0)
            self.processed_actions_right[:, 3:6] = 0.0  # No rotation change
            self._ik_controller_right.set_command(self.processed_actions_right, ee_pos_b_right, ee_quat_b_right)

        # Control finger joints: use manual positions if specified and auto control is enabled
        # NOTE: If _auto_finger_control is False, allow manual control (e.g., via GUI or keyboard)
        if len(self._finger_joint_ids_right) > 0:
            if self._auto_finger_control_right and self.cfg.manual_finger_joint_positions_right is not None:
                # Use manual finger positions from config
                finger_targets = self._get_manual_finger_targets_right()
                # Smooth transition using exponential moving average
                alpha = 0.1  # Smoothing factor
                self.finger_targets_right = alpha * finger_targets + (1.0 - alpha) * self.finger_targets_right
            # else: Don't modify finger_targets - allow manual control (e.g., GUI or keyboard)

    def _get_observations(self) -> dict:
        """Get observations."""
        # Left arm - exactly match single-arm version (lines 595-649), just variable names differ
        ee_pos_b_left, ee_quat_b_left = self._compute_frame_pose_left()
        ee_euler_left = euler_xyz_from_quat(ee_quat_b_left)
        ex_left = wrap_to_pi(ee_euler_left[0]).unsqueeze(1)
        ey_left = wrap_to_pi(ee_euler_left[1]).unsqueeze(1)
        ez_left = wrap_to_pi(ee_euler_left[2]).unsqueeze(1)

        object_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot_left.data.root_link_pos_w,
            self._robot_left.data.root_link_quat_w,
            self.object.data.root_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1),
        )

        joint_pos_left = self._robot_left.data.joint_pos[:, :]
        # Use first 11 joints for observation (adjust based on your robot)
        num_joints_obs = min(11, joint_pos_left.shape[1])
        joint_pos_obs = joint_pos_left[:, :num_joints_obs]

        proprio_obs = torch.cat(
            (ee_pos_b_left, ex_left, ey_left, ez_left, object_pos_b, joint_pos_obs, self.processed_actions_left),
            dim=-1,
        )

        # Stack 5 tactile RGB images: (num_envs, H, W, 15) = 5 * 3 channels
        imgs = []  # Initialize imgs list
        if self.cfg.use_images:
            for sensor in (self.gelsighthand_ff_left, self.gelsighthand_mf_left, self.gelsighthand_rf_left, 
                          self.gelsighthand_lf_left, self.gelsighthand_th_left):
                tactile = sensor.data.output.get("tactile_rgb", None)
                if tactile is not None:
                    tensor = torch.as_tensor(tactile, device=self.device, dtype=torch.float32)
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                    imgs.append(tensor)

            if len(imgs) == 0:
                vision_obs = torch.zeros((self.num_envs, 32, 32, 15), device=self.device)
            else:
                # Ensure 5 images
                while len(imgs) < 5:
                    imgs.append(torch.zeros_like(imgs[0]) if len(imgs) > 0 else torch.zeros((self.num_envs, 32, 32, 3), device=self.device))
                vision_obs = torch.cat(imgs, dim=-1)
        else:
            vision_obs = torch.zeros((self.num_envs, 32, 32, 15), device=self.device)

        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}

        if self.cfg.debug_vis and hasattr(self, "visualizers") and "Observations" in self.visualizers:
            for i, sensor_name in enumerate(["ff", "mf", "rf", "lf", "th"]):
                if i < len(imgs):
                    self.visualizers["Observations"].terms[f"sensor_output_tactile_{sensor_name}"] = imgs[i]

        return {"policy": obs}

    def _apply_action(self):
        """Apply IK control to both arms and finger joints."""
        # Left arm - exactly match single-arm version
        ee_pos_curr_b_left, ee_quat_curr_b_left = self._compute_frame_pose_left()
        joint_pos_left = self._robot_left.data.joint_pos[:, :]

        if ee_pos_curr_b_left.norm() != 0:
            jacobian_left = self._compute_frame_jacobian_left()
            joint_pos_des_left = self._ik_controller_left.compute(ee_pos_curr_b_left, ee_quat_curr_b_left, jacobian_left, joint_pos_left)
        else:
            joint_pos_des_left = joint_pos_left.clone()

        # Apply finger joint targets if available
        if len(self._finger_joint_ids_left) > 0:
            joint_pos_des_left[:, self._finger_joint_ids_left] = self.finger_targets_left

        self._robot_left.set_joint_position_target(joint_pos_des_left)
        
        # Right arm - exactly match single-arm version, just variable names differ
        ee_pos_curr_b_right, ee_quat_curr_b_right = self._compute_frame_pose_right()
        joint_pos_right = self._robot_right.data.joint_pos[:, :]

        if ee_pos_curr_b_right.norm() != 0:
            jacobian_right = self._compute_frame_jacobian_right()
            joint_pos_des_right = self._ik_controller_right.compute(ee_pos_curr_b_right, ee_quat_curr_b_right, jacobian_right, joint_pos_right)
        else:
            joint_pos_des_right = joint_pos_right.clone()

        # Apply finger joint targets if available
        if len(self._finger_joint_ids_right) > 0:
            joint_pos_des_right[:, self._finger_joint_ids_right] = self.finger_targets_right

        self._robot_right.set_joint_position_target(joint_pos_des_right)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments for dual-arm setup."""
        env_ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
        
        # Call parent reset first
        super()._reset_idx(env_ids)
        
        # Update scene to get latest robot state (same as single-arm version)
        self.scene.update(dt=0.0)
        
        # Left arm - exactly match single-arm version
        # Reset to default joint positions (original default)
        joint_pos_left = self._robot_left.data.default_joint_pos[env_ids]
        joint_vel_left = torch.zeros_like(joint_pos_left)
        self._robot_left.set_joint_position_target(joint_pos_left, env_ids=env_ids)
        self._robot_left.write_joint_state_to_sim(joint_pos_left, joint_vel_left, env_ids=env_ids)
        
        # Right arm: Different initial pose to avoid collision with left arm
        # Rotate shoulder_pan by 180 degrees (π) so right arm points in opposite direction
        joint_pos_right = self._robot_right.data.default_joint_pos[env_ids].clone()
        if self._arm_joint_ids_right is not None and len(self._arm_joint_ids_right) >= 6:
            # Right arm: shoulder_pan=π (point backward/opposite direction), shoulder_lift=-0.5, elbow=1.0, wrist_1=-0.5
            # This makes right arm point away from left arm to avoid collision
            import math
            arm_joint_targets = torch.tensor([math.pi, -0.5, 1.0, -0.5, 0.0, 0.0], device=self.device)
            for i, joint_idx in enumerate(self._arm_joint_ids_right[:6]):
                if joint_idx < joint_pos_right.shape[1]:
                    joint_pos_right[:, joint_idx] = arm_joint_targets[i]
        
        joint_vel_right = torch.zeros_like(joint_pos_right)
        self._robot_right.set_joint_position_target(joint_pos_right, env_ids=env_ids)
        self._robot_right.write_joint_state_to_sim(joint_pos_right, joint_vel_right, env_ids=env_ids)
        
        # Reset finger targets for left arm - exactly match single-arm version
        if len(self._finger_joint_ids_left) > 0:
            if self.cfg.manual_finger_joint_positions_left is not None:
                manual_targets_left = self._get_manual_finger_targets_left()
                self.finger_targets_left[env_ids] = manual_targets_left[env_ids]
            else:
                finger_lower_limits_left = self._robot_left.data.soft_joint_pos_limits[:, self._finger_joint_ids_left, 0]
                self.finger_targets_left[env_ids] = finger_lower_limits_left[env_ids]

        # Reset finger targets for right arm
        if len(self._finger_joint_ids_right) > 0:
            if self.cfg.manual_finger_joint_positions_right is not None:
                manual_targets_right = self._get_manual_finger_targets_right()
                self.finger_targets_right[env_ids] = manual_targets_right[env_ids]
            else:
                finger_lower_limits_right = self._robot_right.data.soft_joint_pos_limits[:, self._finger_joint_ids_right, 0]
                self.finger_targets_right[env_ids] = finger_lower_limits_right[env_ids]

        # Reset folder position and joint states
        if hasattr(self, 'folder'):
            # 直接使用配置的位置，而不是default_root_state
            # 因为default_root_state可能被sim.reset()覆盖为USD文件的默认值（可能是(0,0,0)）
            folder_pos = torch.tensor(self.cfg.folder.init_state.pos, device=self.device)
            folder_rot = torch.tensor(self.cfg.folder.init_state.rot, device=self.device)
            
            # 构建root_state tensor (13维: pos(3) + rot(4) + lin_vel(3) + ang_vel(3))
            root_state = torch.zeros((len(env_ids), 13), device=self.device)
            root_state[:, :3] = folder_pos.unsqueeze(0) + self.scene.env_origins[env_ids]
            root_state[:, 3:7] = folder_rot.unsqueeze(0)
            root_state[:, 7:] = 0.0  # 速度设为0
            
            # 写入仿真，确保folder在正确位置
            self.folder.write_root_state_to_sim(root_state, env_ids=env_ids)
            
            # Reset folder joint positions
            folder_joint_pos = self.folder.data.default_joint_pos[env_ids].clone()
            folder_joint_vel = torch.zeros_like(folder_joint_pos)
            
            # 如果配置了初始关节角度，应用它
            if hasattr(self.cfg, 'folder_joint_initial_angle') and self.cfg.folder_joint_initial_angle is not None:
                # 找到关节索引
                joint_names = self.folder.joint_names
                if self.cfg.folder_joint_name in joint_names:
                    joint_idx = joint_names.index(self.cfg.folder_joint_name)
                    initial_angle = self.cfg.folder_joint_initial_angle
                    folder_joint_pos[:, joint_idx] = initial_angle
                    print(f"[DEBUG] 设置文件夹关节 '{self.cfg.folder_joint_name}' 初始角度为 {initial_angle:.4f} rad ({initial_angle * 180 / math.pi:.2f}°)")
                    
                    # 重要：同时设置关节的驱动目标位置，这样驱动才会保持关节在目标位置
                    # 如果不设置驱动目标，驱动可能会将关节拉回到默认位置（可能是0或闭合角度）
                    self.folder.set_joint_position_target(folder_joint_pos, env_ids=env_ids)
            
            self.folder.write_joint_state_to_sim(folder_joint_pos, folder_joint_vel, env_ids=env_ids)

        self.processed_actions_left.zero_()
        self.actions_left[env_ids] = 0.0
        
        # Reset action buffers for right arm
        if hasattr(self, 'processed_actions_right'):
            self.processed_actions_right.zero_()
        
        # Reset episode length
        self.episode_length_buf[env_ids] = 0

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute end-effector pose (defaults to left arm for compatibility)."""
        return self._compute_frame_pose_left()

    def _compute_frame_jacobian(self) -> torch.Tensor:
        """Compute end-effector Jacobian (defaults to left arm for compatibility)."""
        return self._compute_frame_jacobian_left()

    # ============================================================================
    # Folder Joint Inspection Methods
    # ============================================================================

    def inspect_all_folder_joints(self, env_idx: int = 0):
        """递归遍历文件夹的所有joint并打印其属性。
        
        Args:
            env_idx: 环境索引，默认为0（第一个环境）
        """
        try:
            import omni.usd
            from pxr import PhysxSchema, UsdPhysics
            
            # 获取 USD stage
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("[WARNING] 无法获取 USD stage，无法检查joint属性")
                return
            
            folder_prim_path = f"/World/envs/env_{env_idx}/folder"
            folder_prim = stage.GetPrimAtPath(folder_prim_path)
            
            if not folder_prim or not folder_prim.IsValid():
                print(f"[WARNING] 无法找到文件夹 prim: {folder_prim_path}")
                return
            
            print(f"\n[INFO] ========== 文件夹Joint属性检查 (环境 {env_idx}) ==========")
            print(f"[INFO] Prim路径: {folder_prim_path}")
            print(f"[INFO] 开始递归遍历所有joints...\n")
            
            # 收集所有找到的FixedJoint
            all_fixed_joints = []
            
            def inspect_joint_recursive(prim, depth=0):
                """递归检查prim及其子节点中的所有joint"""
                nonlocal all_fixed_joints
                indent = "  " * depth
                prim_path = str(prim.GetPath())
                prim_type = prim.GetTypeName()
                
                # 检查当前prim是否是joint
                is_joint = False
                joint_type = None
                
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    is_joint = True
                    joint_type = "RevoluteJoint (旋转关节)"
                elif prim.IsA(UsdPhysics.PrismaticJoint):
                    is_joint = True
                    joint_type = "PrismaticJoint (平移关节)"
                elif prim.IsA(UsdPhysics.FixedJoint):
                    is_joint = True
                    joint_type = "FixedJoint (固定关节)"
                    # 记录FixedJoint路径
                    fixed_joint_path = str(prim.GetPath())
                    if fixed_joint_path not in all_fixed_joints:
                        all_fixed_joints.append(fixed_joint_path)
                elif "Joint" in prim_type:
                    is_joint = True
                    joint_type = prim_type
                
                if is_joint:
                    print(f"{indent}[JOINT] {prim.GetName()}")
                    print(f"{indent}  路径: {prim_path}")
                    print(f"{indent}  类型: {joint_type}")
                    
                    # 检查joint的基本属性
                    if prim.HasAttribute("physics:jointEnabled"):
                        enabled = prim.GetAttribute("physics:jointEnabled").Get()
                        print(f"{indent}  启用状态: {enabled}")
                        if not enabled:
                            print(f"{indent}    ⚠️  警告：此joint已被禁用")
                    else:
                        print(f"{indent}  启用状态: 未设置（默认True）")
                    
                    # 检查RevoluteJoint的属性
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        if prim.HasAttribute("physics:axis"):
                            axis = prim.GetAttribute("physics:axis").Get()
                            print(f"{indent}  旋转轴: {axis}")
                        
                        if prim.HasAttribute("physics:lowerLimit"):
                            lower = prim.GetAttribute("physics:lowerLimit").Get()
                            print(f"{indent}  下限: {lower} rad ({lower * 180 / math.pi:.2f} deg)")
                        else:
                            print(f"{indent}  下限: 未设置")
                        
                        if prim.HasAttribute("physics:upperLimit"):
                            upper = prim.GetAttribute("physics:upperLimit").Get()
                            print(f"{indent}  上限: {upper} rad ({upper * 180 / math.pi:.2f} deg)")
                        else:
                            print(f"{indent}  上限: 未设置")
                    
                    # 检查PrismaticJoint的属性
                    elif prim.IsA(UsdPhysics.PrismaticJoint):
                        if prim.HasAttribute("physics:axis"):
                            axis = prim.GetAttribute("physics:axis").Get()
                            print(f"{indent}  平移轴: {axis}")
                        
                        if prim.HasAttribute("physics:lowerLimit"):
                            lower = prim.GetAttribute("physics:lowerLimit").Get()
                            print(f"{indent}  下限: {lower} m")
                        else:
                            print(f"{indent}  下限: 未设置")
                        
                        if prim.HasAttribute("physics:upperLimit"):
                            upper = prim.GetAttribute("physics:upperLimit").Get()
                            print(f"{indent}  上限: {upper} m")
                        else:
                            print(f"{indent}  上限: 未设置")
                    
                    # 检查FixedJoint的特殊标记
                    elif prim.IsA(UsdPhysics.FixedJoint):
                        print(f"{indent}  ⚠️  警告：这是FixedJoint，会固定连接的物体！")
                    
                    # 检查连接的刚体
                    if prim.HasAttribute("physics:body0"):
                        body0 = prim.GetAttribute("physics:body0").Get()
                        print(f"{indent}  body0: {body0}")
                    
                    if prim.HasAttribute("physics:body1"):
                        body1 = prim.GetAttribute("physics:body1").Get()
                        print(f"{indent}  body1: {body1}")
                    
                    # 检查PhysX属性
                    if prim.HasAPI(PhysxSchema.PhysxJointAPI):
                        print(f"{indent}  有 PhysX Joint API")
                        max_vel_attr = prim.GetAttribute("physx:joint:maxJointVelocity")
                        if max_vel_attr:
                            max_vel = max_vel_attr.Get()
                            print(f"{indent}  最大速度: {max_vel}")
                    
                    print()  # 空行分隔
                
                # 递归检查子节点
                for child in prim.GetChildren():
                    inspect_joint_recursive(child, depth + 1)
            
            # 从根节点开始递归检查
            inspect_joint_recursive(folder_prim, depth=0)
            
            # 打印总结
            print(f"[INFO] ========== Joint检查总结 ==========")
            print(f"[INFO] 找到的FixedJoint数量: {len(all_fixed_joints)}")
            if all_fixed_joints:
                print(f"[INFO] FixedJoint路径列表:")
                for i, fixed_joint_path in enumerate(all_fixed_joints):
                    print(f"[INFO]   {i+1}. {fixed_joint_path}")
                print(f"[INFO] 注意：这些FixedJoint将在Articulation初始化后根据配置决定是否禁用")
                # 保存FixedJoint路径，稍后在初始化后禁用（如果需要）
                self._fixed_joints_to_disable = all_fixed_joints
            else:
                print(f"[INFO] ✓ 未发现FixedJoint")
                self._fixed_joints_to_disable = []
            
            print(f"[INFO] ====================================\n")
            
        except Exception as e:
            print(f"[WARNING] 检查joint属性时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def check_folder_kinematic_properties(self, env_idx: int = 0):
        """检查文件夹的kinematic属性（调试用）。
        
        注意：文件夹是Articulation，需要检查各个link（子节点）的RigidBodyAPI。
        
        Args:
            env_idx: 环境索引，默认为0（第一个环境）
        """
        try:
            import omni.usd
            from pxr import UsdPhysics
            
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("[WARNING] 无法获取 USD stage，无法检查kinematic属性")
                return
            
            folder_prim_path = f"/World/envs/env_{env_idx}/folder"
            folder_prim = stage.GetPrimAtPath(folder_prim_path)
            
            if not folder_prim or not folder_prim.IsValid():
                print(f"[WARNING] 无法找到文件夹 prim: {folder_prim_path}")
                return
            
            print(f"\n[DEBUG] ========== 检查文件夹Kinematic属性 (环境 {env_idx}) ==========")
            print(f"[DEBUG] Prim路径: {folder_prim_path}")
            print(f"[DEBUG] 注意：文件夹是Articulation，需要检查各个link的RigidBodyAPI\n")
            
            # 检查Articulation根节点的属性
            articulation_api = UsdPhysics.ArticulationRootAPI(folder_prim)
            if articulation_api:
                print(f"[DEBUG] 检查Articulation根节点属性:")
                # 检查是否启用
                enabled_attr = folder_prim.GetAttribute("physics:articulationEnabled")
                if enabled_attr and enabled_attr.HasAuthoredValue():
                    enabled = enabled_attr.Get()
                    print(f"  physics:articulationEnabled = {enabled}")
                    if not enabled:
                        print(f"  ⚠️  警告：Articulation被禁用！这会导致文件夹完全无法移动！")
                else:
                    print(f"  physics:articulationEnabled 未设置（默认True，已启用）")
                
                # 检查根节点的kinematic状态
                root_rigid_body_api = UsdPhysics.RigidBodyAPI(folder_prim)
                if root_rigid_body_api:
                    root_kinematic_attr = root_rigid_body_api.GetKinematicEnabledAttr()
                    if root_kinematic_attr and root_kinematic_attr.HasAuthoredValue():
                        root_kinematic = root_kinematic_attr.Get()
                        print(f"  根节点 physics:kinematicEnabled = {root_kinematic}")
                        if root_kinematic:
                            print(f"  ⚠️  警告：根节点被设置为kinematic！这会导致整个Articulation无法移动！")
                
                print()
            
            # 递归检查所有子节点的RigidBodyAPI
            def check_prim_rigid_body(prim, depth=0):
                """递归检查prim及其子节点的RigidBodyAPI"""
                indent = "  " * depth
                prim_path = str(prim.GetPath())
                
                # 检查当前prim是否有RigidBodyAPI
                rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
                if rigid_body_api:
                    print(f"{indent}[DEBUG] 检查 {prim_path}:")
                    
                    # 检查kinematic属性
                    kinematic_attr = rigid_body_api.GetKinematicEnabledAttr()
                    if kinematic_attr and kinematic_attr.HasAuthoredValue():
                        is_kinematic = kinematic_attr.Get()
                        print(f"{indent}  physics:kinematicEnabled = {is_kinematic}")
                        if is_kinematic:
                            print(f"{indent}  ⚠️  警告：此link被设置为kinematic（不可移动）！")
                        else:
                            print(f"{indent}  ✓ 此link是可移动的（dynamic）")
                    else:
                        print(f"{indent}  physics:kinematicEnabled 未设置（默认False，可移动）")
                    
                    # 检查其他相关属性
                    disable_gravity_attr = prim.GetAttribute("physics:disableGravity")
                    if disable_gravity_attr and disable_gravity_attr.HasAuthoredValue():
                        disable_gravity = disable_gravity_attr.Get()
                        print(f"{indent}  physics:disableGravity = {disable_gravity}")
                    
                    print()  # 空行分隔
                
                # 递归检查子节点
                for child in prim.GetChildren():
                    check_prim_rigid_body(child, depth + 1)
            
            # 从根节点开始递归检查
            check_prim_rigid_body(folder_prim, depth=0)
            
            print(f"[DEBUG] ========================================================\n")
            
        except Exception as e:
            print(f"[WARNING] 检查kinematic属性时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _disable_fixed_joints(self):
        """禁用之前找到的FixedJoint（在Articulation初始化后调用）。"""
        if not hasattr(self, '_fixed_joints_to_disable') or not self._fixed_joints_to_disable:
            return
        
        try:
            import omni.usd
            from pxr import UsdPhysics
            
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("[WARNING] 无法获取 USD stage，无法禁用FixedJoint")
                return
            
            print(f"[DEBUG] 禁用 {len(self._fixed_joints_to_disable)} 个FixedJoint...")
            for fixed_joint_path in self._fixed_joints_to_disable:
                try:
                    fixed_joint_prim = stage.GetPrimAtPath(fixed_joint_path)
                    if fixed_joint_prim and fixed_joint_prim.IsValid():
                        # 尝试设置joint的enabled属性为False（只禁用，不删除）
                        enabled_attr = fixed_joint_prim.GetAttribute("physics:jointEnabled")
                        if enabled_attr:
                            enabled_attr.Set(False)
                            # 验证是否真的被禁用了
                            enabled_value = enabled_attr.Get()
                            if enabled_value == False:
                                print(f"[DEBUG] ✓ 已成功禁用FixedJoint: {fixed_joint_path} (physics:jointEnabled={enabled_value})")
                            else:
                                print(f"[WARNING] FixedJoint可能未被禁用: {fixed_joint_path} (physics:jointEnabled={enabled_value})")
                        else:
                            # 如果没有enabled属性，尝试删除prim（更彻底的方法）
                            print(f"[WARNING] FixedJoint {fixed_joint_path} 没有physics:jointEnabled属性，尝试删除...")
                            try:
                                from pxr import Sdf
                                stage.RemovePrim(fixed_joint_path)
                                print(f"[DEBUG] ✓ 已删除FixedJoint: {fixed_joint_path}")
                            except Exception as del_e:
                                print(f"[WARNING] 无法删除FixedJoint {fixed_joint_path}: {del_e}")
                    else:
                        print(f"[WARNING] 无法找到FixedJoint: {fixed_joint_path}")
                except Exception as e:
                    print(f"[WARNING] 无法禁用FixedJoint {fixed_joint_path}: {e}")
                    import traceback
                    traceback.print_exc()
            print()
            
            # 清除列表，避免重复处理
            self._fixed_joints_to_disable = []
        except Exception as e:
            print(f"[WARNING] 禁用FixedJoint时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _set_folder_joint_drives(self):
        """设置文件夹关节的驱动属性，平衡 GUI 拖动能力和重力抵抗能力。
        
        将关节的 drive stiffness 和 damping 设置为配置的值（而不是原来的极高值），
        以允许 GUI 拖动同时能够抵抗重力。
        """
        try:
            import omni.usd
            from pxr import UsdPhysics, Sdf
            
            # 获取 USD stage
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("[WARNING] 无法获取 USD stage，无法设置关节驱动属性")
                return
            
            # 获取所有环境的关节并设置驱动属性
            for env_idx in range(self.num_envs):
                folder_prim_path = f"/World/envs/env_{env_idx}/folder"
                folder_prim = stage.GetPrimAtPath(folder_prim_path)
                
                if not folder_prim or not folder_prim.IsValid():
                    continue
                
                # 找到指定名称的关节并设置驱动属性
                def set_joint_drive(joint_prim):
                    """设置单个关节的驱动属性"""
                    from pxr import UsdPhysics, Sdf
                    
                    joint_name = joint_prim.GetName()
                    if joint_name != self.cfg.folder_joint_name:
                        return  # 只设置指定的关节
                    
                    joint_type = "RevoluteJoint" if joint_prim.IsA(UsdPhysics.RevoluteJoint) else "Unknown"
                    
                    # 对于 RevoluteJoint，使用 drive:angular:physics:*
                    angular_stiffness_attr = joint_prim.GetAttribute("drive:angular:physics:stiffness")
                    angular_damping_attr = joint_prim.GetAttribute("drive:angular:physics:damping")
                    
                    # 方法2: drive:angular:* (如果没有physics:前缀)
                    if not angular_stiffness_attr or not angular_stiffness_attr.IsValid():
                        angular_stiffness_attr = joint_prim.GetAttribute("drive:angular:stiffness")
                    if not angular_damping_attr or not angular_damping_attr.IsValid():
                        angular_damping_attr = joint_prim.GetAttribute("drive:angular:damping")
                    
                    # 设置角驱动（用于RevoluteJoint）
                    if angular_stiffness_attr and angular_stiffness_attr.IsValid():
                        angular_stiffness_attr.Set(self.cfg.folder_joint_drive_stiffness)
                    elif joint_prim.IsA(UsdPhysics.RevoluteJoint):
                        # 如果属性不存在，尝试创建它
                        try:
                            # 先尝试 drive:angular:physics:stiffness
                            angular_stiffness_attr = joint_prim.CreateAttribute("drive:angular:physics:stiffness", Sdf.ValueTypeNames.Double)
                            if angular_stiffness_attr:
                                angular_stiffness_attr.Set(self.cfg.folder_joint_drive_stiffness)
                        except:
                            try:
                                # 如果失败，尝试 drive:angular:stiffness
                                angular_stiffness_attr = joint_prim.CreateAttribute("drive:angular:stiffness", Sdf.ValueTypeNames.Double)
                                if angular_stiffness_attr:
                                    angular_stiffness_attr.Set(self.cfg.folder_joint_drive_stiffness)
                            except:
                                pass
                    
                    if angular_damping_attr and angular_damping_attr.IsValid():
                        angular_damping_attr.Set(self.cfg.folder_joint_drive_damping)
                    elif joint_prim.IsA(UsdPhysics.RevoluteJoint):
                        # 如果属性不存在，尝试创建它
                        try:
                            # 先尝试 drive:angular:physics:damping
                            angular_damping_attr = joint_prim.CreateAttribute("drive:angular:physics:damping", Sdf.ValueTypeNames.Double)
                            if angular_damping_attr:
                                angular_damping_attr.Set(self.cfg.folder_joint_drive_damping)
                        except:
                            try:
                                # 如果失败，尝试 drive:angular:damping
                                angular_damping_attr = joint_prim.CreateAttribute("drive:angular:damping", Sdf.ValueTypeNames.Double)
                                if angular_damping_attr:
                                    angular_damping_attr.Set(self.cfg.folder_joint_drive_damping)
                            except:
                                pass
                    
                    # 调试输出：显示哪些属性被设置了
                    print(f"[DEBUG] 设置关节 '{joint_name}' ({joint_type}) 驱动属性:")
                    print(f"  角驱动: stiffness={angular_stiffness_attr is not None} (值={self.cfg.folder_joint_drive_stiffness}), damping={angular_damping_attr is not None} (值={self.cfg.folder_joint_drive_damping})")
                    
                    # 检查是否被锁定
                    locked_attr = joint_prim.GetAttribute("physics:jointEnabled")
                    if locked_attr:
                        enabled = locked_attr.Get()
                        print(f"  关节启用状态: {enabled}")
                        if not enabled:
                            # 如果被禁用，启用它
                            locked_attr.Set(True)
                            print(f"  [INFO] 已启用被禁用的关节 '{joint_name}'")
                
                # 遍历所有子 prim 查找关节
                def find_and_set_joints(prim):
                    """递归查找并设置所有关节的驱动属性"""
                    from pxr import UsdPhysics
                    
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        set_joint_drive(prim)
                    for child in prim.GetChildren():
                        find_and_set_joints(child)
                
                find_and_set_joints(folder_prim)
            
            print(f"[INFO] 已设置文件夹关节的驱动属性: stiffness={self.cfg.folder_joint_drive_stiffness}, damping={self.cfg.folder_joint_drive_damping}")
            
        except Exception as e:
            print(f"[WARNING] 设置关节驱动属性时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _set_folder_joint_limits(self):
        """设置文件夹关节的物理限制（上限/下限）。
        
        根据配置中的 folder_joint_min_angle, folder_joint_max_angle
        来设置USD文件中关节的 lowerLimit 和 upperLimit 属性。
        """
        if not self.cfg.set_folder_joint_limits_in_usd:
            return  # 如果配置为False，跳过设置
        
        try:
            import omni.usd
            from pxr import UsdPhysics, Sdf
            
            # 获取 USD stage
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("[WARNING] 无法获取 USD stage，无法设置关节限制")
                return
            
            # 获取所有环境的关节并设置限制
            for env_idx in range(self.num_envs):
                folder_prim_path = f"/World/envs/env_{env_idx}/folder"
                folder_prim = stage.GetPrimAtPath(folder_prim_path)
                
                if not folder_prim or not folder_prim.IsValid():
                    continue
                
                def set_joint_limits(joint_prim, joint_name):
                    """设置单个关节的限制"""
                    if joint_name != self.cfg.folder_joint_name:
                        return  # 只设置指定的关节
                    
                    # 设置旋转关节的限制（RevoluteJoint）
                    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
                        lower_limit_attr = joint_prim.GetAttribute("physics:lowerLimit")
                        upper_limit_attr = joint_prim.GetAttribute("physics:upperLimit")
                        
                        if lower_limit_attr and lower_limit_attr.IsValid():
                            lower_limit_attr.Set(self.cfg.folder_joint_min_angle)
                        else:
                            lower_limit_attr = joint_prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Double)
                            if lower_limit_attr:
                                lower_limit_attr.Set(self.cfg.folder_joint_min_angle)
                        
                        if upper_limit_attr and upper_limit_attr.IsValid():
                            upper_limit_attr.Set(self.cfg.folder_joint_max_angle)
                        else:
                            upper_limit_attr = joint_prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Double)
                            if upper_limit_attr:
                                upper_limit_attr.Set(self.cfg.folder_joint_max_angle)
                        
                        print(f"[INFO] 设置旋转关节 '{joint_name}' 限制: [{self.cfg.folder_joint_min_angle:.6f}, {self.cfg.folder_joint_max_angle:.6f}] rad "
                              f"({self.cfg.folder_joint_min_angle * 180 / math.pi:.2f}°, {self.cfg.folder_joint_max_angle * 180 / math.pi:.2f}°)")
                
                # 递归查找所有关节
                def find_and_set_joint_limits(prim):
                    """递归查找并设置关节限制"""
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        joint_name = prim.GetName()
                        set_joint_limits(prim, joint_name)
                    for child in prim.GetChildren():
                        find_and_set_joint_limits(child)
                
                find_and_set_joint_limits(folder_prim)
            
            print(f"[INFO] 已设置文件夹关节的物理限制")
            
        except Exception as e:
            print(f"[WARNING] 设置关节限制时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _set_folder_mass_properties(self):
        """设置文件夹的质量属性（密度或质量）。
        
        如果配置了folder_mass或folder_mass_density，则通过USD API设置质量。
        注意：此方法需要在场景初始化后调用。
        """
        print(f"[DEBUG] 正在设置文件夹的质量属性")
        try:
            import omni.usd
            from pxr import UsdPhysics, Sdf
            
            # 获取 USD stage
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("[WARNING] 无法获取 USD stage，无法设置质量属性")
                return
            
            # 获取所有环境的folder并设置质量
            for env_idx in range(self.num_envs):
                folder_prim_path = f"/World/envs/env_{env_idx}/folder"
                folder_prim = stage.GetPrimAtPath(folder_prim_path)
                
                if not folder_prim or not folder_prim.IsValid():
                    continue
                
                # 找到所有rigid body links并设置质量
                def set_link_mass(prim):
                    """设置单个link的质量"""
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        # 如果设置了质量，直接设置质量
                        if self.cfg.folder_mass is not None:
                            mass_attr = prim.GetAttribute("physics:mass")
                            if mass_attr and mass_attr.IsValid():
                                mass_attr.Set(self.cfg.folder_mass)
                                print(f"[DEBUG] 设置 {prim.GetPath()} 的质量为 {self.cfg.folder_mass} kg")
                            else:
                                # 如果属性不存在，创建它（使用正确的类型）
                                mass_attr = prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Double)
                                if mass_attr:
                                    mass_attr.Set(self.cfg.folder_mass)
                                    print(f"[DEBUG] 创建并设置 {prim.GetPath()} 的质量为 {self.cfg.folder_mass} kg")
                        # 如果设置了密度，设置密度
                        elif self.cfg.folder_mass_density is not None:
                            density_attr = prim.GetAttribute("physics:density")
                            if density_attr and density_attr.IsValid():
                                density_attr.Set(self.cfg.folder_mass_density)
                                print(f"[DEBUG] 设置 {prim.GetPath()} 的密度为 {self.cfg.folder_mass_density} kg/m³")
                            else:
                                # 如果属性不存在，创建它（使用正确的类型）
                                density_attr = prim.CreateAttribute("physics:density", Sdf.ValueTypeNames.Double)
                                if density_attr:
                                    density_attr.Set(self.cfg.folder_mass_density)
                                    print(f"[DEBUG] 创建并设置 {prim.GetPath()} 的密度为 {self.cfg.folder_mass_density} kg/m³")
                
                # 递归查找所有rigid body links
                def find_and_set_mass(prim):
                    """递归查找并设置所有rigid body的质量"""
                    set_link_mass(prim)
                    for child in prim.GetChildren():
                        find_and_set_mass(child)
                
                find_and_set_mass(folder_prim)
            
            if self.cfg.folder_mass is not None:
                print(f"[INFO] 已设置文件夹的质量: {self.cfg.folder_mass} kg")
            elif self.cfg.folder_mass_density is not None:
                print(f"[INFO] 已设置文件夹的密度: {self.cfg.folder_mass_density} kg/m³")
            
        except Exception as e:
            print(f"[WARNING] 设置质量属性时出错: {e}")
            import traceback
            traceback.print_exc()

