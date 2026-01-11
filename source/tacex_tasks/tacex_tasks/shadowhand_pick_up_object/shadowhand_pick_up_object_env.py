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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.035)),
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

