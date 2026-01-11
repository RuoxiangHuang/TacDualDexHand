from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.tacdexhand.tacdexhand_gelsighthand import TACDEXHAND_5FINGERS_GELSIGHTHAND_RIGID_CFG
from tacex_assets.sensors.gelsight_hand.gelsighthand_cfg import GelSightHandCfg
from tacex_tasks.utils import DirectLiveVisualizer
from tacex_uipc import (
    TetMeshCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)


def create_sensor_cfg(finger_id: str) -> GelSightHandCfg:
    """Create GelSight Hand sensor configuration for a specific finger."""
    marker_cfg = None  # Disable marker motion for RL training to reduce computation
    sensor_cfg = GelSightHandCfg(
        prim_path=f"/World/envs/env_.*/Robot/fingercase3_{finger_id}",
        sensor_camera_cfg=GelSightHandCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),  # Lower resolution for RL training
            data_types=["depth"],
            clipping_range=(0.017, 0.024),
        ),
        device="cuda",
        debug_vis=False,
        marker_motion_sim_cfg=marker_cfg,  # Disable marker motion for faster training
        data_types=["tactile_rgb"],  # Only use tactile RGB for observations
    )
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),  # Lower resolution for RL training
    )
    return sensor_cfg


@configclass
class TacDexHandPickUpBallEnvCfg(DirectRLEnvCfg):
    """Config for a tactile pick-up task using TacDexHand with 5 GelSight Hand sensors."""

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.5, 0.5, 0.5)
    viewer.lookat = (0.0, 0.0, 0.1)

    debug_vis = True
    ui_window_class_type = BaseEnvWindow

    decimation = 1
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    uipc_sim = UipcSimCfg(
        logger_level="Error",  # reduce UIPC meshing / simulation console output
        ground_height=0.0025,
        contact=UipcSimCfg.Contact(d_hat=0.0001),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, 0)),
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

    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1 / 5,
    )
    ball = UipcObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0.05]),
        spawn=sim_utils.UsdFileCfg(
            scale=(7, 7, 7),
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
        ),
        mesh_cfg=mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.0005),
    )

    # TacDexHand robot configuration with 5 GelSight Hand sensors
    robot: ArticulationCfg = TACDEXHAND_5FINGERS_GELSIGHTHAND_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # Create 5 sensor configurations (one for each finger)
    gelsighthands: dict[str, GelSightHandCfg] = {
        f"finger_{i:02d}": create_sensor_cfg(f"{i:02d}") for i in range(5)
    }

    arm_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/ArmCamera",
        offset=OffsetCfg(
            pos=(0.5, 0.5, 0.5),
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

    episode_length_s = 6.0
    # Note: Adjust action_space based on actual TacDexHand joint count
    # This will be set dynamically in __init__ based on robot DOF
    action_space = 0  # Will be set to controllable joint count in __init__
    observation_space = {
        "proprio_obs": 0,  # Will be set to joint_pos + ball_pos(3) in __init__
        # 5 fingers tactile (15 channels) + global RGB camera (3 channels) = 18 channels
        "vision_obs": [32, 32, 18],
    }
    state_space = 0

    # Joint names pattern to control (use regex pattern, e.g., ".*" for all joints)
    # If None, all joints will be controlled
    controllable_joint_names_expr: list[str] | None = None
    """Joint names pattern to control. If None, all joints will be controlled.
    
    Example: [".*"] for all joints, or ["finger.*"] for only finger joints.
    This allows filtering out fixed joints or sensor-related joints that shouldn't be controlled.
    """

    # Joint position action scaling
    joint_action_scale = 0.1  # Scale for joint position actions
    min_ball_height = 0.003
    success_height = 0.15

    use_images: bool = True


class TacDexHandPickUpBallEnv(UipcRLEnv):
    """RL env for picking up a soft ball with TacDexHand and 5 GelSight Hand sensors."""

    cfg: TacDexHandPickUpBallEnvCfg

    def __init__(self, cfg: TacDexHandPickUpBallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find controllable joints based on joint name pattern
        if self.cfg.controllable_joint_names_expr is not None:
            # Find joints matching the pattern
            self._controllable_joint_ids, self._controllable_joint_names = self._robot.find_joints(
                self.cfg.controllable_joint_names_expr
            )
            print(f"[INFO] Found {len(self._controllable_joint_ids)} controllable joints: {self._controllable_joint_names}")
        else:
            # Control all joints
            self._controllable_joint_ids = list(range(self._robot.num_joints))
            self._controllable_joint_names = self._robot.joint_names
            print(f"[INFO] Controlling all {len(self._controllable_joint_ids)} joints: {self._controllable_joint_names}")

        # Convert to tensor for indexing
        self._controllable_joint_ids = torch.tensor(self._controllable_joint_ids, device=self.device, dtype=torch.long)

        # Get joint limits for action scaling (only for controllable joints)
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._controllable_joint_ids, 0].to(
            device=self.device
        )
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, self._controllable_joint_ids, 1].to(
            device=self.device
        )
        self.robot_dof_default = self._robot.data.default_joint_pos[0, self._controllable_joint_ids].to(
            device=self.device
        )

        # Update action and observation spaces based on controllable joints
        num_controllable_joints = len(self._controllable_joint_ids)
        self.cfg.action_space = num_controllable_joints
        # Observation includes all joints (for proprioception) + ball position
        self.cfg.observation_space["proprio_obs"] = self._robot.num_joints + 3  # all joints + ball_pos

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

        # Setup 5 GelSight Hand sensors
        self.gelsighthands = {}
        for finger_id, sensor_cfg in self.cfg.gelsighthands.items():
            sensor = GelSightSensor(sensor_cfg)
            self.scene.sensors[f"gelsighthand_{finger_id}"] = sensor
            self.gelsighthands[finger_id] = sensor

        # Global RGB camera
        if not hasattr(self.cfg.arm_camera.offset, "convention"):
            self.cfg.arm_camera.offset.convention = "usd"
        self.arm_camera = Camera(self.cfg.arm_camera)
        self.scene.sensors["arm_camera"] = self.arm_camera

        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

        self.object = UipcObject(self.cfg.ball, self.uipc_sim)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions: scale and clamp joint positions."""
        actions = actions.clamp(-1.0, 1.0)
        self.actions[:] = actions

    def _apply_action(self):
        """Apply joint position actions directly to the robot."""
        # Scale actions from [-1, 1] to joint position deltas (only for controllable joints)
        joint_deltas = self.actions[:, :] * self.cfg.joint_action_scale

        # Get current joint positions for all joints
        current_joint_pos = self._robot.data.joint_pos[:, :]

        # Compute target joint positions (only update controllable joints)
        target_joint_pos = current_joint_pos.clone()
        # Apply deltas only to controllable joints
        target_joint_pos[:, self._controllable_joint_ids] = (
            current_joint_pos[:, self._controllable_joint_ids] + joint_deltas
        )

        # Clamp controllable joints to joint limits
        target_joint_pos[:, self._controllable_joint_ids] = torch.clamp(
            target_joint_pos[:, self._controllable_joint_ids],
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
        )

        # Set joint position targets for all joints
        self._robot.set_joint_position_target(target_joint_pos)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episode is done."""
        # Get ball position relative to robot base
        # Expand quaternion and position to match num_envs
        ball_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        ball_pos_w = self.object.data.root_pos_w.expand(self.num_envs, -1)
        ball_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            ball_pos_w,
            ball_quat_w,
        )
        min_height = ball_pos_b[:, 2] < self.cfg.min_ball_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reset_cond = min_height | time_out
        return reset_cond, time_out

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Get ball position relative to robot base
        # Expand quaternion and position to match num_envs
        ball_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        ball_pos_w = self.object.data.root_pos_w.expand(self.num_envs, -1)
        ball_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            ball_pos_w,
            ball_quat_w,
        )

        # Height reward: encourage lifting the ball
        height_reward = torch.clamp(ball_pos_b[:, 2] - 0.02, min=0.0)

        # Distance reward: encourage getting close to ball center (x, y)
        dist_xy = torch.norm(ball_pos_b[:, :2], dim=1)
        dist_reward = -dist_xy

        # Success bonus: ball lifted high enough
        success = ball_pos_b[:, 2] > self.cfg.success_height
        success_bonus = success.float() * 5.0

        return 2.0 * height_reward + dist_reward + success_bonus

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environment."""
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset UIPC object - only reset when all environments are being reset
        # since write_vertex_positions_to_sim doesn't support per-environment resets
        # Ensure tensor is contiguous before passing to avoid CUDA errors
        if env_ids is None or (isinstance(env_ids, torch.Tensor) and len(env_ids) == self.num_envs):
            # Ensure the tensor is contiguous before conversion to numpy
            vertex_pos = self.object.init_vertex_pos.contiguous()
            self.object.write_vertex_positions_to_sim(vertex_positions=vertex_pos)

        self.actions[env_ids] = 0.0

    def _get_observations(self) -> dict:
        """Get observations."""
        # Joint positions
        joint_pos = self._robot.data.joint_pos[:, :]

        # Ball position relative to robot base
        # Expand quaternion and position to match num_envs
        ball_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        ball_pos_w = self.object.data.root_pos_w.expand(self.num_envs, -1)
        ball_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            ball_pos_w,
            ball_quat_w,
        )

        # Proprioceptive observations: joint positions + ball position
        proprio_obs = torch.cat((joint_pos, ball_pos_b), dim=-1)

        # Vision observations: 5 tactile images + RGB camera
        if self.cfg.use_images:
            imgs = []
            # Add tactile images from all 5 sensors
            for finger_id in sorted(self.gelsighthands.keys()):
                sensor = self.gelsighthands[finger_id]
                tactile = sensor.data.output.get("tactile_rgb", None)
                if tactile is not None:
                    tensor = torch.as_tensor(tactile, device=self.device, dtype=torch.float32)
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                    imgs.append(tensor)
                else:
                    # If sensor data is not available, add zero tensor
                    imgs.append(torch.zeros((self.num_envs, 32, 32, 3), device=self.device))

            # Add global RGB camera image
            rgb = self.arm_camera.data.output.get("rgb", None)
            if rgb is not None:
                rgb_tensor = torch.as_tensor(rgb, device=self.device, dtype=torch.float32)
                if rgb_tensor.max() > 1.0:
                    rgb_tensor = rgb_tensor / 255.0
                imgs.append(rgb_tensor)
            else:
                imgs.append(torch.zeros((self.num_envs, 32, 32, 3), device=self.device))

            # Concatenate all images: 5 tactile (15 channels) + 1 RGB (3 channels) = 18 channels
            vision_obs = torch.cat(imgs, dim=-1)
        else:
            vision_obs = torch.zeros((self.num_envs, 32, 32, 18), device=self.device)
            imgs = [torch.zeros((self.num_envs, 32, 32, 3), device=self.device) for _ in range(6)]

        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}

        # Debug visualization
        if self.cfg.debug_vis and len(imgs) >= 6:
            for i, finger_id in enumerate(sorted(self.gelsighthands.keys())):
                if i < len(imgs) - 1:  # Last image is RGB
                    self.visualizers["Observations"].terms[f"sensor_output_tactile_{finger_id}"] = imgs[i]
            self.visualizers["Observations"].terms["sensor_output_rgb"] = imgs[-1]

        return {"policy": obs}

