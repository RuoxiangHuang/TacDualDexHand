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
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.allegro_gsmini import ALLEGRO_HAND_GSMINI_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from tacex_tasks.utils import DirectLiveVisualizer
from tacex_uipc import (
    TetMeshCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)


@configclass
class AllegroPickUpBallEnvCfg(DirectRLEnvCfg):
    """Config for a tactile pick-up task using Allegro Hand with GelSight Mini sensors."""

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

    # Override Allegro Hand initial pose only for this task
    robot: ArticulationCfg = ALLEGRO_HAND_GSMINI_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # world position of the hand base
            pos=(-0.10579, 0.00103, 0.12746),
            rot=(0.2706, 0.65328, 0.2706, 0.65328),
            # keep joint init from the base cfg (optional: you can also copy and tweak ALLEGRO_HAND_GSMINI_CFG.init_state.joint_pos)
            joint_pos=ALLEGRO_HAND_GSMINI_CFG.init_state.joint_pos,
        ),
    )

    # Sensor configuration based on Allegro Hand USD structure
    # Sensors are named Case_thumb_3, Case_index_3, etc. in the USD file
    gsmini_thumb = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/Case_thumb_3",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
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
    gsmini_thumb.optical_sim_cfg = gsmini_thumb.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

    gsmini_index = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/Case_index_3",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
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
    gsmini_index.optical_sim_cfg = gsmini_index.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

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
    action_space = 16  # 16 joints for Allegro Hand (4 fingers × 4 joints)
    observation_space = {
        "proprio_obs": 19,  # joint_pos(16) + ball_pos(3)
        # thumb/index tactile (6 channels) + global RGB camera (3 channels)
        "vision_obs": [32, 32, 9],
    }
    state_space = 0

    # Joint position action scaling
    joint_action_scale = 0.1  # Scale for joint position actions
    min_ball_height = 0.003
    success_height = 0.15

    use_images: bool = True


class AllegroPickUpBallEnv(UipcRLEnv):
    """RL env for picking up a soft ball with Allegro Hand tactile sensors."""

    cfg: AllegroPickUpBallEnvCfg

    def __init__(self, cfg: AllegroPickUpBallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint limits for action scaling
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_default = self._robot.data.default_joint_pos[0, :].to(device=self.device)

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

        # Setup GelSight sensors
        self.gsmini_thumb = GelSightSensor(self.cfg.gsmini_thumb)
        self.scene.sensors["gsmini_thumb"] = self.gsmini_thumb

        self.gsmini_index = GelSightSensor(self.cfg.gsmini_index)
        self.scene.sensors["gsmini_index"] = self.gsmini_index

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
        # Scale actions from [-1, 1] to joint position deltas
        joint_deltas = self.actions[:, :] * self.cfg.joint_action_scale

        # Get current joint positions
        current_joint_pos = self._robot.data.joint_pos[:, :]

        # Compute target joint positions
        target_joint_pos = current_joint_pos + joint_deltas

        # Clamp to joint limits
        target_joint_pos = torch.clamp(
            target_joint_pos,
            self.robot_dof_lower_limits.unsqueeze(0),
            self.robot_dof_upper_limits.unsqueeze(0),
        )

        # Set joint position targets
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
        # Joint positions (16 joints)
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

        # Vision observations: tactile images + RGB camera
        if self.cfg.use_images:
            imgs = []
            # Add tactile images from sensors
            for sensor in [self.gsmini_thumb, self.gsmini_index]:
                tactile = sensor.data.output.get("tactile_rgb", None)
                if tactile is not None:
                    tensor = torch.as_tensor(tactile, device=self.device, dtype=torch.float32)
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                    imgs.append(tensor)

            # Add global RGB camera image
            rgb = self.arm_camera.data.output.get("rgb", None)
            if rgb is not None:
                rgb_tensor = torch.as_tensor(rgb, device=self.device, dtype=torch.float32)
                if rgb_tensor.max() > 1.0:
                    rgb_tensor = rgb_tensor / 255.0
                imgs.append(rgb_tensor)

            # Ensure we have 3 images (2 tactile + 1 RGB = 9 channels)
            while len(imgs) < 3:
                if len(imgs) > 0:
                    imgs.append(imgs[-1])
                else:
                    imgs.append(torch.zeros((self.num_envs, 32, 32, 3), device=self.device))

            vision_obs = torch.cat(imgs, dim=-1)
        else:
            vision_obs = torch.zeros((self.num_envs, 32, 32, 9), device=self.device)
            imgs = [torch.zeros((self.num_envs, 32, 32, 3), device=self.device) for _ in range(3)]

        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}

        # Debug visualization
        if self.cfg.debug_vis and len(imgs) >= 3:
            self.visualizers["Observations"].terms["sensor_output_tactile_thumb"] = imgs[0]
            self.visualizers["Observations"].terms["sensor_output_tactile_index"] = imgs[1]
            self.visualizers["Observations"].terms["sensor_output_rgb"] = imgs[2]

        return {"policy": obs}

