from __future__ import annotations

import torch
import math
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_uipc import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from tacex_tasks.utils import DirectLiveVisualizer
from tacex_uipc import (
    TetMeshCfg,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcRLEnv,
    UipcSimCfg,
)


@configclass
class PickUpBallEnvCfg(DirectRLEnvCfg):
    """Config for a tactile pick-up task using two GelSight Mini sensors."""

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.5, 1.0, 0.5)
    viewer.lookat = (0.2, 0.0, 0.1)

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

    mesh_cfg = TetMeshCfg(
        stop_quality=8,
        max_its=100,
        edge_length_r=1 / 5,
    )
    ball = UipcObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.035]),
        spawn=sim_utils.UsdFileCfg(
            scale=(5, 5, 5),
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd",
        ),
        mesh_cfg=mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.0005),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    gelpad_mesh_cfg = TetMeshCfg(stop_quality=8, max_its=100, edge_length_r=1 / 15)
    gelpad_left_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_left",
        mesh_cfg=gelpad_mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )
    gelpad_attachment_left_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=100.0,
        body_name="gelsight_mini_case_left",
    )

    gelpad_right_cfg = UipcObjectCfg(
        prim_path="/World/envs/env_.*/Robot/gelpad_right",
        mesh_cfg=gelpad_mesh_cfg,
        constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(),
    )
    gelpad_attachment_right_cfg = UipcIsaacAttachmentsCfg(
        constraint_strength_ratio=100.0,
        body_name="gelsight_mini_case_right",
    )

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 32),
            # only request depth here; RGB annotator 'camera_rgb' is not available in current replicator build
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,
        marker_motion_sim_cfg=None,
        # only use tactile RGB for RL observations (avoid camera_rgb annotator)
        data_types=["tactile_rgb"],
    )
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

    gsmini_right = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
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
    gsmini_right.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(32, 32),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    episode_length_s = 6.0
    action_space = 8  # 6 DoF IK + 2 finger targets
    observation_space = {
        "proprio_obs": 17,  # ee pos(3)+euler(3)+ball pos(3)+fingers(2)+actions(6)
        # left/right tactile (6 channels) + global RGB camera (3 channels)
        "vision_obs": [32, 32, 9],
    }
    state_space = 0

    # 更保守的平移缩放，避免 IK 命令过于激进导致抖动
    action_scale_translation = 0.05
    action_scale_rotation = 0.35
    finger_action_scale = 0.04
    min_ball_height = 0.003
    success_height = 0.15

    # default scripted EE trajectory heights (used by PickUpBallGripperEnv and scripted policies)
    # 稍微降低 approach / grasp 高度，让末端真正“贴到球附近”再抬起
    approach_height: float = 0.02  # m above ball
    grasp_height: float = -0.01  # m relative to ball center (略低一些，便于夹住软球)
    lift_height: float = 0.18  # target additional height above ball

    # whether to actually read image tensors from sensors (tactile + rgb)
    # if False, we will feed zero images of the right shape. Useful for scripted policies
    # to avoid heavy / fragile Replicator image pipelines.
    use_images: bool = True

    # per-env third-person RGB camera attached to each env root at a fixed offset
    # use a global prim path with env regex, so Camera/Simulation can expand it per env
    # arm_camera: CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/ArmCamera",
    #     update_period=0,
    #     height=32,
    #     width=32,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0,
    #         focus_distance=0.8,
    #         horizontal_aperture=20.0,
    #         clipping_range=(0.1, 2.0),
    #     ),
    # )
    # === 用 Tensor，而不是 float ===
    # euler_xyz = torch.tensor(
    #     [-51.524, 39.897, 169.702], dtype=torch.float32
    # ) * torch.pi / 180.0  # deg -> rad

    # quat = math_utils.quat_from_euler_xyz(
    #     euler_xyz[0], euler_xyz[1], euler_xyz[2]
    # )

    arm_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/ArmCamera",

        offset=OffsetCfg(
            pos=(1.04358, 0.43, 0.47413),

            # ✅ 纯 Python tuple（Hydra 可序列化）
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

class PickUpBallEnv(UipcRLEnv):
    """RL env for picking up a soft ball with two tactile fingers."""

    cfg: PickUpBallEnvCfg

    def __init__(self, cfg: PickUpBallEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.finger_targets = torch.zeros((self.num_envs, 2), device=self.device)

        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        # Debug visualizers in Isaac client (for observations etc.)
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

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.11841)),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        # global RGB camera seeing the arm and table (one per env, spawned by Camera)
        # Note: we rely on CameraCfg.prim_path ("/World/envs/env_.*/ArmCamera") and
        #      Simulation/Camera to expand it per env. No direct omni.usd usage here.
        if not hasattr(self.cfg.arm_camera.offset, "convention"):
        # 等价于旧版默认行为
            self.cfg.arm_camera.offset.convention = "usd"
        self.arm_camera = Camera(self.cfg.arm_camera)
        self.scene.sensors["arm_camera"] = self.arm_camera

        # ground = self.cfg.ground
        # ground.spawn.func(
        #     ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        # )

        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

        self._uipc_gelpad_left = UipcObject(self.cfg.gelpad_left_cfg, self.uipc_sim)
        self._uipc_gelpad_right = UipcObject(self.cfg.gelpad_right_cfg, self.uipc_sim)

        self.attachment_left = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_left_cfg, self._uipc_gelpad_left, self.scene.articulations["robot"]
        )
        self.attachment_right = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_right_cfg, self._uipc_gelpad_right, self.scene.articulations["robot"]
        )

        self.object = UipcObject(self.cfg.ball, self.uipc_sim)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clamp(-1.0, 1.0)
        self.actions[:] = actions
        ee_pos_b, ee_quat_b = self._compute_frame_pose()

        self.processed_actions.zero_()
        trans_scale = self.cfg.action_scale_translation
        rot_scale = self.cfg.action_scale_rotation
        self.processed_actions[:, :3] = actions[:, :3] * trans_scale
        self.processed_actions[:, 3:6] = actions[:, 3:6] * rot_scale
        self._ik_controller.set_command(self.processed_actions, ee_pos_b, ee_quat_b)

        finger_cmd = (actions[:, 6:8] * 0.5 + 0.5) * self.cfg.finger_action_scale
        self.finger_targets[:] = finger_cmd.clamp(0.0, self.cfg.finger_action_scale)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        joint_pos_des[:, self._finger_joint_ids[0]] = self.finger_targets[:, 0]
        joint_pos_des[:, self._finger_joint_ids[1]] = self.finger_targets[:, 1]
        self._robot.set_joint_position_target(joint_pos_des)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ball_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            self.object.data.root_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
        )
        min_height = ball_pos_b[:, 2] < self.cfg.min_ball_height
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reset_cond = min_height | time_out
        return reset_cond, time_out

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_b, _ = self._compute_frame_pose()
        ball_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            self.object.data.root_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
        )

        height_reward = torch.clamp(ball_pos_b[:, 2] - 0.02, min=0.0)
        dist = torch.norm(ee_pos_b - ball_pos_b, dim=1)
        dist_reward = -dist
        success = (ball_pos_b[:, 2] > self.cfg.success_height) & (dist < 0.05)
        success_bonus = success.float() * 5.0

        return 2.0 * height_reward + dist_reward + success_bonus

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self._uipc_gelpad_left.write_vertex_positions_to_sim(vertex_positions=self._uipc_gelpad_left.init_vertex_pos)
        self._uipc_gelpad_right.write_vertex_positions_to_sim(vertex_positions=self._uipc_gelpad_right.init_vertex_pos)
        self.object.write_vertex_positions_to_sim(vertex_positions=self.object.init_vertex_pos)

        self.processed_actions.zero_()
        self.finger_targets.zero_()
        self.actions[env_ids] = 0.0

        # For UIPC deformable objects we reset by restoring vertex positions above.
        # If position randomization is needed later, it should be implemented via UIPC APIs.

    def _get_observations(self) -> dict:
        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        ee_euler = euler_xyz_from_quat(ee_quat_b)
        ex = wrap_to_pi(ee_euler[0]).unsqueeze(1)
        ey = wrap_to_pi(ee_euler[1]).unsqueeze(1)
        ez = wrap_to_pi(ee_euler[2]).unsqueeze(1)

        ball_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_link_pos_w,
            self._robot.data.root_link_quat_w,
            self.object.data.root_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
        )

        joint_pos = self._robot.data.joint_pos[:, :]
        left_finger = joint_pos[:, self._finger_joint_ids[0]].unsqueeze(1)
        right_finger = joint_pos[:, self._finger_joint_ids[1]].unsqueeze(1)

        proprio_obs = torch.cat(
            (ee_pos_b, ex, ey, ez, ball_pos_b, left_finger, right_finger, self.processed_actions),
            dim=-1,
        )

        # stack left/right tactile RGB images and global RGB image:
        # (num_envs, H, W, 9) = 3 (left tactile) + 3 (right tactile) + 3 (global rgb)
        if self.cfg.use_images:
            imgs = []
            for sensor in (self.gsmini_left, self.gsmini_right):
                tactile = sensor.data.output.get("tactile_rgb", None)
                if tactile is not None:
                    tensor = torch.as_tensor(tactile, device=self.device, dtype=torch.float32)
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                    imgs.append(tensor)

            # add global RGB camera image
            rgb = self.arm_camera.data.output.get("rgb", None)
            if rgb is not None:
                rgb_tensor = torch.as_tensor(rgb, device=self.device, dtype=torch.float32)
                if rgb_tensor.max() > 1.0:
                    rgb_tensor = rgb_tensor / 255.0
                imgs.append(rgb_tensor)

            if len(imgs) == 0:
                vision_obs = torch.zeros((self.num_envs, 32, 32, 9), device=self.device)
            else:
                # ensure three images so that channel count matches 9
                while len(imgs) < 3:
                    imgs.append(imgs[-1])
                vision_obs = torch.cat(imgs, dim=-1)
        else:
            # scripted policies / lightweight runs: just feed zeros with correct shape
            vision_obs = torch.zeros((self.num_envs, 32, 32, 9), device=self.device)

        obs = {"proprio_obs": proprio_obs, "vision_obs": vision_obs}

        # push vision observations into Isaac client's debug visualizer
        if self.cfg.debug_vis:
            self.visualizers["Observations"].terms["sensor_output_tactile_left"] = imgs[0]
            self.visualizers["Observations"].terms["sensor_output_tactile_right"] = imgs[1]
            self.visualizers["Observations"].terms["sensor_output_rgb"] = imgs[2]

        return {"policy": obs}

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot)
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


@configclass
class PickUpBallGripperEnvCfg(PickUpBallEnvCfg):
    """Variant where end-effector trajectory is scripted and RL only controls gripper."""

    # only 2 actions: left/right finger targets
    action_space = 2


class PickUpBallGripperEnv(PickUpBallEnv):
    """Env: scripted end-effector motion, RL controls gripper only."""

    cfg: PickUpBallGripperEnvCfg

    def __init__(self, cfg: PickUpBallGripperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Scripted loop for EE:

        - 初始阶段：末端对齐到小球上方（x,y 和球一致，z = ball_z + grasp_height），夹爪完全张开并停留一段时间；
        - 然后：沿 z 轴向上抬到 ball_z + lift_height；

        RL 仅在“抬起阶段”控制两根手指的开合（2 维动作）。
        """
        # 当前 EE 位姿（base 坐标系）
        ee_pos_b, ee_quat_b = self._compute_frame_pose()

        # 球在 base 坐标系中的位置
        ball_pos_w = self.object.data.root_pos_w
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ball_pos_b, _ = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            ball_pos_w,
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
        )

        # 归一化时间 [0, 1]
        frac = (self.episode_length_buf.to(torch.float32) / max(self.max_episode_length - 1, 1)).unsqueeze(1)

        # 目标高度：先在抓取高度处停留，再抬到 lift_height
        z_hold = ball_pos_b[:, 2:3] + self.cfg.grasp_height
        z_lift = ball_pos_b[:, 2:3] + self.cfg.lift_height
        # 前 40% 时间停在 z_hold，后 60% 抬到 z_lift
        z = torch.where(frac <= 0.4, z_hold, z_lift)

        # 目标 EE 位置：x,y 始终对齐小球，z 按上述策略
        target_pos_b = ee_pos_b.clone()
        target_pos_b[:, 0:2] = ball_pos_b[:, 0:2]
        target_pos_b[:, 2:3] = z

        # IK 命令：用一个温和的比例系数，避免抖动
        delta_pos = target_pos_b - ee_pos_b
        trans_scale = max(self.cfg.action_scale_translation, 1e-6)
        self.processed_actions.zero_()
        # 更温和的速度系数，进一步减小每步位移，避免末端“抢跑”
        self.processed_actions[:, :3] = torch.clamp(0.1 * delta_pos / trans_scale, -1.0, 1.0)
        self.processed_actions[:, 3:6] = 0.0
        self._ik_controller.set_command(self.processed_actions, ee_pos_b, ee_quat_b)

        # 夹爪：全程由 RL 控制两根手指的开合
        actions = actions.clamp(-1.0, 1.0)
        fingers = (actions * 0.5 + 0.5) * self.cfg.finger_action_scale
        fingers = fingers.clamp(0.0, self.cfg.finger_action_scale)

        self.finger_targets[:, 0] = fingers[:, 0]
        self.finger_targets[:, 1] = fingers[:, 1]


