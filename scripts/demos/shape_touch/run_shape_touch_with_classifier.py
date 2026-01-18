"""Shape touching demo with tactile classifier inference.

This script extends run_shape_touch.py to include real-time tactile shape classification
using the trained attention-based classifier. Users can touch different shapes and see
the classifier's predictions in real-time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Control Franka with GelSight Mini Sensor and classify shapes using tactile data"
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
    "--classifier_path",
    type=str,
    default=None,
    help="Path to trained classifier model checkpoint (.pth file). If not provided, uses random initialization.",
)
parser.add_argument(
    "--inference_interval",
    type=int,
    default=10,
    help="Steps between classifier inference calls.",
)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
from contextlib import suppress
from collections import deque

import numpy as np
import torch
import omni.ui

import carb
import pynvml
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_single_rigid import (
    FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

# Import tactile classifier
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from tactile_classifier import AttentionPoolingClassifier

with suppress(ImportError):
    import isaacsim.gui.components.ui_utils as ui_utils


class ClassifierEnvWindow(BaseEnvWindow):
    """Window manager with classifier display."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)

        self.object_names = list(create_shapes_cfg().keys())
        self.current_object_name = self.object_names[0]

        # Classification statistics
        self.prediction_history = deque(maxlen=100)
        self.correct_predictions = 0
        self.total_predictions = 0

        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)

        with self.ui_window_elements["main_vstack"]:
            self._build_control_frame()
            self._build_classifier_frame()
            self.ui_window_elements["debug_frame"].collapsed = True
            self.ui_window_elements["sim_frame"].collapsed = True

    def _build_control_frame(self):
        """Build control frame for shape selection."""
        self.ui_window_elements["action_frame"] = omni.ui.CollapsableFrame(
            title="Shape Touching Demo",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["action_frame"]:
            self.ui_window_elements["action_vstack"] = omni.ui.VStack(spacing=5, height=50)
            with self.ui_window_elements["action_vstack"]:
                objects_dropdown_cfg = {
                    "label": "Objects",
                    "type": "dropdown",
                    "default_val": 0,
                    "items": self.object_names,
                    "tooltip": "Select a shape to touch",
                    "on_clicked_fn": self._set_main_obj,
                }
                self.ui_window_elements["object_dropdown"] = ui_utils.dropdown_builder(**objects_dropdown_cfg)

                self.ui_window_elements["reset_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Reset Env",
                    tooltip="Resets the environment",
                    on_clicked_fn=self._reset_env,
                )

    def _build_classifier_frame(self):
        """Build classifier display frame."""
        if ui_utils is None:
            return
        
        import omni.ui
        
        self.ui_window_elements["classifier_frame"] = omni.ui.CollapsableFrame(
            title="Tactile Classifier",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
        )
        with self.ui_window_elements["classifier_frame"]:
            self.ui_window_elements["classifier_vstack"] = omni.ui.VStack(spacing=5, height=50)
            with self.ui_window_elements["classifier_vstack"]:
                # Prediction display
                self.ui_window_elements["prediction_label"] = omni.ui.Label(
                    "Prediction: --",
                    style={"color": omni.ui.color.white},
                )
                
                # Confidence display
                self.ui_window_elements["confidence_label"] = omni.ui.Label(
                    "Confidence: --",
                    style={"color": omni.ui.color.white},
                )
                
                # Accuracy display
                self.ui_window_elements["accuracy_label"] = omni.ui.Label(
                    "Accuracy: --",
                    style={"color": omni.ui.color.white},
                )
                
                # Reset stats button
                self.ui_window_elements["reset_stats_button"] = ui_utils.btn_builder(
                    type="button",
                    text="Reset Statistics",
                    tooltip="Reset classification statistics",
                    on_clicked_fn=self._reset_stats,
                )

    def get_current_object_name(self):
        """Get currently selected object name."""
        current_obj_idx = (
            self.ui_window_elements["object_dropdown"].get_item_value_model().get_value_as_int()
        )
        self.current_object_name = self.object_names[current_obj_idx]
        return self.current_object_name

    def _set_main_obj(self, value):
        """Set main object to touch."""
        print("Set new main obj ", value)
        old_obj: RigidObject = self.env.scene.rigid_objects[self.current_object_name]

        obj_name = self.get_current_object_name()
        new_obj: RigidObject = self.env.scene.rigid_objects[obj_name]
        new_pose = new_obj.data.root_pose_w

        old_obj.write_root_pose_to_sim(new_pose)
        new_obj.write_root_pose_to_sim(self.env.main_pose)

    def _reset_env(self):
        """Reset environment."""
        self.env.reset()

    def _reset_stats(self):
        """Reset classification statistics."""
        self.prediction_history.clear()
        self.correct_predictions = 0
        self.total_predictions = 0
        if ui_utils is not None and "accuracy_label" in self.ui_window_elements:
            self.ui_window_elements["accuracy_label"].text = "Accuracy: --"
            self.ui_window_elements["prediction_label"].text = "Prediction: --"
            self.ui_window_elements["confidence_label"].text = "Confidence: --"
        print("[INFO] Statistics reset")

    def update_classifier_display(self, pred_name: str, confidence: float, is_correct: bool):
        """Update classifier display in GUI."""
        if ui_utils is None or "prediction_label" not in self.ui_window_elements:
            return
        
        # Update labels
        color = omni.ui.color(0.0, 1.0, 0.0) if is_correct else omni.ui.color(1.0, 0.0, 0.0)
        self.ui_window_elements["prediction_label"].text = f"Prediction: {pred_name}"
        self.ui_window_elements["prediction_label"].style = {"color": color}
        
        self.ui_window_elements["confidence_label"].text = f"Confidence: {confidence:.3f}"
        
        if self.total_predictions > 0:
            accuracy = 100.0 * self.correct_predictions / self.total_predictions
            self.ui_window_elements["accuracy_label"].text = f"Accuracy: {accuracy:.1f}% ({self.correct_predictions}/{self.total_predictions})"


def create_shapes_cfg() -> dict[str, RigidObjectCfg]:
    """Creates RigidObjectCfg's for each usd file in the shapes directory."""
    shapes = {}
    usd_files_path = list(Path(f"{TACEX_ASSETS_DATA_DIR}/Props/tactile_test_shapes/").glob("*.usd"))

    for i, file_path in enumerate(usd_files_path):
        file_name = file_path.stem

        shapes[file_name] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{file_name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.025 * i, 0.02]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(file_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                    disable_gravity=False,
                ),
            ),
        )

    return shapes


@configclass
class BallRollingEnvCfg(DirectRLEnvCfg):
    """Configuration for shape touching environment."""

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.55, -0.06, 0.025)
    viewer.lookat = (-4.8, 6.0, -0.2)

    debug_vis = True
    ui_window_class_type = ClassifierEnvWindow

    decimation = 1
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
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
        num_envs=1,
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

    robot: ArticulationCfg = FRANKA_PANDA_ARM_SINGLE_GSMINI_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    shapes: dict = create_shapes_cfg()

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    gsmini = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(320, 240),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,
        marker_motion_sim_cfg=FOTSMarkerSimulatorCfg(
            lamb=[0.00125, 0.00021, 0.00038],
            pyramid_kernel_size=[51, 21, 11, 5],
            kernel_size=5,
            marker_params=FOTSMarkerSimulatorCfg.MarkerParams(
                num_markers_col=11,
                num_markers_row=9,
                num_markers=99,
                x0=15,
                y0=26,
                dx=26,
                dy=29,
            ),
            tactile_img_res=(320, 240),
            device="cuda",
            frame_transformer_cfg=FrameTransformerCfg(
                prim_path="/World/envs/env_.*/Robot/gelsight_mini_gelpad",
                source_frame_offset=OffsetCfg(),
                target_frames=[
                    FrameTransformerCfg.FrameCfg(prim_path=f"/World/envs/env_.*/{obj_name}")
                    for obj_name in list(shapes.keys())
                ],
                debug_vis=True,
                visualizer_cfg=marker_cfg,
            ),
        ),
        data_types=["marker_motion", "tactile_rgb"],
    )

    gsmini.optical_sim_cfg = gsmini.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(320, 240),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    main_pose = [0.5, 0.0, 0.02, 1, 0, 0, 0]

    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class BallRollingEnv(DirectRLEnv):
    """Shape touching environment with classifier."""

    cfg: BallRollingEnvCfg

    def __init__(self, cfg: BallRollingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # IK controller setup
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        self._offset_pos = torch.tensor([0.0, 0.0, 0.131], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.step_count = 0
        self.goal_prim_view = None
        self.main_pose = torch.tensor([self.cfg.main_pose], device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        for obj_name, cfg in self.cfg.shapes.items():
            self.scene.rigid_objects[obj_name] = RigidObject(cfg)

        self.object = list(self.scene.rigid_objects.values())[0]

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
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.131),
                    ),
                ),
            ],
        )

        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini = GelSightSensor(self.cfg.gsmini)
        self.scene.sensors["gsmini"] = self.gsmini

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path,
            ground.spawn,
            translation=ground.init_state.pos,
            orientation=ground.init_state.rot,
        )

        VisualCuboid(
            prim_path="/Goal",
            size=0.01,
            position=np.array([0.5, 0.0, 0.021]),
            orientation=np.array([0, 1, 0, 0]),
            visible=False,
        )

        VisualCuboid(
            prim_path="/Visuals/main_area",
            size=0.02,
            position=np.array([0.5, 0.0, -0.005]),
            color=np.array([255.0, 0.0, 0.0]),
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._ik_controller.set_command(self.ik_commands)

    def _apply_action(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._robot.set_joint_position_target(joint_pos_des)

        self.step_count += 1

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def _get_rewards(self) -> torch.Tensor:
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if self.goal_prim_view is not None:
            goal_pos = self.main_pose[:, :3]
            goal_pos[:, 2] += 0.001

            goal_orient = torch.tensor([[0, 1, 0, 0]], device=self.device)
            self.goal_prim_view.set_world_poses(positions=goal_pos, orientations=goal_orient)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _get_observations(self) -> dict:
        pass

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
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


def load_classifier(shape_names: list[str], device: torch.device) -> AttentionPoolingClassifier:
    """Load tactile classifier model.
    
    Args:
        shape_names: List of shape names (determines num_classes).
        device: Device to load model on.
        
    Returns:
        Loaded classifier model.
        
    Note:
        Model accepts original sensor resolution (240x320) without resizing.
    """
    num_classes = len(shape_names)
    
    # Create model (supports variable input resolutions, designed for 240x320)
    model = AttentionPoolingClassifier(
        num_classes=num_classes,
        image_channels=3,
        hidden_channels=[32, 64, 128, 256],
        mlp_dims=[512, 256, 128],
    ).to(device)
    
    # Load checkpoint if provided
    if args_cli.classifier_path is not None:
        checkpoint_path = Path(args_cli.classifier_path)
        if checkpoint_path.exists():
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.eval()
                print(f"[INFO] Loaded classifier from: {checkpoint_path}")
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                print("[INFO] Using random initialization")
        else:
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            print("[INFO] Using random initialization")
    else:
        print("[INFO] No checkpoint provided, using random initialization")
    
    model.eval()
    return model


def run_simulator(env: BallRollingEnv):
    """Run simulation loop with classifier inference."""
    # Enable debug visualization
    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    print(f"Starting simulation with {env.num_envs} envs")

    env.reset()
    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)

    # Load classifier
    shape_names = list(env.cfg.shapes.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = load_classifier(shape_names, device)
    
    print(f"[INFO] Classifier loaded with {len(shape_names)} classes: {shape_names}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Inference interval: {args_cli.inference_interval} steps")
    print(f"[INFO] Input resolution: Original sensor resolution (240x320)")
    print("\n" + "="*70)
    print("Classification started. Move the Goal marker to control robot.")
    print("="*70 + "\n")

    step_count = 0
    last_prediction_step = 0

    # Simulation loop
    while simulation_app.is_running():
        # Physics step
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)

        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations

        # Update scene
        env.scene.update(dt=env.physics_dt)
        env.sim.render()

        # Classifier inference
        if step_count % args_cli.inference_interval == 0:
            tactile_rgb = env.gsmini.data.output.get("tactile_rgb", None)

            if tactile_rgb is not None:
                # Convert to tensor and normalize
                tactile_tensor = torch.as_tensor(tactile_rgb, device=device, dtype=torch.float32)
                if tactile_tensor.max() > 1.0:
                    tactile_tensor = tactile_tensor / 255.0
                
                # Reshape: (1, H, W, 3) -> (1, 3, H, W)
                tactile_tensor = tactile_tensor.permute(0, 3, 1, 2)
                
                # Use original resolution (240x320) - no resizing needed
                # The AttentionPoolingClassifier supports variable input resolutions
                
                # Inference
                pred_classes, confidence, probs, attention_map = classifier.infer(tactile_tensor)
                
                # Get true label
                current_shape_name = env._window.get_current_object_name()
                true_label_idx = shape_names.index(current_shape_name)
                
                # Get prediction
                pred_label_idx = pred_classes[0].item()
                pred_shape_name = shape_names[pred_label_idx]
                conf = confidence[0].item()
                
                # Update statistics
                is_correct = pred_label_idx == true_label_idx
                env._window.total_predictions += 1
                if is_correct:
                    env._window.correct_predictions += 1
                
                env._window.prediction_history.append({
                    'step': step_count,
                    'true': current_shape_name,
                    'pred': pred_shape_name,
                    'confidence': conf,
                    'correct': is_correct
                })
                
                # Update GUI display
                env._window.update_classifier_display(pred_shape_name, conf, is_correct)
                
                # Terminal output
                status = "✓" if is_correct else "✗"
                accuracy = 100.0 * env._window.correct_predictions / env._window.total_predictions
                
                print(f"[Step {step_count:5d}] {status} "
                      f"True: {current_shape_name:15s} | "
                      f"Pred: {pred_shape_name:15s} | "
                      f"Conf: {conf:.3f} | "
                      f"Acc: {accuracy:.1f}% ({env._window.correct_predictions}/{env._window.total_predictions})")

        step_count += 1

    # Print final statistics
    print("\n" + "="*70)
    print("Final Classification Statistics:")
    print("="*70)
    print(f"Total predictions: {env._window.total_predictions}")
    print(f"Correct predictions: {env._window.correct_predictions}")
    if env._window.total_predictions > 0:
        final_accuracy = 100.0 * env._window.correct_predictions / env._window.total_predictions
        print(f"Final accuracy: {final_accuracy:.2f}%")
    print("="*70)

    env.close()
    pynvml.nvmlShutdown()


def main():
    """Main function."""
    env_cfg = BallRollingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis

    experiment = BallRollingEnv(env_cfg)

    print("[INFO]: Setup complete...")
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
