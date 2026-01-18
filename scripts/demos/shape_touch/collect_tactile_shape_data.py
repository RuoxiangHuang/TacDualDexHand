"""Collect tactile data from shape touching for training tactile classifiers.

This script collects tactile images from GelSight sensor while touching different shapes,
automatically labeling data with shape class names for supervised learning.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect tactile shape data for training classifiers"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--output_dir", type=str, default="./data/tactile_shapes", help="Output directory for collected data.")
parser.add_argument("--samples_per_shape", type=int, default=100, help="Number of samples to collect per shape.")
parser.add_argument("--sample_interval", type=int, default=5, help="Steps between samples.")
parser.add_argument("--auto_mode", action="store_true", help="Enable automatic data collection mode.")
parser.add_argument("--randomize_pose", action="store_true", help="Randomize end-effector pose for diversity.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the GUI",
)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import traceback
from contextlib import suppress

import numpy as np
import torch
import zarr

import carb
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

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

with suppress(ImportError):
    import isaacsim.gui.components.ui_utils as ui_utils


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
class TactileDataCollectionEnvCfg(DirectRLEnvCfg):
    """Configuration for tactile data collection environment."""

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.55, -0.06, 0.025)
    viewer.lookat = (-4.8, 6.0, -0.2)

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

    # Main pose: object position on ground (Z=0.02m for object center)
    # End-effector will move DOWN from this position to contact
    main_pose = [0.5, 0.0, 0.02, 1, 0, 0, 0]

    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0


class TactileDataCollectionEnv(DirectRLEnv):
    """Environment for collecting tactile shape data."""

    cfg: TactileDataCollectionEnvCfg

    def __init__(self, cfg: TactileDataCollectionEnvCfg, render_mode: str | None = None, **kwargs):
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
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.131)),
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
            visible=True,
            color=np.array([0.0, 255.0, 0.0]),
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

    def set_target_shape(self, shape_name: str):
        """Set the target shape to touch."""
        if shape_name not in self.scene.rigid_objects:
            raise ValueError(f"Shape {shape_name} not found in scene")
        
        target_obj = self.scene.rigid_objects[shape_name]
        
        # Move target object to main area
        main_pos = self.main_pose[0, :3]
        main_quat = self.main_pose[0, 3:]
        main_pose_7d = torch.cat([main_pos, main_quat], dim=0).unsqueeze(0)  # (1, 7)
        
        # Use torch.Tensor for env_ids (Isaac Lab requirement)
        env_ids_tensor = torch.tensor([0], dtype=torch.int32, device=self.device)
        target_obj.write_root_pose_to_sim(main_pose_7d, env_ids=env_ids_tensor)

    def randomize_goal_pose(self, base_pos: np.ndarray, base_quat: np.ndarray):
        """Randomize goal pose around base position for data diversity."""
        if not args_cli.randomize_pose:
            # Even without randomization, move DOWN to contact the object
            contact_pos = base_pos.copy()
            contact_pos[2] -= 0.005  # Move 5mm down to ensure contact
            return contact_pos, base_quat
        
        # Add small random offset for XY plane
        xy_offset = np.random.uniform(-0.015, 0.015, size=2)
        
        # Z offset should be DOWNWARD to contact the object surface
        # Random contact depth: -3mm to -8mm (negative = downward, ensuring contact)
        z_offset = np.random.uniform(-0.008, -0.003)
        
        pos_offset = np.array([xy_offset[0], xy_offset[1], z_offset])
        randomized_pos = base_pos + pos_offset
        
        # Add small rotation randomization for diverse contact angles
        angle_variation = np.random.uniform(-0.1, 0.1, size=3)  # ~5 degrees
        from scipy.spatial.transform import Rotation as R
        base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])  # xyzw
        delta_rot = R.from_euler('xyz', angle_variation)
        randomized_rot = delta_rot * base_rot
        randomized_quat_xyzw = randomized_rot.as_quat()
        randomized_quat = np.array([randomized_quat_xyzw[3], randomized_quat_xyzw[0], 
                                     randomized_quat_xyzw[1], randomized_quat_xyzw[2]])  # wxyz
        
        return randomized_pos, randomized_quat


def collect_data_for_shape(
    env: TactileDataCollectionEnv,
    shape_name: str,
    shape_idx: int,
    num_samples: int,
    sample_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect tactile data for a specific shape.
    
    Args:
        env: Environment instance.
        shape_name: Name of the shape to collect data for.
        shape_idx: Index of the shape (for labeling).
        num_samples: Number of samples to collect.
        sample_interval: Steps between samples.
        
    Returns:
        tactile_images: Array of tactile images (N, H, W, 3).
        labels: Array of shape indices (N,).
    """
    print(f"\n[{shape_idx+1}/{len(env.cfg.shapes)}] Collecting data for shape: {shape_name}")
    
    # Set target shape
    env.set_target_shape(shape_name)
    env.reset()
    
    # Move goal to initial position
    base_pos = env.main_pose[0, :3].cpu().numpy()
    base_quat = env.main_pose[0, 3:].cpu().numpy()
    goal_pos, goal_quat = env.randomize_goal_pose(base_pos, base_quat)
    
    env.goal_prim_view.set_world_poses(
        positions=torch.tensor([goal_pos], device=env.device),
        orientations=torch.tensor([goal_quat], device=env.device)
    )
    
    # Data buffers
    tactile_images = []
    labels = []
    
    samples_collected = 0
    step_count = 0
    last_randomize_step = 0
    
    while samples_collected < num_samples and simulation_app.is_running():
        # Physics step
        env._pre_physics_step(None)
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        
        # Update IK commands
        positions, orientations = env.goal_prim_view.get_world_poses()
        env.ik_commands[:, :3] = positions - env.scene.env_origins
        env.ik_commands[:, 3:] = orientations
        
        # Update scene
        env.scene.update(dt=env.physics_dt)
        env.sim.render()
        
        # Sample tactile data
        if step_count % sample_interval == 0:
            tactile_rgb = env.gsmini.data.output.get("tactile_rgb", None)
            
            if tactile_rgb is not None:
                # Convert to numpy and normalize
                tactile_np = tactile_rgb[0].cpu().numpy()
                if tactile_np.max() > 1.0:
                    tactile_np = tactile_np / 255.0
                
                tactile_images.append(tactile_np)
                labels.append(shape_idx)
                
                samples_collected += 1
                
                if samples_collected % 10 == 0:
                    print(f"  Collected {samples_collected}/{num_samples} samples", end='\r')
        
        # Randomize goal pose periodically for diversity
        if args_cli.randomize_pose and step_count - last_randomize_step > 50:
            goal_pos, goal_quat = env.randomize_goal_pose(base_pos, base_quat)
            env.goal_prim_view.set_world_poses(
                positions=torch.tensor([goal_pos], device=env.device),
                orientations=torch.tensor([goal_quat], device=env.device)
            )
            last_randomize_step = step_count
        
        step_count += 1
    
    print(f"  ✓ Collected {samples_collected} samples for {shape_name}")
    
    return np.array(tactile_images), np.array(labels)


def collect_all_shapes_data(
    env: TactileDataCollectionEnv,
    output_path: Path,
    samples_per_shape: int,
    sample_interval: int,
):
    """Collect tactile data for all shapes and save to Zarr format.
    
    Args:
        env: Environment instance.
        output_path: Path to save Zarr file.
        samples_per_shape: Number of samples per shape.
        sample_interval: Steps between samples.
    """
    shape_names = list(env.cfg.shapes.keys())
    print(f"\n{'='*70}")
    print(f"Starting data collection for {len(shape_names)} shapes")
    print(f"Samples per shape: {samples_per_shape}")
    print(f"Total samples: {len(shape_names) * samples_per_shape}")
    print(f"{'='*70}\n")
    
    # Initialize data buffers
    all_tactile_images = []
    all_labels = []
    
    # Collect data for each shape
    for shape_idx, shape_name in enumerate(shape_names):
        tactile_images, labels = collect_data_for_shape(
            env, shape_name, shape_idx, samples_per_shape, sample_interval
        )
        
        all_tactile_images.append(tactile_images)
        all_labels.append(labels)
    
    # Concatenate all data
    all_tactile_images = np.concatenate(all_tactile_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\n{'='*70}")
    print("Data collection completed!")
    print(f"Total samples: {len(all_labels)}")
    print(f"Tactile image shape: {all_tactile_images.shape} (original resolution: 240x320)")
    print(f"Label distribution: {np.bincount(all_labels)}")
    print(f"{'='*70}\n")
    
    # Save to Zarr
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output_path), mode='w')
    
    root.create_dataset(
        'tactile_images',
        data=all_tactile_images,
        chunks=(100, 240, 320, 3),
        dtype=np.float32
    )
    root.create_dataset(
        'labels',
        data=all_labels,
        chunks=(1000,),
        dtype=np.int64
    )
    # Convert string list to fixed-length string array (avoids object_codec requirement)
    shape_names_array = np.array(shape_names, dtype='U50')  # U50: max 50 chars per name
    root.create_dataset(
        'shape_names',
        data=shape_names_array,
        chunks=(len(shape_names),)
    )
    
    # Metadata
    root.attrs['num_samples'] = len(all_labels)
    root.attrs['num_shapes'] = len(shape_names)
    root.attrs['samples_per_shape'] = samples_per_shape
    root.attrs['image_shape'] = all_tactile_images.shape[1:]
    root.attrs['collection_date'] = datetime.datetime.now().isoformat()
    
    print(f"Data saved to: {output_path}")
    print(f"Shape names: {shape_names}")


def run_simulator(env: TactileDataCollectionEnv):
    """Run simulation loop for data collection."""
    if env.cfg.gsmini.debug_vis:
        for data_type in env.cfg.gsmini.data_types:
            env.gsmini._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    
    print(f"Starting data collection with {env.num_envs} envs")
    
    env.reset()
    env.goal_prim_view = XFormPrim(prim_paths_expr="/Goal", name="Goal", usd=True)
    
    # Prepare output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args_cli.output_dir)
    output_path = output_dir / f"tactile_shape_data_{timestamp}.zarr"
    
    if args_cli.auto_mode:
        # Automatic data collection
        collect_all_shapes_data(
            env,
            output_path,
            args_cli.samples_per_shape,
            args_cli.sample_interval,
        )
    else:
        # Manual mode: user controls via GUI
        print("\nManual collection mode:")
        print("  - Move the green Goal marker in Isaac Sim to control robot")
        print("  - Press 's' in terminal to save current sample")
        print("  - Press 'n' to switch to next shape")
        print("  - Press 'q' to quit and save")
        
        shape_names = list(env.cfg.shapes.keys())
        current_shape_idx = 0
        all_tactile_images = []
        all_labels = []
        
        env.set_target_shape(shape_names[current_shape_idx])
        
        step_count = 0
        while simulation_app.is_running():
            env._pre_physics_step(None)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            
            positions, orientations = env.goal_prim_view.get_world_poses()
            env.ik_commands[:, :3] = positions - env.scene.env_origins
            env.ik_commands[:, 3:] = orientations
            
            env.scene.update(dt=env.physics_dt)
            env.sim.render()
            
            step_count += 1
    
    env.close()


def main():
    """Main function."""
    env_cfg = TactileDataCollectionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini.debug_vis = args_cli.debug_vis
    
    env = TactileDataCollectionEnv(env_cfg)
    
    print("[INFO]: Setup complete...")
    run_simulator(env=env)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
