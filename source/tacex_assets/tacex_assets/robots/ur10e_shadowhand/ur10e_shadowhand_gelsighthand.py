# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Ur10e_ShadowHand robot with GelSight Hand sensors.

The following configurations are available:

* :obj:`UR10E_SHADOWHAND_GELSIGHTHAND_RIGID_CFG`: Ur10e_ShadowHand with PhysX rigid-body GelSight Hand sensors.

Reference: Ur10e_ShadowHand robot with finger-shaped tactile sensors.

"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR

##
# Base Configuration Function
##


def _create_base_shadowhand_cfg() -> ArticulationCfg:
    """Create base configuration for Ur10e_ShadowHand robot.
    
    This is a helper function that creates the common configuration shared by
    both left and right hands. The only differences are:
    - USD file path (left vs right)
    - Initial position/orientation (if needed)
    
    Returns:
        Base ArticulationCfg with common settings
    """
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            # Default joint positions:
            # - UR10e arm: neutral/home position
            # - ShadowHand fingers: open position (will be set to joint upper limits)
            # Note: Specific joint positions can be set here if needed, e.g.:
            # joint_pos={
            #     "left_shoulder_pan_joint": 0.0,
            #     "left_shoulder_lift_joint": -1.57,
            #     "left_elbow_joint": 1.57,
            #     "left_wrist_1_joint": -1.57,
            #     "left_wrist_2_joint": 0.0,
            #     "left_wrist_3_joint": 0.0,
            #     # Fingers will be set to open position programmatically
            # },
        ),
        actuators={
            # UR10e arm joints (6 DOF: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
            # Note: This regex matches UR10e arm joints (lowercase: shoulder, elbow, wrist_1/2/3)
            # It does NOT match ShadowHand wrist joints (WRJ1, WRJ2) which are part of the hand
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*(shoulder|elbow|wrist_1|wrist_2|wrist_3|ur10).*"],
                effort_limit_sim=150.0,  # UR10e typical torque limits
                stiffness=400.0,  # Position control with high stiffness (similar to Franka HIGH_PD)
                damping=80.0,  # High damping for stable control
                friction=0.0,
            ),
            # ShadowHand finger and wrist joints (24 DOF total):
            # - FFJ1-4 (index finger, 4 joints)
            # - MFJ1-4 (middle finger, 4 joints)
            # - RFJ1-4 (ring finger, 4 joints)
            # - LFJ1-5 (little finger, 5 joints)
            # - THJ1-5 (thumb, 5 joints)
            # - WRJ1-2 (ShadowHand wrist, 2 joints)
            # Note: Joint names are UPPERCASE in the USD file
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*(FFJ|MFJ|RFJ|LFJ|THJ|WRJ).*"],  # Match finger and hand wrist joints (uppercase)
                effort_limit_sim=0.5,  # ShadowHand finger torque limits
                stiffness=3.0,  # Position control with stiffness
                damping=0.1,
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


##
# Left Hand Configuration
##

UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG = _create_base_shadowhand_cfg()
UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG.spawn.usd_path = (
    f"{TACEX_ASSETS_DATA_DIR}/Robots/Ur10e_ShadowHand/ur10e_shadow_left_hand_glb_WITH_TAC.usd"
)
# Ensure the path uses forward slashes (works on both Windows and Linux)
UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG.spawn.usd_path = UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG.spawn.usd_path.replace("\\", "/")
"""Configuration of Ur10e_ShadowHand LEFT hand robot with PhysX rigid-body GelSight Hand sensors.

Sensor case prim names:
- left_ffdistal_case (index finger)
- left_mfdistal_case (middle finger)
- left_rfdistal_case (ring finger)
- left_lfdistal_case (little finger)
- left_thdistal_case (thumb)

Gelpad prim names:
- left_ffdistal_gelpad
- left_mfdistal_gelpad
- left_rfdistal_gelpad
- left_lfdistal_gelpad
- left_thdistal_gelpad

Camera prim paths: `left_XXdistal_case/Camera` (where XX is ff, mf, rf, lf, th)
"""


##
# Right Hand Configuration
##

UR10E_SHADOWHAND_RIGHT_GELSIGHTHAND_RIGID_CFG = _create_base_shadowhand_cfg()
UR10E_SHADOWHAND_RIGHT_GELSIGHTHAND_RIGID_CFG.spawn.usd_path = (
    f"{TACEX_ASSETS_DATA_DIR}/Robots/Ur10e_ShadowHand/ur10e_shadow_right_hand_glb_WITH_TAC.usd"
)
"""Configuration of Ur10e_ShadowHand RIGHT hand robot with PhysX rigid-body GelSight Hand sensors.

Sensor case prim names:
- right_ffdistal_case (index finger)
- right_mfdistal_case (middle finger)
- right_rfdistal_case (ring finger)
- right_lfdistal_case (little finger)
- right_thdistal_case (thumb)

Gelpad prim names:
- right_ffdistal_gelpad
- right_mfdistal_gelpad
- right_rfdistal_gelpad
- right_lfdistal_gelpad
- right_thdistal_gelpad

Camera prim paths: `right_XXdistal_case/Camera` (where XX is ff, mf, rf, lf, th)
"""


##
# Backward Compatibility: Default to Left Hand
##

UR10E_SHADOWHAND_GELSIGHTHAND_RIGID_CFG = UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG
"""Backward compatibility alias. Defaults to LEFT hand configuration."""

