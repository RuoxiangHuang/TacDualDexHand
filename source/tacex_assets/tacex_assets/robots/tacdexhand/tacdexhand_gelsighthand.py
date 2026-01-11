# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the TacDexHand robot with GelSight Hand sensors.

The following configurations are available:

* :obj:`TACDEXHAND_GELSIGHTHAND_RIGID_CFG`: TacDexHand with PhysX rigid-body GelSight Hand sensors.

Reference: Custom TacDexHand robot with finger-shaped tactile sensors.

"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR

##
# Configuration
##

TACDEXHAND_GELSIGHTHAND_RIGID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/TacDexHand/figure_hand_tac.usd",
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
        # Adjust joint positions based on your robot's joint names
        # joint_pos={"joint_name": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of TacDexHand robot with PhysX rigid-body GelSight Hand sensors.

Sensor case prim name: `fingercase3` (located under root_joint)
Gelpad prim name: `fingergelpad3` (located under fingercase3)
Camera prim path: `fingercase3/Camera`
"""

TACDEXHAND_5FINGERS_GELSIGHTHAND_RIGID_CFG = TACDEXHAND_GELSIGHTHAND_RIGID_CFG.copy()
TACDEXHAND_5FINGERS_GELSIGHTHAND_RIGID_CFG.spawn.usd_path = (
    f"{TACEX_ASSETS_DATA_DIR}/Robots/TacDexHand/5figures_hand_tac.usd"
)
"""Configuration of TacDexHand robot with 5 GelSight Hand sensors (one on each finger).

Sensor case prim names: `fingercase3_00` to `fingercase3_04` (located under root_joint)
Gelpad prim names: `fingergelpad3_00` to `fingergelpad3_04` (located under respective fingercase3_XX)
Camera prim paths: `fingercase3_XX/Camera` (where XX is 00-04)
"""

