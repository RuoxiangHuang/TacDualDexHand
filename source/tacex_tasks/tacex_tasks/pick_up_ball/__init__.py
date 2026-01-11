import gymnasium as gym

from . import agents
from .pick_up_ball_env import PickUpBallEnv, PickUpBallEnvCfg, PickUpBallGripperEnv, PickUpBallGripperEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="TacEx-Pick-Up-Ball-Uipc-v0",
    entry_point=f"{__name__}.pick_up_ball_env:PickUpBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PickUpBallEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="TacEx-Pick-Up-Ball-GripperOnly-Uipc-v0",
    entry_point=f"{__name__}.pick_up_ball_env:PickUpBallGripperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PickUpBallGripperEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

