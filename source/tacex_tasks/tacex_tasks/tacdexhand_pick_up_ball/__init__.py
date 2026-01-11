import gymnasium as gym

from . import agents
from .tacdexhand_pick_up_ball_env import TacDexHandPickUpBallEnv, TacDexHandPickUpBallEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="TacEx-TacDexHand-Pick-Up-Ball-5Fingers-Uipc-v0",
    entry_point=f"{__name__}.tacdexhand_pick_up_ball_env:TacDexHandPickUpBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TacDexHandPickUpBallEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

