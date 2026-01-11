import gymnasium as gym

from . import agents
from .allegro_pick_up_ball_env import AllegroPickUpBallEnv, AllegroPickUpBallEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="TacEx-Allegro-Pick-Up-Ball-v0",
    entry_point=f"{__name__}.allegro_pick_up_ball_env:AllegroPickUpBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AllegroPickUpBallEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

