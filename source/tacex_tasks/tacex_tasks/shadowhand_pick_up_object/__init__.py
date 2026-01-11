import gymnasium as gym

from .shadowhand_pick_up_object_env import (
    ShadowHandPickUpObjectEnv,
    ShadowHandPickUpObjectEnvCfg,
    ShadowHandPickUpObjectScriptedEnv,
    ShadowHandPickUpObjectScriptedEnvCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="TacEx-ShadowHand-Pick-Up-Object-v0",
    entry_point=f"{__name__}.shadowhand_pick_up_object_env:ShadowHandPickUpObjectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandPickUpObjectEnvCfg,
    },
)

gym.register(
    id="TacEx-ShadowHand-Pick-Up-Object-Scripted-v0",
    entry_point=f"{__name__}.shadowhand_pick_up_object_env:ShadowHandPickUpObjectScriptedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ShadowHandPickUpObjectScriptedEnvCfg,
    },
)

