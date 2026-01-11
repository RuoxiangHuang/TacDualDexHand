"""Package containing robot configurations."""

import os
import toml

from isaaclab_tasks.utils import import_packages

from .ur10e_shadowhand.ur10e_shadowhand_gelsighthand import (
    UR10E_SHADOWHAND_LEFT_GELSIGHTHAND_RIGID_CFG,
    UR10E_SHADOWHAND_RIGHT_GELSIGHTHAND_RIGID_CFG,
    UR10E_SHADOWHAND_GELSIGHTHAND_RIGID_CFG,
)

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
