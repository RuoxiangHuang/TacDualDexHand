# TacDualDexHand - Tactile Extension for Isaac Sim/Isaac Lab
**TacDualDexHand** brings **Vision-Based Tactile Sensor (VBTS)** into Isaac Sim/Lab for dual dexterous hands.

The framework integrates multiple simulation approaches for VBTS's and aims to be modular and extendable.
Components can be easily switched out, added and modified.

Currently, only the **GelSight Mini** is supported, but you can also easily add your own sensor (guide coming soon). We also plan to add more VBTS types later.

## **Main features**:
- [GPU accelerated Tactile RGB simulation](https://github.com/TimSchneider42/taxim) via [Taxim](https://github.com/Robo-Touch/Taxim)'s simulation approach
- Marker Motion Simulation via [FOTS](https://github.com/Rancho-zhao/FOTS)
- Integration of [UIPC](https://github.com/spiriMirror/libuipc) for GPU accelerated incremental potential contact to simulate FEM soft bodies, rigid bodies, cloth, etc. in a penetration-free and robust manner
- Marker Motion Simulation with FEM soft body based on the simulator used by the [ManiSkill-ViTac challenge](https://github.com/chuanyune/ManiSkill-ViTac2025) that leverages UIPC


Checkout the [website](https://sites.google.com/view/tacex) for showcases and the documentation for details, guides and tutorials.


## Installation
> [!NOTE]
> TacEx currently works with **Isaac Sim 4.5** and **IsaacLab 2.1.1**.
> The installation was tested on Ubuntu 22.04 with a 4090 GPU and Driver Version 550.163.01 + Cuda 12.4.

**0.** Make sure that you have **git-lfs**:

```bash
# Need it for the USD assets
git lfs install
```

**1.** Clone this repository and its submodules:
```bash
git clone --recurse-submodules https://github.com/RuoxiangHuang/TacDualDexHand
cd TacDualDexHand
```

Then **install TacDualDexHand** [locally](docs/source/installation/Local-Installation.md)
or build a [Docker Container](docs/source/installation/Docker-Container-Setup.md).


## Contributing
Contributions of any kind are, of course, very welcome.
Be it suggestions, feedback, bug reports or pull requests.

Let's work together to advance tactile sensing in robotics!!!

## Citation
```bibtex
@article{nguyen2024tacexgelsighttactilesimulation,
      title={TacEx: GelSight Tactile Simulation in Isaac Sim -- Combining Soft-Body and Visuotactile Simulators},
      author={Duc Huy Nguyen and Tim Schneider and Guillaume Duret and Alap Kshirsagar and Boris Belousov and Jan Peters},
      year={2024},
      eprint={2411.04776},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04776},
}
```

## Acknowledgements

TacEx is built upon code from
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/main)
- [Taxim](https://github.com/Robo-Touch/Taxim)
- [FOTS](https://github.com/Rancho-zhao/FOTS)
- [UIPC](https://github.com/spiriMirror/libuipc)
- [ManiSkill-ViTac challenge](https://github.com/chuanyune/ManiSkill-ViTac2025)
