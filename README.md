# FetchBench + OMY-F3M + GraspGen Integration

This repository extends the [FetchBench](https://arxiv.org/abs/2406.11793) benchmark with:
- **OMY-F3M robot** (ROBOTIS OpenMANIPULATOR-Y F3M, 6-DOF + RH-P12-RN gripper)
- **GraspGen** ([NVIDIA, ICRA'26](https://arxiv.org/abs/2507.13097)) as a grasp generation method

Based on [FetchBench-CORL2024](https://github.com/princeton-vl/FetchBench-CORL2024) (Isaac Gym).


## Overview

```
FetchBench (Isaac Gym)
├── Franka Panda (7-DOF, original)
└── OMY-F3M (6-DOF, added)

Grasp Generation:
  ├── ContactGraspNet (CGN)  ← original baseline
  └── GraspGen (Diffusion)   ← added, runs as ZMQ server

Pipeline:
  Camera → Point Cloud → [CGN or GraspGen] → Grasp Poses
    → CuRobo (IK + Motion Planning + Collision Check) → Execute → Evaluate
```

### Grasp Retargeting

GraspGen is trained on Franka Panda. To use with OMY-F3M:
1. **GraspGen outputs** grasp poses in Franka EEF frame (approach=+Z, closing=±X in GraspGen convention → ±Y in URDF)
2. **`graspgen_to_franka`** rotates from GraspGen internal convention to Franka panda_hand frame (+90 deg Z)
3. **`grasp_eef_correction`** rotates from Franka EEF to OMY-F3M EEF (axis mapping: Franka +Z approach → OMY -Y approach)
4. **`depth_offset`** (+0.017m) compensates for OMY-F3M's shorter fingers (Franka depth 0.105m vs OMY 0.088m)

For CGN, the same `grasp_eef_correction` is applied. CGN also has `cgn_gripper_depth` in `robot_config.py` for its internal depth parameter.


## Installation

### 1. FetchBench Environment (Python 3.8, PyTorch 1.12/1.13)

Follow the original FetchBench installation. Then install ZMQ client dependencies:

```bash
conda activate fetchbench
pip install pyzmq msgpack msgpack-numpy
```

### 2. GraspGen Environment (Python 3.10, PyTorch 2.1)

GraspGen requires a separate conda environment due to PyTorch version mismatch.

```bash
# Create environment
conda create -n graspgen python=3.10 -y
conda activate graspgen

# Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install GraspGen
cd third_party/GraspGen
pip install -e .

# Build PointNet2 CUDA extension
# Rename source dir so the installed package is used at runtime
# (the local pointnet2_ops/ dir would shadow the compiled site-packages version)
mv pointnet2_ops pointnet2_ops_src
cd pointnet2_ops_src && pip install --no-build-isolation . && cd ..
```

### 3. Download GraspGen Model Checkpoints

```bash
# Install git-lfs if not installed
apt-get install git-lfs && git lfs install

# Clone checkpoints (~2.5 GB for all grippers, ~1 GB for Franka only)
git clone https://huggingface.co/adithyamurali/GraspGenModels
cd GraspGenModels && git lfs pull --include="checkpoints/graspgen_franka_panda*"
```

### 4. CuRobo OMY-F3M Configuration

CuRobo needs robot config, collision spheres, URDF and meshes inside `third_party/curobo/`. These are not tracked by git. Run:

```bash
python scripts/setup_omy_curobo.py
```

This copies robot config, collision spheres, URDF and meshes from `scripts/curobo_configs/omy_f3m/` (tracked by git) into `third_party/curobo/`. To regenerate collision spheres from visual meshes, run `python scripts/generate_omy_spheres.py`.


## Running Experiments

### Environment Variables

```bash
export ASSET_PATH=/path/to/FetchBench-CORL2024
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### CGN + Franka (Original Baseline)

```bash
conda activate fetchbench
cd InfiniGym
python isaacgymenvs/eval.py task=FetchMeshCuroboPtdCGNBeta scene=benchmark_eval/RigidObjDesk_0
```

### CGN + OMY-F3M

```bash
python isaacgymenvs/eval.py task=FetchMeshCuroboPtdCGNBeta scene=benchmark_eval/RigidObjDesk_0 \
    task.env.robot.robot_name=omy_f3m
```

### GraspGen + OMY-F3M

**Terminal 1 — GraspGen Server** (graspgen environment):
```bash
conda activate graspgen
cd third_party/GraspGen
python -c "
import logging; logging.basicConfig(level=logging.INFO)
from grasp_gen.serving.zmq_server import GraspGenZMQServer
GraspGenZMQServer(gripper_config='<path_to_GraspGenModels>/checkpoints/graspgen_franka_panda.yml', port=5556).serve_forever()
"
```
> export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` 해야 할 수 있음. (CXXABI_1.3.15 not found)

**Terminal 2 — FetchBench Evaluation** (fetchbench environment):
```bash
conda activate fetchbench
cd InfiniGym
python isaacgymenvs/eval.py task=FetchMeshCuroboGraspGen scene=benchmark_eval/RigidObjDesk_0 \
    task.env.robot.robot_name=omy_f3m
```

### GraspGen + Franka

```bash
# Same GraspGen server as above, then:
python isaacgymenvs/eval.py task=FetchMeshCuroboGraspGen scene=benchmark_eval/RigidObjDesk_0
```

### Useful Overrides

```bash
task.env.numTasks=5                    # limit number of tasks (default: 60)
task.viewer.enable=False               # headless mode
task.solution.pre_grasp_offset=0.04    # distance from pre-grasp to grasp pose
task.solution.retract_offset=0.02      # lift height after grasping
```

## Technical Notes

### CuRobo Collision Spheres

The URDF collision geometries for OMY-F3M are conservatively oversized (e.g., link6 collision cylinder r=0.08 vs actual visual mesh r=0.04). The collision spheres are auto-generated from **visual meshes** (not URDF collision) using CuRobo's `fit_spheres_to_mesh`. To regenerate:

```bash
conda activate fetchbench
python scripts/generate_omy_spheres.py
```

See `scripts/generate_omy_spheres.py` for details. 임시 파일이고, 실제 실험할 때는, [Isaac Sim Robot Description Editor](https://curobo.org/tutorials/1_robot_configuration.html) 사용해야 할 것.

### GraspGen ZMQ Server Architecture

GraspGen requires PyTorch 2.1+ while FetchBench uses PyTorch 1.12. Communication is via ZMQ:

```
[graspgen env]                      [fetchbench env]
GraspGenZMQServer ──tcp:5556──>  GraspGenWrapper (graspgen_utils.py)
PyTorch 2.1, CUDA 12.1            PyTorch 1.12, CUDA 11.2
Only: pyzmq, msgpack               Only: pyzmq, msgpack (no torch needed)
```

### Free-Space Target Quaternions (OMY-F3M)

OMY-F3M is 6-DOF, so not all position+orientation combinations are reachable. The `free_space_target_quaternions_wxyz` must be orientations that the IK solver can reach exactly (rot_err < 0.02). The current values `[0.5, 0.5, -0.5, 0.5]` and `[0.5, 0.5, 0.5, -0.5]` achieve rot_err=0.0 at the free-space positions.


## Known Issues

1. **cartesian_linear has no collision check**: The pre-grasp → grasp approach uses `follow_cartesian_linear_motion` without CuRobo collision checking. The robot may collide with the desk or objects during this phase. Applies to both CGN and GraspGen, both Franka and OMY-F3M.

2. **6-DOF workspace limitations**: OMY-F3M cannot reach some positions/orientations that Franka (7-DOF) can. Some benchmark tasks may be unreachable.

3. **Collision sphere precision**: Auto-fitted spheres are good but not perfect. Isaac Sim Robot Description Editor produces more precise spheres (used for Franka). Manual tuning may improve OMY-F3M results.

4. **Scenes optimized for Franka**: FetchBench scenes (robot position, object placement) are designed for Franka's workspace. OMY-F3M may need scene adjustments for optimal performance.

5. **Fetch "Invalid Problem"**: Some tasks fail at the fetch (retract) phase with "Invalid Problem" status. This occurs when the attached object creates a collision state that CuRobo cannot plan from. Increasing `retract_offset` may help.