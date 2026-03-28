"""
Set up OMY-F3M CuRobo files.

Copies robot config, collision spheres, URDF and meshes into third_party/curobo/.

Usage:
    python scripts/setup_omy_curobo.py
"""

import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_SRC = os.path.join(REPO_ROOT, "scripts/curobo_configs/omy_f3m")
CUROBO = os.path.join(REPO_ROOT, "third_party/curobo/src/curobo/content")

COPIES = [
    ("omy_f3m.yml", "configs/robot/omy_f3m.yml"),
    ("spheres_omy_f3m.yml", "configs/robot/spheres/omy_f3m.yml"),
]

URDF_SRC = os.path.join(REPO_ROOT, "InfiniGym/assets/urdf/omy_f3m")
URDF_DST = os.path.join(CUROBO, "assets/robot/omy_f3m")

def main():
    for src_name, dst_rel in COPIES:
        src = os.path.join(CONFIGS_SRC, src_name)
        dst = os.path.join(CUROBO, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {dst_rel}")

    if os.path.exists(URDF_DST):
        shutil.rmtree(URDF_DST)
    shutil.copytree(URDF_SRC, URDF_DST)
    print(f"  assets/robot/omy_f3m/")
    print("Done.")

if __name__ == "__main__":
    main()
