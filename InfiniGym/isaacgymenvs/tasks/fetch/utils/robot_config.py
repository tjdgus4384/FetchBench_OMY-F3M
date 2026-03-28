from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class RobotConfig:
    name: str
    num_arm_dofs: int
    num_gripper_dofs: int

    eef_link_name: str
    left_finger_link_name: str
    right_finger_link_name: str

    left_finger_contact_substr: str
    right_finger_contact_substr: str

    eef_joint_name: str
    arm_joint_names: List[str]
    default_dof_pos: List[float]

    urdf_file: str
    flip_visual_attachments: bool

    gripper_type: str  # "prismatic" or "revolute_mimic"
    gripper_mimic_multiplier: List[float]

    curobo_config_name: str

    # Collision filter for create_actor (0 = self-collision enabled, 1 = disabled)
    self_collision_filter: int = 0

    # Whether "open" maps to upper DOF limit (True for Franka prismatic, False for OMY revolute)
    gripper_open_is_upper: bool = True

    # Per-robot gripper DOF property overrides (None = use global defaults from YAML)
    gripper_stiffness: Optional[float] = None
    gripper_damping: Optional[float] = None
    gripper_effort: Optional[float] = None

    # Gripper velocity for velocity-mode control (rad/s for revolute, m/s for prismatic).
    # Must be large enough to fully close/open within gripper_steps simulation steps.
    gripper_velocity: float = 0.1

    # EEF approach axis in EEF-local frame (direction the gripper advances toward object).
    # Franka: Z-axis [0,0,1], OMY: -Y axis [0,-1,0]
    eef_approach_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])

    # CGN gripper depth: distance from EEF frame origin to finger contact baseline
    # along the approach axis (metres). Used by Contact GraspNet to position the
    # gripper frame relative to predicted contact points.
    # Franka panda_hand: 0.1034,  OMY end_effector_flange_link: 0.083
    cgn_gripper_depth: float = 0.1034

    # Free-space retract target poses in robot frame (positions + quaternions wxyz).
    # motion_gen_to_free_space uses these to move the robot away after grasping.
    free_space_target_positions: List[List[float]] = field(
        default_factory=lambda: [[-0.2, -0.25, 0.66], [-0.2, 0.25, 0.66]])
    free_space_target_quaternions_wxyz: List[List[float]] = field(
        default_factory=lambda: [[0, 0.707, -0.707, 0], [0, 0.707, 0.707, 0]])

    # Grasp EEF frame correction: 4x4 matrix applied to Franka-frame grasp poses
    # to convert them to this robot's EEF frame. Identity for Franka.
    grasp_eef_correction: Optional[np.ndarray] = None

    @property
    def num_total_dofs(self) -> int:
        return self.num_arm_dofs + self.num_gripper_dofs


FRANKA_CONFIG = RobotConfig(
    name="franka_panda",
    num_arm_dofs=7,
    num_gripper_dofs=2,
    eef_link_name="panda_hand",
    left_finger_link_name="panda_leftfinger",
    right_finger_link_name="panda_rightfinger",
    left_finger_contact_substr="leftfinger",
    right_finger_contact_substr="rightfinger",
    eef_joint_name="panda_hand_joint",
    arm_joint_names=[
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ],
    default_dof_pos=[
        -0.31267092, -1.1996635, 0.05781832, -2.1767514,
        0.06494738, 0.9786396, 0.53001183, 0.04, 0.04,
    ],
    urdf_file="urdf/franka_description/robots/{type}.urdf",
    flip_visual_attachments=True,
    gripper_type="prismatic",
    gripper_mimic_multiplier=[1.0, 1.0],
    curobo_config_name="franka_r3.yml",
)


OMY_F3M_CONFIG = RobotConfig(
    name="omy_f3m",
    num_arm_dofs=6,
    num_gripper_dofs=4,
    eef_link_name="end_effector_flange_link",
    left_finger_link_name="rh_p12_rn_l2",
    right_finger_link_name="rh_p12_rn_r2",
    left_finger_contact_substr="rh_p12_rn_l",
    right_finger_contact_substr="rh_p12_rn_r",
    eef_joint_name="gripper_fixed",
    arm_joint_names=[
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    ],
    default_dof_pos=[
        0.0, -1.57, 1.57, 0.0, 1.57, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ],
    urdf_file="urdf/omy_f3m/omy_f3m.urdf",
    flip_visual_attachments=False,
    gripper_type="revolute_mimic",
    gripper_mimic_multiplier=[1.0, 1.0, 1.0, 1.0],
    curobo_config_name="omy_f3m.yml",
    gripper_open_is_upper=False,
    self_collision_filter=1,
    eef_approach_axis=[0.0, -1.0, 0.0],
    # EEF -Y pointing down (world -Z), two wrist rotations for goalset flexibility.
    # Positions lowered to 0.55 (vs Franka 0.66) to stay within OMY workspace.
    free_space_target_positions=[[-0.15, -0.2, 0.55], [-0.15, 0.2, 0.55]],
    # Use orientations that OMY-F3M can reach exactly (IK rot_err=0.0).
    # Approach (-Y) points sideways, not down, but that's fine for retract.
    free_space_target_quaternions_wxyz=[[0.5, 0.5, -0.5, 0.5],
                                        [0.5, 0.5, 0.5, -0.5]],
    gripper_stiffness=1e4,
    gripper_damping=10.0,
    gripper_effort=15.0,
    gripper_velocity=0.9,  # 1.135 rad range / (90 steps * 0.0166s) ≈ 0.76; use 0.9 for margin
    # RH-P12-RN has revolute fingers so depth varies with grip width:
    #   80mm opening → 101mm,  60mm → 107mm,  40mm → 110mm
    # Representative value for CGN's typical 40-80mm prediction range.
    cgn_gripper_depth=0.105,
    # Franka panda_hand → OMY end_effector_flange_link frame correction:
    #   Rotation: Franka Z (approach) → OMY -Y,  Franka Y (spread) → OMY X
    #   Translation: depth delta ≈ 0.105 - 0.1034 ≈ 0.002m along Franka Z,
    #   negligible so set to zero.  (Revolute-finger depth varies by grip width;
    #   a single fixed correction cannot capture this exactly.)
    grasp_eef_correction=np.array([
        [ 0,  0, -1,  0.0],
        [ 1,  0,  0,  0.0],
        [ 0, -1,  0,  0.0],
        [ 0,  0,  0,  1.0],
    ], dtype=np.float32),
)


_ROBOT_CONFIGS = {
    "franka_panda": FRANKA_CONFIG,
    "omy_f3m": OMY_F3M_CONFIG,
}


def get_robot_config(robot_name: str, robot_type: Optional[str] = None) -> RobotConfig:
    """Factory function to get a RobotConfig by name.

    Args:
        robot_name: "franka_panda" or "omy_f3m"
        robot_type: URDF variant (used by franka only, e.g. "franka_r3_cvx")
    """
    if robot_name not in _ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_name}. Available: {list(_ROBOT_CONFIGS.keys())}")

    import copy
    cfg = copy.deepcopy(_ROBOT_CONFIGS[robot_name])

    # Format URDF path with type if needed
    if robot_type is not None and "{type}" in cfg.urdf_file:
        cfg.urdf_file = cfg.urdf_file.format(type=robot_type)
    elif "{type}" in cfg.urdf_file:
        # Default type for franka
        cfg.urdf_file = cfg.urdf_file.format(type="franka_r3")

    return cfg
