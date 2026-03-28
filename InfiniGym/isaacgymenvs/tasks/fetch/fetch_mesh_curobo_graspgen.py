import isaacgym
import numpy as np
import os
import torch
import time
import trimesh
import trimesh.transformations as tr

from curobo.types.math import Pose
from curobo.types.robot import JointState

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, tf_inverse
from isaacgymenvs.tasks.fetch.fetch_ptd import FetchPointCloudBase
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import FetchMeshCurobo
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video

from isaacgymenvs.tasks.fetch.utils.graspgen_utils import GraspGenWrapper

# Depth difference: Franka (0.10527m) - OMY-F3M (0.088m)
GRASPGEN_DEPTH_OFFSET = 0.017


class FetchMeshCuroboGraspGen(FetchMeshCurobo, FetchPointCloudBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        graspgen_cfg = self.cfg["solution"]["graspgen"]
        self.grasp_net = GraspGenWrapper(
            host=graspgen_cfg.get("host", "localhost"),
            port=graspgen_cfg.get("port", 5556),
            num_grasps=graspgen_cfg.get("num_grasps", 200),
            topk=graspgen_cfg.get("top_k", 100),
        )
        self.graspgen_confidence_th = graspgen_cfg.get("confidence_th", 0.0)
        self.graspgen_top_k = graspgen_cfg.get("top_k", 100)

        self.graspgen_log_dir = f'./logs/graspgen_log/{self.cfg["experiment_name"]}'
        if not os.path.exists(self.graspgen_log_dir):
            os.makedirs(self.graspgen_log_dir)

    """
    Sample Utils
    """

    def get_robot_world_matrix(self, env_idx, device):
        rq, rt = tf_inverse(self._robot_base_state[..., 3:7].clone(), self._robot_base_state[..., :3].clone())
        rq = torch.concat([rq[:, -1:], rq[:, :-1]], dim=-1).cpu().numpy()
        rt = rt.cpu().numpy()

        trans = tr.translation_matrix(rt[env_idx])
        rot = tr.quaternion_matrix(rq[env_idx])
        pose = trans @ rot

        pose = to_torch([pose], device=device)
        return pose

    def get_graspgen_input(self, cam_data):
        """Extract object point cloud from the best camera view.

        Follows CGN pipeline: select the camera with most goal object pixels,
        use only that camera's partial point cloud (matching GraspGen's training
        distribution of single-viewpoint partial observations).
        """
        inputs = []
        for env_idx in range(self.num_envs):
            goal_seg_id = self.task_obj_index[env_idx][self.get_task_idx()] + 4

            segs = cam_data["camera_pointcloud_raw"][env_idx]["seg"]
            world_pts = cam_data["camera_pointcloud_raw"][env_idx]["pts"]

            # Select camera with most goal object pixels (same as CGN)
            num_goal_pixels = []
            for seg in segs:
                num_goal_pixels.append((seg == goal_seg_id).float().sum().cpu().numpy())
            cam_idx = np.argmax(np.array(num_goal_pixels))

            # Use only the best camera's partial point cloud
            goal_pts = world_pts[cam_idx][segs[cam_idx] == goal_seg_id]

            inputs.append(goal_pts.cpu().numpy())
        return inputs

    """
    Motion Gen
    """

    def motion_gen_to_pose_goalset_single_env(self, target_poses, env_idx, offset=False):
        self._refresh()

        q_start = JointState.from_position(
            self.states["q"][env_idx:env_idx+1, :self.n_arm].clone().to(self.tensor_args.device),
            joint_names=self.robot_joint_names
        )

        if target_poses is None:
            return None, False, None, None

        assert len(target_poses.shape) == 3
        config = self.motion_plan_config_line.clone() if offset else self.motion_plan_config_graph.clone()
        result = self.motion_generators[env_idx].plan_goalset(q_start, target_poses, config)
        traj = result.get_interpolated_plan()

        return traj, result.success.cpu().numpy()[0], target_poses, result

    def motion_gen_to_grasp_pose_ordered(self, command_poses, lsts):
        trajs, poses, success, results = [], [], [], []
        for i in range(self.num_envs):
            lst = lsts[i]
            for l in lst:
                target_poses = command_poses[i][l].unsqueeze(dim=0).clone()
                traj, suc, pose, result = self.motion_gen_to_pose_goalset_single_env(target_poses, env_idx=i)

                if suc:
                    trajs.append(traj)
                    poses.append(pose)
                    results.append(result)
                    success.append(suc)
                    break

            if len(trajs) < i + 1:
                trajs.append(None)
                poses.append(None)
                results.append(None)
                success.append(False)

        return self._get_batch_joint_trajs(trajs), success, poses, results

    def get_grasp_masks(self, scores):
        indexed_lst = list(enumerate(scores))
        k = min(len(indexed_lst), self.graspgen_top_k)
        top_k = sorted(indexed_lst, key=lambda x: x[1], reverse=True)[:k]

        top_lst = []
        for i in top_k:
            if i[1] >= self.graspgen_confidence_th:
                top_lst.append(i[0])

        return np.array(top_lst)

    def sample_goal_obj_collision_free_grasp_pose(self):
        cam_data = self.get_camera_data(tensor_ptd=True, ptd_in_robot_base=False, segmented_ptd=False)
        graspgen_inputs = self.get_graspgen_input(cam_data)

        grasp_poses_all, pre_grasp_poses_all, grasp_successes_all, grasp_lsts_all = [], [], [], []
        logs = []

        # GraspGen convention (closing=±X) → FetchBench Franka panda_hand (closing=±Y)
        # +90° Z rotation
        graspgen_to_franka = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ], dtype=np.float32)

        # For non-Franka robots: add depth offset + EEF frame correction
        if self.robot_cfg.grasp_eef_correction is not None:
            depth_offset = tr.translation_matrix([0, 0, GRASPGEN_DEPTH_OFFSET])
            grasp_correction = graspgen_to_franka @ depth_offset @ self.robot_cfg.grasp_eef_correction
        else:
            grasp_correction = graspgen_to_franka

        grasp_correction = to_torch([grasp_correction],
                                    device=self.tensor_args.device, dtype=torch.float)

        for env_idx in range(self.num_envs):
            object_pc = graspgen_inputs[env_idx]

            # Run GraspGen inference (output in world frame, Franka EEF convention)
            grasp_poses_np, scores_np = self.grasp_net.predict(object_pc)
            logs.append({'grasp_poses': grasp_poses_np, 'scores': scores_np})

            if len(grasp_poses_np) == 0:
                grasp_poses_all.append([])
                pre_grasp_poses_all.append([])
                grasp_successes_all.append([])
                grasp_lsts_all.append([])
                continue

            grasp_in_world = to_torch(grasp_poses_np, device=self.tensor_args.device)

            # Transform: world → robot frame
            robot_world_pose = self.get_robot_world_matrix(env_idx, self.tensor_args.device).reshape(1, 4, 4)

            # GraspGen output is Franka EEF frame.
            # Apply depth offset (in Franka frame) then rotate to target robot EEF.
            grasp_in_robot = robot_world_pose @ grasp_in_world @ grasp_correction

            grasp_pose = Pose.from_matrix(grasp_in_robot)

            # Pre-grasp: offset along approach axis
            pre_grasp_offset_pos = to_torch(
                self.get_approach_offset(-self.cfg["solution"]["pre_grasp_offset"],
                                         device=self.tensor_args.device),
                device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_pos = pre_grasp_offset_pos.unsqueeze(dim=0)
            pre_grasp_offset_quat = to_torch([1, 0, 0, 0], device=self.tensor_args.device, dtype=torch.float)
            pre_grasp_offset_quat = pre_grasp_offset_quat.unsqueeze(dim=0)
            pre_grasp_offset = Pose(pre_grasp_offset_pos, pre_grasp_offset_quat)
            pre_grasp_pose = grasp_pose.multiply(pre_grasp_offset)

            grasp_lst = self.get_grasp_masks(scores_np)
            grasp_success = torch.zeros((grasp_in_world.shape[0],), dtype=torch.long, device=self.device)
            grasp_success[grasp_lst] += 1

            grasp_poses_all.append(grasp_pose)
            pre_grasp_poses_all.append(pre_grasp_pose)
            grasp_successes_all.append(grasp_success)
            grasp_lsts_all.append(grasp_lst)

        res = {
            'grasp_poses': grasp_poses_all,
            'pre_grasp_poses': pre_grasp_poses_all,
            'grasp_success': grasp_successes_all,
            'grasp_ordered_lst': grasp_lsts_all,
        }

        return res, logs

    """
    Solve
    """

    def solve(self):
        log = {}

        self.set_target_color()
        self._solution_video = []
        self._video_frame = 0
        computing_time = 0.

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()

        self.update_cuRobo_world_collider_pose()

        st = time.time()
        graspgen_result, graspgen_logs = self.sample_goal_obj_collision_free_grasp_pose()
        computing_time += time.time() - st
        if self.graspgen_log_dir is not None:
            np.save(f'{self.graspgen_log_dir}/log_{self.get_task_idx()}.npy', graspgen_logs)

        if self.cfg["solution"]["direct_grasp"]:
            self.update_cuRobo_world_collider_pose()
            start_time = time.time()
            traj, success, poses, results = self.motion_gen_to_grasp_pose_ordered(
                graspgen_result['grasp_poses'], graspgen_result['grasp_ordered_lst'])
            print("Grasp Plan", success)
            log['grasp_plan_success'] = success
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=0)
            log['grasp_execute_error'] = self.get_end_effect_error(poses)

        else:
            start_time = time.time()
            traj, success, poses, results = self.motion_gen_to_grasp_pose_ordered(
                graspgen_result["pre_grasp_poses"], graspgen_result['grasp_ordered_lst'])
            print("Pre Grasp Plan", success)
            log['pre_grasp_plan_success'] = success
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=0)
            print("Pre Grasp Phase End")
            log['pre_grasp_execute_error'] = self.get_end_effect_error(poses)

            if self.cfg["solution"]["move_offset_method"] == 'motion_planning':
                self.update_cuRobo_world_collider_pose()
                if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                    self._enable_goal_obj_collision_checking(False)

                start_time = time.time()
                traj, success, poses, results = self.motion_gen_by_z_offset(
                    z=self.cfg["solution"]["pre_grasp_offset"], mask=success)
                computing_time += time.time() - start_time
                print("Grasp Plan", success)

                if self.cfg["solution"]["disable_grasp_obj_motion_gen"]:
                    self._enable_goal_obj_collision_checking(True)

                log['grasp_plan_success'] = success
                self.follow_motion_trajs(traj, gripper_state=0)
                log['grasp_execute_error'] = self.get_end_effect_error(poses)
                print("Grasp Phase End")

            elif self.cfg["solution"]["move_offset_method"] == 'cartesian_linear':
                approach = np.array(self.robot_cfg.eef_approach_axis)
                offset = approach * (self.cfg["solution"]["pre_grasp_offset"] * self.cfg["solution"]["grasp_overshoot_ratio"])
                self.follow_cartesian_linear_motion(offset, gripper_state=0)
        print("Grasp Phase End")

        self.close_gripper()
        log['grasp_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Gripper Close End")

        if self.cfg["solution"]["retract_offset"] > 0:
            offset = np.array([0, 0, self.cfg["solution"]["retract_offset"]])
            self.follow_cartesian_linear_motion(offset, gripper_state=-1, eef_frame=False)
            log['retract_finger_obj_contact'] = self.finger_goal_obj_contact()

        self.update_cuRobo_motion_gen_config(attach_goal_obj=True)

        start_time = time.time()
        traj, success, poses, results = self.motion_gen_to_free_space(mask=success)
        print("Fetch Plan", success)
        computing_time += time.time() - start_time

        self.update_cuRobo_motion_gen_config(attach_goal_obj=False)
        log['fetch_plan_success'] = success
        fetch_failure = []
        for r in results:
            if r is None:
                fetch_failure.append(r)
            else:
                fetch_failure.append(r.status)
        log['fetch_plan_failure'] = fetch_failure

        self.follow_motion_trajs(traj, gripper_state=-1)
        log['fetch_execute_error'] = self.get_end_effect_error(poses)
        print("Fetch Phase End")

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log
