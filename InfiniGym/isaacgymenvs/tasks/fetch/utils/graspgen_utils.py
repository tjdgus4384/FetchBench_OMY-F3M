"""
GraspGen wrapper for FetchBench via ZMQ client.

Follows the same pattern as demo_object_mesh.py:
  1. Pre-center point cloud (subtract mean)
  2. Send centered point cloud to GraspGen server
  3. Uncenter output grasps (add mean back)

This avoids mismatch between our mean and GraspGen's internal mean
(which is computed after outlier removal).

Start the server first:
    conda activate graspgen
    python -c "
    import logging; logging.basicConfig(level=logging.INFO)
    from grasp_gen.serving.zmq_server import GraspGenZMQServer
    server = GraspGenZMQServer(
        gripper_config='/path/to/graspgen_franka_panda.yml', port=5556)
    server.serve_forever()
    "
"""

import numpy as np
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


class GraspGenWrapper:
    def __init__(self, host="localhost", port=5556, num_grasps=200, topk=100):
        self.num_grasps = num_grasps
        self.topk = topk

        self._addr = f"tcp://{host}:{port}"
        self._ctx = zmq.Context()
        self._socket = None
        self._connect()

    def _connect(self):
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, 60000)
        self._socket.setsockopt(zmq.SNDTIMEO, 60000)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._addr)

        self._socket.send(msgpack.packb({"action": "health"}, use_bin_type=True))
        resp = msgpack.unpackb(self._socket.recv(), raw=False)
        if resp.get("status") != "ok":
            raise RuntimeError(f"GraspGen server not healthy: {resp}")
        print(f"Connected to GraspGen server at {self._addr}")

    def _request(self, payload):
        self._socket.send(msgpack.packb(payload, use_bin_type=True))
        raw = self._socket.recv()
        response = msgpack.unpackb(raw, raw=False)
        if "error" in response:
            raise RuntimeError(f"GraspGen server error: {response['error']}")
        return response

    def predict(self, object_pc):
        """Run GraspGen on a single object point cloud.

        Follows demo_object_mesh.py pattern: pre-center, infer, uncenter.

        Args:
            object_pc: np.array (N, 3) in any frame.

        Returns:
            grasp_poses: np.array (M, 4, 4) in the same frame as input.
            scores: np.array (M,) confidence scores.
        """
        if len(object_pc) < 1:
            return np.empty((0, 4, 4)), np.empty((0,))

        object_pc = np.asarray(object_pc, dtype=np.float32)

        # Step 1: Pre-center (same as demo_object_mesh.py)
        pc_mean = object_pc.mean(axis=0)
        centered_pc = object_pc - pc_mean

        # Step 2: Send centered point cloud to server
        # GraspGen's internal centering becomes a no-op since mean ≈ [0,0,0]
        response = self._request({
            "action": "infer",
            "point_cloud": centered_pc,
            "num_grasps": self.num_grasps,
            "topk_num_grasps": self.topk,
            "grasp_threshold": -1.0,
            "remove_outliers": True,
        })

        grasps = np.asarray(response["grasps"], dtype=np.float32)
        scores = np.asarray(response["confidences"], dtype=np.float32)

        if len(grasps) == 0:
            return np.empty((0, 4, 4)), np.empty((0,))

        # Step 3: Uncenter grasps back to original frame
        T_uncenter = np.eye(4, dtype=np.float32)
        T_uncenter[:3, 3] = pc_mean
        grasps = np.array([T_uncenter @ g for g in grasps])

        return grasps, scores

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._ctx.term()
