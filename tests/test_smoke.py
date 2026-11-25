import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import video_to_vrm as v2vrm  # noqa: E402


class TestVideoToVrmSmoke(unittest.TestCase):
    def test_vrm1_and_vrm0_export_validation(self):
        from PIL import Image

        H = v2vrm.WORLD_H
        params = {
            "H": H, "is_tpose": True, "torso_hw": 0.13, "leg_gap_top": 0.65,
            "hcx": 0.0, "hcz": 0.01, "rx_head": 0.095, "rz_head": 0.102,
            "y_crown": H, "y_chin": 1.311, "head_h": H - 1.311, "y_clavicle": 1.269,
            "neck_r": 0.038, "cz_neck": 0.008, "y_neck_base": 1.254,
            "shoulder_y": 1.187, "sh_x": 0.145, "cz_arm": 0.01,
            "wrist_x": 0.49, "hand_len": 0.135, "arm_tip_x": 0.625,
            "u_arm_bins": np.linspace(0.0, 1.0, 5, dtype=np.float32),
            "r_arm_bins": np.array([0.044, 0.036, 0.031, 0.026, 0.023], dtype=np.float32),
            "alpha_L": 0.0, "alpha_R": 0.0, "cz_torso": 0.005, "cz_legs": 0.002,
            "leg_ys_k": np.array([0.0, 0.087, 0.237, 0.41, 0.57, 0.76], dtype=np.float32),
            "leg_cx_L": np.array([0.072, 0.072, 0.072, 0.074, 0.076, 0.076], dtype=np.float32),
            "leg_rx_k": np.array([0.044, 0.034, 0.042, 0.038, 0.048, 0.056], dtype=np.float32),
            "leg_rz_k": np.array([0.048, 0.036, 0.044, 0.040, 0.050, 0.058], dtype=np.float32),
            "z_shoe_heel": -0.055, "z_shoe_toe": 0.095,
            "has_leg_skin": True, "leg_skin_rgb": np.array([225.0, 185.0, 165.0], dtype=np.float32),
        }

        def make_quad(y0, y1, z0):
            return {
                "verts": np.array([[-0.08, y0, z0], [0.08, y0, z0], [0.08, y1, z0], [-0.08, y1, z0]], dtype=np.float32),
                "tris": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                "uv": np.array([[0.1, 0.9], [0.9, 0.9], [0.9, 0.1], [0.1, 0.1]], dtype=np.float32),
                "normals": np.tile([0.0, 0.0, 1.0], (4, 1)).astype(np.float32),
                "part_tag": np.zeros(4, dtype=np.int32),
                "tex": np.full((v2vrm.TEX_SZ, v2vrm.TEX_SZ, 3), 180, dtype=np.uint8),
            }

        parts = {
            "head": v2vrm.build_procedural_head(params),
            "body": v2vrm.build_procedural_limbs_and_hands(params),
            "torso": make_quad(0.65, 1.27, 0.02),
            "hair": make_quad(1.30, 1.56, -0.04),
        }
        parts["head"]["tex"] = np.full((v2vrm.TEX_SZ, v2vrm.TEX_SZ, 3), 220, dtype=np.uint8)
        parts["body"]["tex"] = np.full((v2vrm.TEX_SZ, v2vrm.TEX_SZ, 3), 210, dtype=np.uint8)

        rigged, bone_pos = v2vrm.rig_avatar(parts, params)
        thumb = Image.new("RGB", (256, 256), (32, 38, 60))

        v1 = v2vrm.write_vrm_bytes(rigged, bone_pos, thumb, name="SmokeV1", author="haruka_apps", version="1.0", shader_mode="mtoon")
        v0 = v2vrm.write_vrm_bytes(rigged, bone_pos, thumb, name="SmokeV0", author="haruka_apps", version="0.x", shader_mode="mtoon")

        v2vrm.validate_vrm_spec(v1, "1.0")
        v2vrm.validate_vrm_spec(v0, "0.x")

        j1, _ = v2vrm._read_glb(v1)
        j0, _ = v2vrm._read_glb(v0)
        self.assertEqual(j1["extensions"]["VRMC_vrm"]["meta"]["authors"], ["haruka_apps"])
        self.assertEqual(j0["extensions"]["VRM"]["meta"]["author"], "haruka_apps")
        self.assertEqual(len(j1["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]), 23)
        self.assertEqual(len(j1["meshes"][1]["weights"]), 13)


if __name__ == "__main__":
    unittest.main()
