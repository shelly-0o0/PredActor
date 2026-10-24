from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dagger.canonical_reference import (
    canonical_root_path,
    quaternion_angular_velocity_wxyz,
)
from dagger.canonical_dataset import _audited_episode_slices
from dagger.canonical_teacher import (
    CanonicalTeacherTrainer,
    excluded_motion_ids_for_commands,
)
from dagger.command_map import MotionCommandMap


class CanonicalTeacherTests(unittest.TestCase):
    def test_teacher_builder_selects_only_audit_passed_episodes(self):
        commands = {
            0: "walk straight",
            1: "turn left while walking",
            2: "turn right while walking",
        }
        reports = []
        for episode, (start, end, motion_idx, passed) in enumerate((
            (0, 10, 0, False),
            (10, 20, 0, True),
            (20, 30, 1, True),
            (30, 40, 2, True),
        )):
            reports.append({
                "episode": episode,
                "start": start,
                "end": end,
                "motion_idx": motion_idx,
                "command": commands[motion_idx],
                "passed": passed,
                "failure_reasons": [] if passed else ["test_rejection"],
            })
        selected, dropped = _audited_episode_slices(
            {"episode_reports": reports},
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            commands,
        )
        self.assertEqual(selected, [(10, 20, 0), (20, 30, 1), (30, 40, 2)])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["failure_reasons"], ["test_rejection"])

    def test_canonical_left_and_right_paths_are_exact_mirrors(self):
        left_xy, left_yaw = canonical_root_path(
            frames=501, duration_s=10.0, distance_m=7.0, turn_angle_rad=1.0
        )
        right_xy, right_yaw = canonical_root_path(
            frames=501, duration_s=10.0, distance_m=7.0, turn_angle_rad=-1.0
        )
        np.testing.assert_allclose(left_xy[:, 0], right_xy[:, 0], atol=1e-12)
        np.testing.assert_allclose(left_xy[:, 1], -right_xy[:, 1], atol=1e-12)
        np.testing.assert_allclose(left_yaw, -right_yaw, atol=1e-12)
        self.assertAlmostEqual(left_yaw[-1], 1.0)
        self.assertAlmostEqual(right_yaw[-1], -1.0)

    def test_quaternion_velocity_recovers_constant_yaw_rate(self):
        dt = 0.02
        yaw = np.arange(101) * dt * 0.4
        quaternion = np.zeros((101, 1, 4), dtype=np.float64)
        quaternion[:, 0, 0] = np.cos(yaw / 2.0)
        quaternion[:, 0, 3] = np.sin(yaw / 2.0)
        velocity = quaternion_angular_velocity_wxyz(quaternion, dt)
        np.testing.assert_allclose(velocity[..., :2], 0.0, atol=1e-10)
        np.testing.assert_allclose(velocity[..., 2], 0.4, atol=1e-10)

    def test_d0_filter_excludes_reviewed_navigation_commands_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.yaml"
            path.write_text(
                """schema: predactor-motion-command-map-v1
motions:
  - motion_idx: 34
    motion_name: straight
    commands: [walk, walk straight]
    reviewed: true
    review_basis: test
    reason: test
  - motion_idx: 36
    motion_name: backward
    commands: [walk backward]
    reviewed: true
    review_basis: test
    reason: test
  - motion_idx: 52
    motion_name: left
    commands: [turn left while walking]
    reviewed: true
    review_basis: test
    reason: test
""",
                encoding="utf-8",
            )
            command_map = MotionCommandMap.load(path)
            self.assertEqual(
                excluded_motion_ids_for_commands(command_map), (34, 52)
            )

    def test_trainable_scope_is_condition_and_cross_attention_only(self):
        selected = {
            "backbone.cond_pos_emb",
            "backbone.cond_obs_emb.weight",
            "backbone.encoder.0.weight",
            "backbone.decoder.layers.0.multihead_attn.in_proj_weight",
            "backbone.decoder.layers.0.norm2.weight",
        }
        rejected = {
            "backbone.x_emb.weight",
            "backbone.y_emb.weight",
            "backbone.decoder.layers.0.self_attn.in_proj_weight",
            "backbone.decoder.layers.0.linear1.weight",
            "backbone.x_head.weight",
            "backbone.y_head.weight",
        }
        self.assertTrue(all(
            CanonicalTeacherTrainer._condition_parameter(name)
            for name in selected
        ))
        self.assertTrue(all(
            not CanonicalTeacherTrainer._condition_parameter(name)
            for name in rejected
        ))


if __name__ == "__main__":
    unittest.main()
