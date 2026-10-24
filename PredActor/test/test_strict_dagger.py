from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dagger.branching import BranchWindow
from dagger.command_map import MotionCommandMap, validate_command_geometry
from dagger.collector import CollectionConfig, StrictDaggerCollector
from dagger.dataset import _weighted_counts
from dagger.eval_collector import WalkRolloutCollector
from dagger.manifest import DaggerManifest
from dagger.metrics import command_distinguishability, straight_walk_metrics
from dagger.policy_benchmark import (
    PolicyBenchmarkSuite,
    cosine_top1,
    first_complete_motion_window,
    rollout_survival,
)
from dagger.text_commands import command_id_for_text
from diffusion_policy.task_provider.fixed_embedding import FixedEmbeddingProvider


class StrictDaggerTests(unittest.TestCase):
    def test_fixed_embedding_provider_uses_named_archive_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prototypes.npz"
            np.savez_compressed(
                path,
                commands=np.asarray(["walk straight", "turn left while walking"]),
                embeddings=np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32),
            )
            provider = FixedEmbeddingProvider(str(path), "walk straight")
            terms = {"root_pos": torch.zeros(2, 1, 3)}
            provider.inject_raw_keys(terms, batch_size=2)
            self.assertEqual(tuple(terms["motion_latent"].shape), (2, 1, 2))
            np.testing.assert_allclose(
                terms["motion_latent"][0, 0].numpy(), [0.6, 0.8], atol=1e-6
            )

    def test_bare_walk_keeps_its_exact_text_embedding_id(self):
        self.assertEqual(command_id_for_text("walk"), 0)
        self.assertEqual(command_id_for_text("walking"), 1)

    def test_collector_keeps_teacher_branch_out_of_live_simulator(self):
        class Simulator:
            def __init__(self, value=0.0):
                self.value = float(value)
                self._terminated = False

            def reset(self):
                self.value = 0.0
                return self.observe()

            def observe(self):
                return {"root_pos": np.asarray([self.value, 0.0, 0.75])}

            def snapshot(self):
                return self.value

            def fork(self, snapshot):
                return Simulator(snapshot)

            def step(self, action):
                self.value += float(np.asarray(action).item())
                return self.observe()

            @property
            def terminated(self):
                return self._terminated

        live = Simulator()
        collector = StrictDaggerCollector(
            CollectionConfig(
                horizon=5, n_obs_steps=2, cooldown_steps=10,
                max_branches_per_episode=1,
            ),
            learner=lambda state: np.asarray([1.0]),
            teacher=lambda state: np.asarray([2.0]),
            risk_trigger=lambda prefix: True,
        )
        branches = collector.collect_episode(
            live, command_id=1, max_steps=3
        )
        self.assertEqual(len(branches), 1)
        self.assertEqual(live.value, 3.0)
        window = branches[0].assemble(horizon=5, n_obs_steps=2)
        np.testing.assert_array_equal(window["act"].reshape(-1), [1, 2, 2, 2, 2])
        np.testing.assert_array_equal(window["action_valid"], [False] + [True] * 4)

    def test_branch_masks_learner_history_actions(self):
        prefix = {"root_pos": np.arange(12, dtype=np.float32).reshape(4, 3)}
        teacher = {"root_pos": np.concatenate([
            prefix["root_pos"][-1:],
            np.ones((16, 3), dtype=np.float32),
        ])}
        window = BranchWindow(
            student_prefix=prefix,
            student_actions=np.full((3, 29), 7.0, dtype=np.float32),
            teacher_states=teacher,
            teacher_actions=np.ones((17, 29), dtype=np.float32),
            command_id=1,
        ).assemble(horizon=20, n_obs_steps=4)
        self.assertEqual(window["root_pos"].shape, (20, 3))
        self.assertEqual(window["act"].shape, (20, 29))
        np.testing.assert_array_equal(
            window["action_valid"], [False, False, False] + [True] * 17
        )
        np.testing.assert_array_equal(window["act"][:3], 7.0)

    def test_weighted_counts_are_exact(self):
        self.assertEqual(_weighted_counts([0.6, 0.25, 0.1, 0.05], 100), [60, 25, 10, 5])

    def test_motion_command_map_allows_reviewed_one_to_many_exact_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.yaml"
            path.write_text(
                """schema: predactor-motion-command-map-v1
motions:
  - motion_idx: 34
    motion_name: walk1
    commands: [walk, walk straight]
    reviewed: true
    review_basis: filename_and_reference_kinematics
    reason: sustained forward walk
""",
                encoding="utf-8",
            )
            command_map = MotionCommandMap.load(path)
            self.assertEqual(
                command_map.commands_by_motion_idx[34],
                ("walk", "walk straight"),
            )

    def test_motion_command_map_rejects_aliases_and_unreviewed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commands.yaml"
            path.write_text(
                """schema: predactor-motion-command-map-v1
motions:
  - motion_idx: 1
    motion_name: walking
    commands: [walking]
    reviewed: false
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-canonical"):
                MotionCommandMap.load(path)

    def test_command_geometry_rejects_directionally_contradictory_label(self):
        metrics = {
            "forward_displacement_m": 1.0,
            "lateral_displacement_m": 0.0,
            "net_yaw_rad": 0.7,
        }
        validate_command_geometry("turn left while walking", metrics)
        with self.assertRaisesRegex(ValueError, "contradict"):
            validate_command_geometry("turn right while walking", metrics)

    def test_manifest_rejects_beta_mixing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            path.write_text(
                """schema: predactor-strict-dagger-v1
normalizer_checkpoint: pi0.ckpt
sources:
  - {name: d0, path: d0.zarr, role: d0, weight: 0.6}
  - {name: mixed, path: mixed.zarr, role: beta_mixed, weight: 0.4}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "intentionally forbidden"):
                DaggerManifest.load(path, require_files=False)

    def test_straight_metrics_use_initial_heading_frame(self):
        root_pos = np.stack([np.linspace(0, 2, 101), np.zeros(101), np.zeros(101)], axis=1)
        root_rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (101, 1))
        result = straight_walk_metrics(root_pos, root_rot, dt=0.1)
        self.assertAlmostEqual(result.forward_distance_m, 2.0)
        self.assertAlmostEqual(result.abs_lateral_displacement_m, 0.0)
        self.assertAlmostEqual(result.cumulative_abs_delta_yaw_rad, 0.0)

    def test_straight_metrics_reject_invalid_foot_quaternions(self):
        root_pos = np.zeros((11, 3), dtype=np.float64)
        root_rot = np.zeros((11, 4), dtype=np.float64)
        root_rot[:, 0] = 1.0
        invalid_foot_rot = np.zeros((11, 4), dtype=np.float64)
        result = straight_walk_metrics(
            root_pos, root_rot, dt=0.02,
            left_foot_rot=invalid_foot_rot,
            right_foot_rot=invalid_foot_rot,
        )
        self.assertTrue(np.isnan(result.mean_abs_foot_yaw_difference_rad))

    def test_command_distinguishability_uses_signed_net_yaw(self):
        def rollout(yaw_end):
            yaw = np.linspace(0.0, yaw_end, 11)
            rotation = np.stack([
                np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw),
                np.sin(yaw / 2),
            ], axis=1)
            position = np.stack([
                np.linspace(0.0, 1.0, 11), np.zeros(11), np.zeros(11)
            ], axis=1)
            return straight_walk_metrics(position, rotation, dt=0.1)

        result = command_distinguishability([
            ("walk straight", rollout(0.02)),
            ("turn left while walking", rollout(0.4)),
            ("turn right while walking", rollout(-0.4)),
        ])
        self.assertEqual(result.accuracy, 1.0)
        self.assertGreater(result.ordering_margin_rad, 0.0)

    def test_walk_rollout_collector_exports_initial_and_terminal_states(self):
        def history(x):
            root_pos = np.asarray([[[x, 0.0, 0.75]]], dtype=np.float32)
            root_rot = np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
            body_rot = np.zeros((1, 1, 30, 4), dtype=np.float32)
            body_rot[..., 0] = 1.0
            return {
                "root_pos": root_pos,
                "root_rot": root_rot,
                "body_rot": body_rot,
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "walk_seed42.npz"
            collector = WalkRolloutCollector(
                output, "walk", dt=0.02, seed=42, requested_duration_s=1.0
            )
            collector.start(history(0.0))
            step_data = collector.collect_step_data(
                raw_history=history(0.1),
                actions=np.zeros((1, 29)),
                inference_time_ms=0.0,
                obs_time_ms=0.0,
                step_time_ms=0.0,
                reward=0.0,
                done=True,
            )
            self.assertFalse(collector.step(step_data))
            collector.flush_episode()
            result = collector.finalize()
            archive = np.load(output, allow_pickle=False)
            self.assertEqual(archive["root_pos"].shape, (2, 3))
            self.assertEqual(str(archive["command"][0]), "walk")
            self.assertEqual(int(archive["seed"]), 42)
            np.testing.assert_array_equal(archive["terminated"], [False, True])
            np.testing.assert_array_equal(
                archive["termination_reason"], ["", "environment"]
            )
            self.assertAlmostEqual(result["metric_results"][0].survival_fraction, 0.02)
            self.assertFalse(result["metric_results"][0].completed_horizon)

    def test_policy_benchmark_suite_has_45_text_cells(self):
        suite_path = (
            Path(__file__).resolve().parents[1]
            / "dagger" / "configs" / "policy_benchmark_text9.yaml"
        )
        suite = PolicyBenchmarkSuite.load(suite_path)
        self.assertEqual(len(suite.text_commands) * len(suite.seeds), 45)
        self.assertEqual(len(suite.navigation_commands) * len(suite.seeds), 15)

    def test_retrieval_window_requires_complete_post_delay_motion(self):
        complete = {
            "body_pos": np.zeros((251, 30, 3), dtype=np.float32),
            "dt": np.asarray(0.02),
        }
        window = first_complete_motion_window(
            complete,
            activation_delay_seconds=3.0,
            window_frames=60,
            target_fps=30.0,
        )
        self.assertEqual(window.shape, (60, 30, 3))
        incomplete = {
            "body_pos": np.zeros((249, 30, 3), dtype=np.float32),
            "dt": np.asarray(0.02),
        }
        self.assertIsNone(first_complete_motion_window(
            incomplete,
            activation_delay_seconds=3.0,
            window_frames=60,
            target_fps=30.0,
        ))

    def test_policy_survival_is_observed_fraction_not_binary(self):
        archive = {
            "root_pos": np.zeros((251, 3)),
            "terminated": np.asarray([False] * 250 + [True]),
            "dt": np.asarray(0.02),
            "requested_duration_s": np.asarray(10.0),
        }
        result = rollout_survival(archive, 10.0)
        self.assertAlmostEqual(result["survival_fraction"], 0.5)
        self.assertFalse(result["completed_horizon"])

    def test_paper_terminal_gate_catches_low_root_before_env_reset(self):
        history = {
            "root_pos": np.asarray([[[0.0, 0.0, 0.54]]], dtype=np.float32),
            "root_rot": np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
            "base_ang_vel": np.zeros((1, 1, 3), dtype=np.float32),
        }
        self.assertTrue(WalkRolloutCollector._paper_terminal(history))
        self.assertEqual(
            WalkRolloutCollector._paper_terminal_reason(history), "root_height"
        )

    def test_cosine_top1_scores_raw_prompt_candidates(self):
        predicted, scores = cosine_top1(
            np.asarray([[0.9, 0.1], [0.0, 1.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        )
        np.testing.assert_array_equal(predicted, [0, 1])
        self.assertEqual(scores.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
