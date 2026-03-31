#!/usr/bin/env python3
"""
Local tests for SWM integration.

Tests that can run WITHOUT RunPod (no checkpoints, no encoder weights needed):
1. MechJEPACostModel interface (get_cost signature, output shapes)
2. ABMPolicy interface (subclasses WorldModelPolicy correctly)
3. MechJEPA model instantiation + inference shapes
4. Import chain correctness

Run: python tests/test_swm_integration.py
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


class TestMechJEPAModel(unittest.TestCase):
    """Test that MechJEPA model shapes are correct."""

    def setUp(self):
        from mechjepa.model import MechJEPA
        self.model = MechJEPA(
            num_slots=4, slot_dim=128, num_mechanisms=8,
            history_frames=3, pred_frames=1, action_dim=2,
            transformer_depth=2, transformer_heads=4,  # small for speed
            transformer_dim_head=32, transformer_mlp_dim=256,
            edge_hidden_dim=64,
        ).eval()

    def test_inference_shape(self):
        """inference() returns (B, pred_frames, S, D)."""
        history = torch.randn(2, 3, 4, 128)
        actions = torch.randn(2, 3, 2)
        out = self.model.inference(history, actions=actions)
        self.assertEqual(out.shape, (2, 1, 4, 128))

    def test_differentiable_inference_shape(self):
        """differentiable_inference() returns same shape but with grads."""
        history = torch.randn(2, 3, 4, 128)
        actions = torch.randn(2, 3, 2)
        out = self.model.differentiable_inference(history, actions=actions)
        self.assertEqual(out.shape, (2, 1, 4, 128))
        self.assertTrue(out.requires_grad)


class TestCostModel(unittest.TestCase):
    """Test MechJEPACostModel with a mock encoder."""

    def setUp(self):
        from mechjepa.model import MechJEPA
        from mechjepa.cost_model import MechJEPACostModel

        self.model = MechJEPA(
            num_slots=4, slot_dim=128, num_mechanisms=8,
            history_frames=3, pred_frames=1, action_dim=2,
            transformer_depth=2, transformer_heads=4,
            transformer_dim_head=32, transformer_mlp_dim=256,
            edge_hidden_dim=64,
        ).eval()

        # Mock encoder that returns random slots
        class MockEncoder(nn.Module):
            def encode(self, frame_rgb):
                return torch.randn(4, 128)
        self.encoder = MockEncoder()
        self.cost_model = MechJEPACostModel(self.model, self.encoder, history_len=3)

    def test_get_cost_shape(self):
        """get_cost returns (n_envs, n_samples, 1)."""
        n_envs, n_samples, horizon = 1, 10, 5
        info_dict = {
            'pixels': np.random.randint(0, 255, (n_envs, 64, 64, 3), dtype=np.uint8),
            'goal': np.random.randint(0, 255, (n_envs, 64, 64, 3), dtype=np.uint8),
        }
        actions = torch.randn(n_envs, n_samples, horizon, 2)
        costs = self.cost_model.get_cost(info_dict, actions)
        self.assertEqual(costs.shape, (n_envs, n_samples, 1))

    def test_costs_are_positive(self):
        """Costs should be non-negative (MSE)."""
        info_dict = {
            'pixels': np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8),
            'goal': np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8),
        }
        actions = torch.randn(1, 5, 3, 2)
        costs = self.cost_model.get_cost(info_dict, actions)
        self.assertTrue((costs >= 0).all())

    def test_reset_clears_history(self):
        """reset() should clear slot and action history."""
        # build up some history
        info = {'pixels': np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8),
                'goal': np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)}
        actions = torch.randn(1, 5, 3, 2)
        self.cost_model.get_cost(info, actions)
        self.assertTrue(len(self.cost_model.slot_history) > 0)
        self.cost_model.reset()
        self.assertEqual(len(self.cost_model.slot_history), 0)

    def test_update_action_history(self):
        """update_action_history stores actions correctly."""
        action = np.array([[0.1, -0.2]])
        self.cost_model.update_action_history(action)
        self.assertEqual(len(self.cost_model.action_history), 1)


class TestABMPolicy(unittest.TestCase):
    """Test ABMPolicy interface (without SWM installed, just import checks)."""

    def test_import(self):
        """ABMPolicy should be importable."""
        try:
            from mechjepa.abm_policy import ABMPolicy
            self.assertTrue(True)
        except ImportError as e:
            if 'stable_worldmodel' in str(e):
                self.skipTest("stable_worldmodel not installed locally")
            raise

    def test_abm_policy_is_world_model_policy_subclass(self):
        """ABMPolicy should subclass WorldModelPolicy."""
        try:
            from mechjepa.abm_policy import ABMPolicy
            from stable_worldmodel.policy import WorldModelPolicy
            self.assertTrue(issubclass(ABMPolicy, WorldModelPolicy))
        except ImportError:
            self.skipTest("stable_worldmodel not installed locally")


class TestEvalScriptImports(unittest.TestCase):
    """Test that the eval script's imports are correct."""

    def test_encoder_import(self):
        """VideoSAUREncoder should be importable."""
        from mechjepa.encoder import VideoSAUREncoder
        self.assertTrue(hasattr(VideoSAUREncoder, 'from_ckpt'))
        self.assertTrue(hasattr(VideoSAUREncoder, 'encode'))

    def test_cost_model_import(self):
        """MechJEPACostModel should be importable."""
        from mechjepa.cost_model import MechJEPACostModel
        self.assertTrue(hasattr(MechJEPACostModel, 'get_cost'))
        self.assertTrue(hasattr(MechJEPACostModel, 'reset'))

    def test_model_import(self):
        """MechJEPA model should be importable."""
        from mechjepa.model import MechJEPA
        self.assertTrue(hasattr(MechJEPA, 'inference'))
        self.assertTrue(hasattr(MechJEPA, 'differentiable_inference'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
