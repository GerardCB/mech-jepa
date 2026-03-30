"""
Unit tests for MechJEPA codebook and model.

Tests shape correctness, gradient flow, mechanism bottleneck,
and action conditioning (AdaLN).

Run: cd ~/GitHub/mechjepa && python -m pytest tests/ -v
"""

import torch
import torch.nn as nn
import pytest

from mechjepa.codebook import MechanismCodebook
from mechjepa.dynamics import MechSlotPredictor, ActionAdaLN
from mechjepa.model import MechJEPA


# ── Constants ──
B, K, D, N = 4, 7, 128, 16  # batch, slots, dim, bottleneck_dim
T_HIST, T_PRED = 3, 1
ACTION_DIM = 2
K_PUSHT = 4  # Push-T uses 4 slots


@pytest.fixture
def codebook():
    return MechanismCodebook(num_mechanisms=N, slot_dim=D)


@pytest.fixture
def predictor():
    return MechSlotPredictor(
        num_slots=K, slot_dim=D, history_frames=T_HIST, pred_frames=T_PRED,
        num_masked_slots=2, depth=2, heads=4, dim_head=32, mlp_dim=256,
    )


@pytest.fixture
def predictor_with_actions():
    return MechSlotPredictor(
        num_slots=K_PUSHT, slot_dim=D, history_frames=T_HIST, pred_frames=T_PRED,
        num_masked_slots=2, depth=2, heads=4, dim_head=32, mlp_dim=256,
        action_dim=ACTION_DIM,
    )


@pytest.fixture
def model():
    return MechJEPA(
        num_slots=K, slot_dim=D, num_mechanisms=N,
        history_frames=T_HIST, pred_frames=T_PRED,
        num_masked_slots=2, transformer_depth=2,
        transformer_heads=4, transformer_dim_head=32, transformer_mlp_dim=256,
    )


@pytest.fixture
def model_pusht():
    return MechJEPA(
        num_slots=K_PUSHT, slot_dim=D, num_mechanisms=8,
        history_frames=T_HIST, pred_frames=T_PRED,
        num_masked_slots=2, transformer_depth=2,
        transformer_heads=4, transformer_dim_head=32, transformer_mlp_dim=256,
        action_dim=ACTION_DIM,
    )


# ===========================================================================
# Codebook (Bottleneck) Tests
# ===========================================================================


class TestCodebook:
    def test_edge_shapes(self, codebook):
        z = torch.randn(B, K, D)
        e_ij = codebook.compute_edges(z)
        assert e_ij.shape == (B, K, K, D)

    def test_binding_shapes(self, codebook):
        z = torch.randn(B, K, D)
        result = codebook(z)

        assert result["m_ij"].shape == (B, K, K, D)
        assert result["h_ij"].shape == (B, K, K, N)
        assert result["e_ij"].shape == (B, K, K, D)

    def test_bottleneck_compresses(self, codebook):
        z = torch.randn(B, K, D)
        result = codebook(z)
        assert result["h_ij"].shape[-1] == N

    def test_gradient_flows_through_bottleneck(self, codebook):
        z = torch.randn(B, K, D, requires_grad=True)
        result = codebook(z)
        loss = result["m_ij"].sum()
        loss.backward()
        assert z.grad is not None
        assert codebook.encode.weight.grad is not None
        assert codebook.decode.weight.grad is not None

    def test_diagnostics(self, codebook):
        stats = codebook.get_codebook_stats()
        assert "codebook/dim_importance" in stats
        assert stats["codebook/dim_importance"].shape == (N,)


# ===========================================================================
# ActionAdaLN Tests
# ===========================================================================


class TestActionAdaLN:
    def test_identity_without_action(self):
        adaln = ActionAdaLN(D)
        x = torch.randn(B, 10, D)
        out = adaln(x, action_emb=None)
        assert torch.equal(out, x)

    def test_shape_with_action(self):
        adaln = ActionAdaLN(D)
        x = torch.randn(B, 10, D)
        action_emb = torch.randn(B, 10, D)
        out = adaln(x, action_emb=action_emb)
        assert out.shape == x.shape

    def test_starts_as_identity(self):
        """AdaLN initializes to identity (scale=1, shift=0)."""
        adaln = ActionAdaLN(D)
        x = torch.randn(B, 10, D)
        action_emb = torch.zeros(B, 10, D)  # zero action
        out = adaln(x, action_emb=action_emb)
        assert torch.allclose(out, x, atol=1e-5)

    def test_gradient_flows(self):
        adaln = ActionAdaLN(D)
        x = torch.randn(B, 10, D, requires_grad=True)
        action_emb = torch.randn(B, 10, D, requires_grad=True)
        out = adaln(x, action_emb)
        out.sum().backward()
        assert x.grad is not None
        assert action_emb.grad is not None


# ===========================================================================
# Predictor Tests
# ===========================================================================


class TestPredictor:
    def test_forward_shapes(self, predictor):
        predictor.eval()
        history = torch.randn(B, T_HIST, K, D)
        m_ij = torch.randn(B, K, K, D)

        pred, mask_indices = predictor(history, m_ij=m_ij)
        T_total = T_HIST + T_PRED
        assert pred.shape == (B, T_total, K, D)
        assert len(mask_indices) <= 2

    def test_forward_without_mechanism(self, predictor):
        history = torch.randn(B, T_HIST, K, D)
        pred, _ = predictor(history, m_ij=None)
        assert pred.shape == (B, T_HIST + T_PRED, K, D)

    def test_inference_shapes(self, predictor):
        history = torch.randn(B, T_HIST, K, D)
        m_ij = torch.randn(B, K, K, D)
        future = predictor.inference(history, m_ij=m_ij)
        assert future.shape == (B, T_PRED, K, D)

    def test_attention_extraction(self, predictor):
        history = torch.randn(B, T_HIST, K, D)
        m_ij = torch.randn(B, K, K, D)
        pred, mask_indices, attn_list = predictor(
            history, m_ij=m_ij, return_attention=True,
        )
        assert attn_list is not None
        assert len(attn_list) > 0


# ===========================================================================
# Action-Conditioned Predictor Tests
# ===========================================================================


class TestPredictorWithActions:
    def test_forward_with_actions(self, predictor_with_actions):
        predictor_with_actions.eval()
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        m_ij = torch.randn(B, K_PUSHT, K_PUSHT, D)
        actions = torch.randn(B, T_HIST, ACTION_DIM)

        pred, mask_indices = predictor_with_actions(
            history, m_ij=m_ij, actions=actions
        )
        assert pred.shape == (B, T_HIST + T_PRED, K_PUSHT, D)

    def test_forward_without_actions_backward_compat(self, predictor_with_actions):
        """Action-capable predictor should work with actions=None."""
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        pred, _ = predictor_with_actions(history, m_ij=None, actions=None)
        assert pred.shape == (B, T_HIST + T_PRED, K_PUSHT, D)

    def test_actions_change_output(self, predictor_with_actions):
        """Different actions should produce different predictions after training."""
        # AdaLN initializes to identity (zero weights), so we need to break
        # that symmetry first. Manually set non-zero AdaLN weights.
        with torch.no_grad():
            for layer in predictor_with_actions.transformer.layers:
                attn, ffn = layer
                nn.init.normal_(attn.adaln.modulation[-1].weight, std=0.1)
                nn.init.normal_(ffn.adaln.modulation[-1].weight, std=0.1)

        predictor_with_actions.eval()
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        m_ij = torch.randn(B, K_PUSHT, K_PUSHT, D)

        actions_a = torch.randn(B, T_HIST, ACTION_DIM)
        actions_b = torch.randn(B, T_HIST, ACTION_DIM) * 5  # very different

        pred_a, _ = predictor_with_actions(history, m_ij=m_ij, actions=actions_a)
        pred_b, _ = predictor_with_actions(history, m_ij=m_ij, actions=actions_b)

        # With non-zero AdaLN weights, different actions should give different outputs
        assert not torch.allclose(pred_a, pred_b, atol=1e-3)

    def test_inference_with_actions(self, predictor_with_actions):
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        m_ij = torch.randn(B, K_PUSHT, K_PUSHT, D)
        actions = torch.randn(B, T_HIST, ACTION_DIM)

        future = predictor_with_actions.inference(
            history, m_ij=m_ij, actions=actions
        )
        assert future.shape == (B, T_PRED, K_PUSHT, D)

    def test_gradient_flows_through_actions(self, predictor_with_actions):
        history = torch.randn(B, T_HIST, K_PUSHT, D, requires_grad=True)
        actions = torch.randn(B, T_HIST, ACTION_DIM, requires_grad=True)

        pred, _ = predictor_with_actions(history, m_ij=None, actions=actions)
        loss = pred.sum()
        loss.backward()

        assert history.grad is not None
        assert actions.grad is not None


# ===========================================================================
# Full Model Tests (CLEVRER — backward compat)
# ===========================================================================


class TestMechJEPA:
    def test_forward_shapes(self, model):
        history = torch.randn(B, T_HIST, K, D)
        outputs = model(history)
        assert outputs["pred_embedding"].shape == (B, T_HIST + T_PRED, K, D)
        assert outputs["codebook_output"]["h_ij"].shape == (B, K, K, N)

    def test_inference_shapes(self, model):
        history = torch.randn(B, T_HIST, K, D)
        future = model.inference(history)
        assert future.shape == (B, T_PRED, K, D)

    def test_loss_computation(self, model):
        history = torch.randn(B, T_HIST, K, D)
        target = torch.randn(B, T_PRED, K, D)
        outputs = model(history)
        losses = model.compute_loss(outputs, history, target)
        assert "loss" in losses
        assert losses["loss"].isfinite()
        assert losses["loss"].requires_grad

    def test_end_to_end_gradient(self, model):
        history = torch.randn(B, T_HIST, K, D)
        target = torch.randn(B, T_PRED, K, D)
        outputs = model(history)
        losses = model.compute_loss(outputs, history, target)
        losses["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                # AdaLN params don't get gradients when actions=None (expected)
                if "adaln" in name:
                    continue
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_counts(self, model):
        counts = model.get_parameter_count()
        assert counts["total_params"] > 0

    def test_diagnostics(self, model):
        history = torch.randn(B, T_HIST, K, D)
        _ = model(history)
        diag = model.get_diagnostics()
        assert "codebook/dim_importance" in diag


# ===========================================================================
# Full Model Tests (Push-T — action-conditioned)
# ===========================================================================


class TestMechJEPAPushT:
    def test_forward_with_actions(self, model_pusht):
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        actions = torch.randn(B, T_HIST, ACTION_DIM)
        outputs = model_pusht(history, actions=actions)
        assert outputs["pred_embedding"].shape == (B, T_HIST + T_PRED, K_PUSHT, D)

    def test_forward_without_actions(self, model_pusht):
        """Action-capable model should work without actions."""
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        outputs = model_pusht(history)
        assert outputs["pred_embedding"].shape == (B, T_HIST + T_PRED, K_PUSHT, D)

    def test_inference_with_actions(self, model_pusht):
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        actions = torch.randn(B, T_HIST, ACTION_DIM)
        future = model_pusht.inference(history, actions=actions)
        assert future.shape == (B, T_PRED, K_PUSHT, D)

    def test_loss_with_actions(self, model_pusht):
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        target = torch.randn(B, T_PRED, K_PUSHT, D)
        actions = torch.randn(B, T_HIST, ACTION_DIM)

        outputs = model_pusht(history, actions=actions)
        losses = model_pusht.compute_loss(outputs, history, target)
        assert losses["loss"].isfinite()
        assert losses["loss"].requires_grad

    def test_end_to_end_gradient_with_actions(self, model_pusht):
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        target = torch.randn(B, T_PRED, K_PUSHT, D)
        actions = torch.randn(B, T_HIST, ACTION_DIM)

        outputs = model_pusht(history, actions=actions)
        losses = model_pusht.compute_loss(outputs, history, target)
        losses["loss"].backward()

        for name, param in model_pusht.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_action_embedder_exists(self, model_pusht):
        """Check that action embedder was created."""
        assert model_pusht.predictor.transformer.action_embedder is not None
        assert model_pusht.action_dim == ACTION_DIM


# ===========================================================================
# Integration Test
# ===========================================================================


class TestIntegration:
    def test_clevrer_training_step(self, model):
        """Simulate one CLEVRER training step (no actions)."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        history = torch.randn(B, T_HIST, K, D)
        target = torch.randn(B, T_PRED, K, D)

        model.train()
        optimizer.zero_grad()
        outputs = model(history)
        losses = model.compute_loss(outputs, history, target)
        losses["loss"].backward()
        optimizer.step()
        assert losses["loss"].isfinite()

    def test_pusht_training_step(self, model_pusht):
        """Simulate one Push-T training step (with actions)."""
        optimizer = torch.optim.AdamW(model_pusht.parameters(), lr=1e-4)
        history = torch.randn(B, T_HIST, K_PUSHT, D)
        target = torch.randn(B, T_PRED, K_PUSHT, D)
        actions = torch.randn(B, T_HIST, ACTION_DIM)

        model_pusht.train()
        optimizer.zero_grad()
        outputs = model_pusht(history, actions=actions)
        losses = model_pusht.compute_loss(outputs, history, target)
        losses["loss"].backward()
        optimizer.step()
        assert losses["loss"].isfinite()

    def test_multiple_pusht_steps(self, model_pusht):
        """Multiple Push-T training steps for stability."""
        optimizer = torch.optim.AdamW(model_pusht.parameters(), lr=1e-4)

        for step in range(5):
            history = torch.randn(B, T_HIST, K_PUSHT, D)
            target = torch.randn(B, T_PRED, K_PUSHT, D)
            actions = torch.randn(B, T_HIST, ACTION_DIM)

            model_pusht.train()
            optimizer.zero_grad()
            outputs = model_pusht(history, actions=actions)
            losses = model_pusht.compute_loss(outputs, history, target)
            losses["loss"].backward()
            optimizer.step()
            assert losses["loss"].isfinite(), f"Loss diverged at step {step}"
