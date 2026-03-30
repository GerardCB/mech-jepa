"""
Unit tests for MechJEPA codebook and model.

Tests shape correctness, gradient flow, and mechanism bottleneck.
Run: cd ~/GitHub/mechjepa && python -m pytest tests/ -v
"""

import torch
import pytest

from mechjepa.codebook import MechanismCodebook
from mechjepa.dynamics import MechSlotPredictor
from mechjepa.model import MechJEPA


# ── Constants ──
B, K, D, N = 4, 7, 128, 16  # batch, slots, dim, bottleneck_dim
T_HIST, T_PRED = 3, 1


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
def model():
    return MechJEPA(
        num_slots=K, slot_dim=D, num_mechanisms=N,
        history_frames=T_HIST, pred_frames=T_PRED,
        num_masked_slots=2, transformer_depth=2,
        transformer_heads=4, transformer_dim_head=32, transformer_mlp_dim=256,
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
        """h_ij should live in a lower-dimensional space than e_ij."""
        z = torch.randn(B, K, D)
        result = codebook(z)
        assert result["h_ij"].shape[-1] == N  # 16 << 128

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
        """Should work without mechanism vectors (falls back to no gating)."""
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

    def test_gradient_flows(self, predictor):
        history = torch.randn(B, T_HIST, K, D, requires_grad=True)
        m_ij = torch.randn(B, K, K, D, requires_grad=True)

        pred, _ = predictor(history, m_ij=m_ij)
        loss = pred.sum()
        loss.backward()

        assert history.grad is not None
        assert m_ij.grad is not None


# ===========================================================================
# Full Model Tests
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

        # Check individual losses exist
        for key in ["loss_jepa", "loss_future", "loss_masked_history"]:
            assert key in losses, f"Missing loss: {key}"

    def test_end_to_end_gradient(self, model):
        """Full forward + backward pass."""
        history = torch.randn(B, T_HIST, K, D)
        target = torch.randn(B, T_PRED, K, D)

        outputs = model(history)
        losses = model.compute_loss(outputs, history, target)
        losses["loss"].backward()

        # Check gradients flow to all components
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_parameter_counts(self, model):
        counts = model.get_parameter_count()
        assert counts["total_params"] > 0
        assert counts["codebook_overhead"] > 0
        assert counts["codebook_overhead"] < counts["total_params"]

    def test_diagnostics(self, model):
        model.train()
        history = torch.randn(B, T_HIST, K, D)
        _ = model(history)

        diag = model.get_diagnostics()
        assert "codebook/dim_importance" in diag


# ===========================================================================
# Integration Test
# ===========================================================================


class TestIntegration:
    def test_training_step(self, model):
        """Simulate one training step."""
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

    def test_multiple_training_steps(self, model):
        """Simulate multiple training steps to check stability."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(5):
            history = torch.randn(B, T_HIST, K, D)
            target = torch.randn(B, T_PRED, K, D)

            model.train()
            optimizer.zero_grad()

            outputs = model(history)
            losses = model.compute_loss(outputs, history, target)
            losses["loss"].backward()
            optimizer.step()

            assert losses["loss"].isfinite(), f"Loss diverged at step {step}"
