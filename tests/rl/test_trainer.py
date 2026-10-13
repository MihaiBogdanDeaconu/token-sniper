from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from token_sniper.rl.policy import VGIBPolicy
from token_sniper.rl.trainer import GRPOConfig, GRPOTrainer, _bernoulli_kl, load_warm_start_policy
from token_sniper.vgib_selector.backbone import GraphSAGEConfig
from token_sniper.vgib_selector.selector import VGIBSelector


def _make_selector() -> VGIBSelector:
    cfg = GraphSAGEConfig(
        num_files=3,
        source_emb_dim=32,
        file_emb_dim=8,
        hidden_dim=64,
        out_dim=32,
        num_sage_layers=2,
        dropout=0.0,
        task_emb_dim=32,
    )
    return VGIBSelector(cfg)


def _make_tasks(n: int = 4):
    from tests.rl.test_rewards import _make_task

    tasks = []
    for i in range(n):
        task = _make_task(num_nodes=8, num_files=3, num_gold=1)
        task.instance_id = f"test-{i:03d}"
        tasks.append(task)
    return tasks


class TestBernoulliKL:
    def test_identical_distributions_zero_kl(self):
        p = torch.tensor([0.3, 0.5, 0.7, 0.9])
        kl = _bernoulli_kl(p, p)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_is_non_negative(self):
        p = torch.tensor([0.2, 0.8, 0.5])
        q = torch.tensor([0.6, 0.3, 0.7])
        kl = _bernoulli_kl(p, q)
        assert kl.item() >= -1e-6

    def test_kl_is_asymmetric(self):
        p = torch.tensor([0.1, 0.9])
        q = torch.tensor([0.5, 0.5])

        kl_pq = _bernoulli_kl(p, q)
        kl_qp = _bernoulli_kl(q, p)

        assert kl_pq.item() != pytest.approx(kl_qp.item(), abs=1e-3)


class TestAdaptiveKLCoeff:
    def _make_trainer(self) -> GRPOTrainer:
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        import tempfile
        cfg = GRPOConfig(
            output_dir=Path(tempfile.mkdtemp()),
            kl_coeff=0.1,
            kl_target=0.01,
            device="cpu",
        )
        return GRPOTrainer(policy=policy, cfg=cfg)

    def test_kl_coeff_floor(self):
        trainer = self._make_trainer()
        for _ in range(100):
            trainer._adapt_kl_coeff(0.0001)
        assert trainer._kl_coeff >= 0.01

    def test_kl_coeff_ceiling(self):
        trainer = self._make_trainer()
        for _ in range(100):
            trainer._adapt_kl_coeff(0.1)
        assert trainer._kl_coeff <= 1.0

    def test_kl_coeff_stays_stable_at_target(self):
        trainer = self._make_trainer()
        initial = trainer._kl_coeff
        trainer._adapt_kl_coeff(0.01)
        assert trainer._kl_coeff == initial

    def test_kl_coeff_simulation_no_collapse(self):
        trainer = self._make_trainer()
        kl_values = [0.0001 + i * 0.0002 for i in range(50)]
        for kl in kl_values:
            trainer._adapt_kl_coeff(kl)
        assert trainer._kl_coeff >= 0.01
        assert trainer._kl_coeff * 0.01 > 1e-5


class TestGRPOTrainer:
    def test_smoke_train_2_epochs(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0)
        tasks = _make_tasks(n=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOConfig(
                output_dir=Path(tmpdir),
                group_size=2,
                epochs=2,
                patience=10,
                lr=1e-4,
                kl_coeff=0.1,
                kl_target=0.01,
                entropy_coeff=0.1,
                clip_eps=0.2,
                grad_accum_steps=2,
                reward_beta=0.5,
                reward_trace_weight=0.0,
                reward_recall_bonus=0.5,
                rollout_temperature=2.0,
                val_fraction=0.25,
                seed=42,
                device="cpu",
            )

            trainer = GRPOTrainer(policy=policy, cfg=cfg)
            best_path = trainer.train(tasks)

            assert best_path.exists(), "Best checkpoint was not saved"

    def test_checkpoint_is_loadable(self):
        selector = _make_selector()
        policy = VGIBPolicy(selector, temperature=1.0, warm_start_beta=0.1)
        tasks = _make_tasks(n=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = GRPOConfig(
                output_dir=Path(tmpdir),
                group_size=2,
                epochs=2,
                patience=10,
                lr=1e-4,
                device="cpu",
            )

            trainer = GRPOTrainer(policy=policy, cfg=cfg)
            best_path = trainer.train(tasks)

            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            assert "model_state" in ckpt
            assert "model_cfg" in ckpt
            assert ckpt["mode"] == "grpo"
            assert "epoch" in ckpt
            assert "best_val_reward" in ckpt
            assert ckpt["beta"] == 0.1

    def test_warm_start_round_trip(self):
        selector = _make_selector()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            torch.save(
                {
                    "model_state": selector.state_dict(),
                    "model_cfg": selector.cfg.__dict__,
                    "epoch": 10,
                    "best_val_loss": 0.42,
                },
                ckpt_path,
            )

            policy = load_warm_start_policy(ckpt_path, device="cpu")

            assert isinstance(policy, VGIBPolicy)
            for (n1, p1), (n2, p2) in zip(
                selector.named_parameters(),
                policy.selector.named_parameters(),
                strict=True,
            ):
                assert n1 == n2
                torch.testing.assert_close(p1, p2)


class _FakeVerified:
    def __init__(self):
        self.n_scored = 0

    def compute_many(self, items):
        from token_sniper.rl.rewards import RewardSignal

        self.n_scored += len(items)
        return [
            RewardSignal(task_reward=float(a[0] > 0.5), recall=0.0, token_ratio=0.0,
                         trace_connectivity=0.0, source="verified")
            for _, a in items
        ]


class TestRewardSource:
    def test_verified_needs_reward_object(self, tmp_path):
        cfg = GRPOConfig(output_dir=tmp_path, reward_source="verified", device="cpu")
        with pytest.raises(ValueError, match="verified_reward"):
            GRPOTrainer(VGIBPolicy(_make_selector()), cfg)

    def test_proxy_rejects_reward_object(self, tmp_path):
        cfg = GRPOConfig(output_dir=tmp_path, device="cpu")
        with pytest.raises(ValueError, match="verified_reward"):
            GRPOTrainer(VGIBPolicy(_make_selector()), cfg, verified_reward=_FakeVerified())

    def test_unknown_source(self, tmp_path):
        cfg = GRPOConfig(output_dir=tmp_path, reward_source="judge", device="cpu")
        with pytest.raises(ValueError, match="reward_source"):
            GRPOTrainer(VGIBPolicy(_make_selector()), cfg)

    def test_verified_reward_scores_rollouts_and_validation(self, tmp_path):
        fake = _FakeVerified()
        cfg = GRPOConfig(output_dir=tmp_path, group_size=2, epochs=1, grad_accum_steps=2,
                         reward_source="verified", device="cpu")
        trainer = GRPOTrainer(VGIBPolicy(_make_selector()), cfg, verified_reward=fake)
        tasks = _make_tasks(n=4)
        trainer.train(tasks[:3], val_tasks=tasks[3:])
        assert fake.n_scored == 3 * 2 + 1
