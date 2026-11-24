import random
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from scripts.train import causal_targets, matching_loss, reference_logits, supervised_logits, use_matching


class AmbientSFTTests(unittest.TestCase):
    def test_host_teacher_returns_the_same_logits_and_stays_frozen(self):
        class TinyTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 5)
                self.register_buffer("bias", torch.randn(5))

            def forward(self, tokens, use_cache=False):
                return SimpleNamespace(logits=self.embedding(tokens) + self.bias)

        teacher = TinyTeacher().eval().requires_grad_(False)
        tokens = torch.tensor([[1, 2, 3]])
        mask, _ = causal_targets(torch.tensor([[-100, -100, 3]]))
        resident = reference_logits(teacher, tokens, mask)
        offloaded = reference_logits(teacher, tokens, mask, offload=True)
        torch.testing.assert_close(resident, offloaded)
        self.assertFalse(offloaded.requires_grad)
        self.assertEqual(teacher.embedding.weight.device.type, "cpu")

    def test_causal_shift_and_supervision(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).requires_grad_()
        labels = torch.tensor([[-100, -100, 2, -100]])
        mask, targets = causal_targets(labels)
        selected = supervised_logits(logits, mask)
        self.assertEqual(targets.tolist(), [2])
        torch.testing.assert_close(selected, logits[0, 1:2].detach())
        F.cross_entropy(selected, targets).backward()
        self.assertEqual(logits.grad.abs().sum(-1).ne(0).tolist(), [[False, True, False, False]])

    def test_soft_cross_entropy_and_forward_kl_share_student_gradient(self):
        student = torch.tensor([[0.4, 1.1, -0.2]], requires_grad=True)
        teacher = torch.tensor([[1.3, 0.1, -0.9]], requires_grad=True)
        temperature = 2.5
        cross_entropy = matching_loss(student, teacher, temperature)
        cross_entropy.backward()
        self.assertIsNone(teacher.grad)
        student_gradient = student.grad.detach().clone()
        student.grad = None

        log_teacher = F.log_softmax(teacher.detach() / temperature, dim=-1)
        kl = F.kl_div(F.log_softmax(student / temperature, dim=-1), log_teacher,
                      reduction="batchmean", log_target=True)
        kl.backward()
        torch.testing.assert_close(cross_entropy - kl, -(log_teacher.exp() * log_teacher).sum())
        torch.testing.assert_close(student.grad, student_gradient)

    def test_identical_tempered_predictions_have_zero_student_gradient(self):
        student = torch.tensor([[1.2, 0.2, -1.2]], requires_grad=True)
        matching_loss(student, student.detach(), 2.5).backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student), atol=1e-7, rtol=0)

    def test_sft_does_not_consume_branch_rng(self):
        generator = random.Random(43)
        before = generator.getstate()
        self.assertFalse(use_matching(1.0, generator))
        self.assertEqual(generator.getstate(), before)

    def test_default_seed_exercises_both_branches(self):
        generator = random.Random(43)
        schedule = [use_matching(0.75, generator) for _ in range(12)]
        self.assertTrue(any(schedule) and not all(schedule))


if __name__ == "__main__":
    unittest.main()
