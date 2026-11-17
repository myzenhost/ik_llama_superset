#!/usr/bin/env python3
"""Tests for moe_dynamic_recipe.py: every rule must be reachable, ordered, and parseable by llama-quantize."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_dynamic_recipe as R  # noqa: E402

# DeepSeek-V3-shaped: 61 layers, 3 dense, MLA attention, shared expert, routed experts, router
def ds3_names(n_layer=61, first_dense=3):
    names = ["token_embd.weight", "output.weight"]
    for il in range(n_layer):
        for a in ("attn_q_a", "attn_q_b", "attn_kv_a_mqa", "attn_k_b", "attn_v_b", "attn_output"):
            names.append(f"blk.{il}.{a}.weight")
        if il < first_dense:
            names += [f"blk.{il}.ffn_{x}.weight" for x in ("gate", "up", "down")]
        else:
            names += [f"blk.{il}.ffn_{x}_exps.weight" for x in ("gate", "up", "down")]
            names += [f"blk.{il}.ffn_{x}_shexp.weight" for x in ("gate", "up", "down")]
            names += [f"blk.{il}.ffn_gate_inp.weight", f"blk.{il}.exp_probs_b.bias"]
    return names


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.t = dict(R.TIERS["iq1"])
        self.rules = R.build_rules(61, 3, 2, 2, self.t)

    def r(self, name):
        return R.resolve(self.rules, name)

    def test_router_is_full_precision(self):
        self.assertEqual(self.r("blk.10.ffn_gate_inp.weight"), "f32")
        self.assertEqual(self.r("blk.10.exp_probs_b.bias"), "f32")

    def test_routed_experts_get_the_low_tier_and_down_is_one_step_up(self):
        self.assertEqual(self.r("blk.20.ffn_gate_exps.weight"), self.t["low"])
        self.assertEqual(self.r("blk.20.ffn_up_exps.weight"), self.t["low"])
        self.assertEqual(self.r("blk.20.ffn_down_exps.weight"), self.t["down"])

    def test_boundary_layers_beat_the_low_tier(self):
        # layers 3,4 (first MoE layers) and 59,60 (last)
        for il in (3, 4, 59, 60):
            for x in ("gate", "up", "down"):
                self.assertEqual(self.r(f"blk.{il}.ffn_{x}_exps.weight"), self.t["boundary"], (il, x))
        self.assertEqual(self.r("blk.5.ffn_gate_exps.weight"), self.t["low"])
        self.assertEqual(self.r("blk.58.ffn_gate_exps.weight"), self.t["low"])

    def test_dense_shared_attention_embeddings(self):
        self.assertEqual(self.r("blk.1.ffn_up.weight"), self.t["dense"])
        self.assertEqual(self.r("blk.30.ffn_down_shexp.weight"), self.t["shared"])
        self.assertEqual(self.r("blk.30.attn_kv_a_mqa.weight"), self.t["attn"])
        self.assertEqual(self.r("blk.30.attn_output.weight"), self.t["attn"])
        self.assertEqual(self.r("token_embd.weight"), self.t["embed"])
        self.assertEqual(self.r("output.weight"), self.t["output"])

    def test_every_2d_tensor_of_a_ds3_shaped_model_is_covered_and_bytes_sit_in_the_low_tier(self):
        names = ds3_names()
        counts = {}
        for n in names:
            ty = self.r(n)
            self.assertIsNotNone(ty, n)
            counts[ty] = counts.get(ty, 0) + 1
        # 58 MoE layers: 4 are boundary, 54 are low (gate+up) / down
        self.assertEqual(counts[self.t["low"]], 2 * (58 - 4))
        self.assertEqual(counts[self.t["down"]], 58 - 4)

    def test_custom_q_string_is_parseable_by_llama_quantize(self):
        s = R.to_custom_q(self.rules)
        items = s.split(",")
        self.assertEqual(len(items), len(self.rules))
        for it in items:
            rx, ty = it.split("=", 1)       # llama-quantize splits at the FIRST '='
            re.compile(rx)                  # must compile
            self.assertTrue(ty)

    def test_narrow_types_can_be_selected_for_the_routed_experts(self):
        t = dict(self.t, low="iq1_xs", down="iq1_s")
        rules = R.build_rules(61, 3, 2, 2, t)
        self.assertEqual(R.resolve(rules, "blk.20.ffn_gate_exps.weight"), "iq1_xs")

    def test_negative_control_wrong_expectation_fails(self):
        # the resolver must be able to say "no": an unknown tensor resolves to None (default type), not to a tier
        self.assertIsNone(R.resolve(self.rules, "blk.3.attn_norm.bias.unknown"))

    def test_tier_table_is_monotone_in_bits(self):
        order = ["iq1_s", "iq2_xxs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs", "q4_k", "q5_k", "q6_k"]
        for tier in ("iq1", "iq2", "iq3"):
            d = R.TIERS[tier]
            self.assertLessEqual(order.index(d["low"]), order.index(d["down"]), tier)
            self.assertLessEqual(order.index(d["down"]), order.index(d["attn"]) if d["attn"] in order else 99, tier)


if __name__ == "__main__":
    unittest.main(verbosity=2)
