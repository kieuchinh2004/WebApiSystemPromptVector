import unittest
from apc_v4_engine import compute_apc_v4

class TestAPCV4Engine(unittest.TestCase):
    
    def test_prefix_non_coding(self):
        # bit 0 is_coding = 0, has_context = 1
        vec = "01" + "0" * 24
        res = compute_apc_v4(vec)
        self.assertEqual(res["level"], "L0")
        self.assertEqual(res["accept"], 0)
        self.assertEqual(res["is_coding"], False)
        self.assertEqual(res["has_context"], True)

    def test_prefix_lacks_context(self):
        # bit 1 has_context = 0
        vec = "10" + "0" * 24
        res = compute_apc_v4(vec)
        self.assertEqual(res["level"], "Thieu context")
        self.assertEqual(res["accept"], 0)
        self.assertEqual(res["is_coding"], True)
        self.assertEqual(res["has_context"], False)

    def test_zero_vector_classification(self):
        # prefix 11, followed by 24 zeros
        vec = "11" + "0" * 24
        res = compute_apc_v4(vec)
        
        # Expected values calculated by hand:
        # S1 = 0.50, S2 = 0.15, S3 = 0.40, S4 = 0.15, S5 = 0.30, S6 = 0.45
        # (S1 leads with r1's positive weight plus a1/d1/d2/r7 absence terms; it no
        # longer carries a2..a5 absence terms, so a fully empty vector no longer gives
        # L1 an inflated score floor over levels with genuine positive signals.)
        self.assertAlmostEqual(res["scores"][0], 0.50, places=4)
        self.assertAlmostEqual(res["scores"][1], 0.15, places=4)
        self.assertAlmostEqual(res["scores"][2], 0.40, places=4)
        self.assertAlmostEqual(res["scores"][3], 0.15, places=4)
        self.assertAlmostEqual(res["scores"][4], 0.30, places=4)
        self.assertAlmostEqual(res["scores"][5], 0.45, places=4)

        # Gatings for zero vector:
        # G1..G3 are True. G4..G6 are False.
        self.assertEqual(res["gatings"], [True, True, True, False, False, False])

        # Math predicted level must be L1 (highest among gated levels S1, S2, S3)
        self.assertEqual(res["predicted_level_math"], "L1")
        self.assertEqual(res["level"], "L1")

        # Accept criteria now uses gated Softmax over active levels L1-L3 only.
        self.assertAlmostEqual(res["confidence"], 0.6622, places=3)
        self.assertAlmostEqual(res["margin"], 0.3646, places=3)
        self.assertEqual(res["accept"], 1)
        self.assertIn("all_probs", res)

    def test_l2_signal_beats_l1_default_on_sparse_vector(self):
        # Prompt supplies a code snippet + specific location and asks for a fix
        # (a1, a7, d5, r2) without setting any other bit.  Before the s1 rebalance
        # this used to lose to L1's inflated absence-driven score (0.60 vs 0.55);
        # the genuine L2 signal must now win.
        x = [0] * 26
        x[0] = 1  # is_coding
        x[1] = 1  # has_context
        x[2] = 1  # a1_code_block
        x[8] = 1  # a7_small_snippet
        x[13] = 1  # d5_location
        x[17] = 1  # r2_fix

        res = compute_apc_v4(x)
        self.assertEqual(res["predicted_level_math"], "L2")

    def test_gating_g4_active(self):
        # We activate d1 (index 9) and r7 (index 22)
        # G4 = (d1 >= 0.5 or d2 >= 0.5 or d3 >= 0.5 or d4 >= 0.5) and r7 >= 0.5
        # We set is_coding=1, has_context=1
        x = [0] * 26
        x[0] = 1 # is_coding
        x[1] = 1 # has_context
        x[9] = 1 # d1 is index 9
        x[22] = 1 # r7 is index 22
        
        res = compute_apc_v4(x)
        self.assertTrue(res["gatings"][3]) # G4 should be True

    def test_gating_g5_active(self):
        # G5 = a6 >= 0.5 and r8 >= 0.5
        # a6 is index 7
        # r8 is index 23
        x = [0] * 26
        x[0] = 1
        x[1] = 1
        x[7] = 1  # a6 is index 7
        x[23] = 1 # r8 is index 23
        
        res = compute_apc_v4(x)
        self.assertTrue(res["gatings"][4]) # G5 should be True

    def test_gating_g6_active(self):
        # G6 = r10 >= 0.5
        x = [0] * 26
        x[0] = 1
        x[1] = 1
        # r10 is index 25 in the 24-feature vector -> index 27 in 26-bit? Wait!
        # Let's count indices:
        # Prefix: index 0, 1 (2 bits)
        # Artifacts: a1..a7 -> indices 2 to 8 (7 bits)
        # Design: d1..d7 -> indices 9 to 15 (7 bits)
        # Request: r1..r10 -> indices 16 to 25 (10 bits)
        # Total size: 2 + 7 + 7 + 10 = 26 bits (indices 0 to 25).
        # So r10 is exactly index 25!
        x[25] = 1 # r10 is index 25
        
        res = compute_apc_v4(x)
        self.assertTrue(res["gatings"][5]) # G6 should be True

    def test_accept_uses_gated_probability_not_ungated_max(self):
        # This vector gives ungated L5 the largest raw probability, but G5 is false.
        # The selected math level is L1; confidence/accept must not be taken from L5.
        vec = "11000000000000010000001110"
        res = compute_apc_v4(vec)
        self.assertEqual(res["predicted_level_math"], "L1")
        self.assertFalse(res["gatings"][4])
        self.assertGreater(res["all_probs"][4], res["all_probs"][0])
        self.assertEqual(res["probs"][4], 0.0)
        self.assertAlmostEqual(res["confidence"], res["probs"][0], places=4)

    def test_accepts_float_vector_values(self):
        x = [0.0] * 26
        x[0] = 1.0  # is_coding
        x[1] = 1.0  # has_context
        x[9] = 0.9  # d1
        x[22] = 0.9 # r7
        res = compute_apc_v4(x)
        self.assertEqual(res["predicted_level_math"], "L4")
        self.assertTrue(res["gatings"][3])
        self.assertEqual(res["vector_str"][9], "1")

if __name__ == "__main__":
    unittest.main()
