import unittest

from apc_v4_engine import compute_apc_v4
from rule_based_extractor import extract_rule_vector, merge_rule_and_llm_vectors


class TestTransactionOutputLayer(unittest.TestCase):
    def test_output_mismatch_keeps_candidate_and_requires_confirmation(self):
        x = [0.0] * 44
        x[0] = 1  # coding
        x[1] = 1  # context
        x[25] = 1 # r10 syntax lookup -> L6 candidate
        x[26] = 1 # output present
        x[27] = 0 # role not aligned
        x[28] = 0 # scope not aligned
        x[29] = 0 # agency not aligned
        x[30] = 0 # form not aligned
        x[31] = 1 # pedagogy not the issue
        x[33] = 1 # complete solution
        x[39] = 1 # over scope
        x[41] = 1 # role escalation
        x[42] = 1 # agency takeover
        res = compute_apc_v4(x)
        self.assertEqual(res["candidate_level"], "L6")
        self.assertEqual(res["level"], "Can hoi lai SV")
        self.assertTrue(res["requires_student_confirmation"])
        self.assertEqual(res["final_status"], "requires_student_confirmation")
        self.assertIn("over_scope_broader", res["confirmation_reasons"])

    def test_output_aligned_keeps_candidate(self):
        x = [0.0] * 44
        x[0] = 1
        x[1] = 1
        x[25] = 1 # L6 prompt
        x[26] = 1 # output present
        x[27] = 1
        x[28] = 1
        x[29] = 1
        x[30] = 1
        x[31] = 1
        x[38] = 1 # narrow reference
        res = compute_apc_v4(x)
        self.assertEqual(res["candidate_level"], "L6")
        self.assertEqual(res["level"], "L6")
        self.assertFalse(res["requires_student_confirmation"])
        self.assertEqual(res["transaction_status"], "output_aligned")

    def test_rule_extractor_obvious_syntax_mismatch(self):
        prompt = "Cú pháp lệnh git để đổi tên branch hiện tại là gì?"
        output = "Đây là full implementation:\n```python\nclass GitBranchManager:\n    pass\n```\n" * 40
        rule = extract_rule_vector(prompt, output)
        final = merge_rule_and_llm_vectors([0.0] * 44, rule)
        res = compute_apc_v4(final)
        self.assertEqual(res["candidate_level"], "L6")
        self.assertTrue(res["requires_student_confirmation"])
        self.assertIn("over_scope_broader", res["confirmation_reasons"])


if __name__ == "__main__":
    unittest.main()
