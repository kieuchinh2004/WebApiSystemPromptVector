import unittest

from apc_v4_engine import compute_apc_v4, VECTOR_SIZE
from rule_based_extractor import extract_rule_vector, merge_rule_and_llm_vectors
from vector_schema import FIELD_INDEX


class TestVectorV4Rubric(unittest.TestCase):
    def _res(self, prompt, output=None):
        rule = extract_rule_vector(prompt, output)
        return compute_apc_v4(rule.vector), rule

    def test_vibe_code_is_l1(self):
        res, _ = self._res("Tạo cho tôi module quản lý sinh viên")
        self.assertEqual(res["candidate_level"], "L1")

    def test_fix_existing_code_is_l2(self):
        res, _ = self._res("Sửa lỗi đoạn code này\n```python\ndef f():\n return x\n```\nNameError")
        self.assertEqual(res["candidate_level"], "L2")

    def test_pseudocode_flow_is_l4(self):
        res, _ = self._res("Em có flow validate input -> hash password -> save user. Code controller theo flow này.")
        self.assertEqual(res["candidate_level"], "L4")

    def test_minimal_lookup_is_l6(self):
        res, _ = self._res("Lệnh git tạo branch là gì?")
        self.assertEqual(res["candidate_level"], "L6")

    def test_printf_prompt_l6_output_l3_warning(self):
        output = (
            "printf in C is a function used to display output on the screen. "
            "It comes from stdio.h. Example code: #include <stdio.h>\n\n"
            "int main() { printf(\"Hello world!\"); return 0; } "
            "Example with variable: int age = 18; printf(\"I am %d years old\", age);"
        )
        res, _ = self._res("What is printf in C?", output)
        self.assertEqual(res["candidate_level"], "L6")
        self.assertEqual(res["output_level"], "L3")
        self.assertEqual(res["warning_level"], "L3")
        self.assertIn("minimal_to_broad_shift", res["confirmation_reasons"])

    def test_legacy_26_vector_still_accepted(self):
        old = [0.0] * 26
        old[0] = 1
        old[1] = 1
        old[25] = 1  # old r10 syntax lookup
        res = compute_apc_v4(old)
        self.assertEqual(res["candidate_level"], "L6")

    def test_60d_direct_vector(self):
        x = [0.0] * VECTOR_SIZE
        x[FIELD_INDEX["is_coding"]] = 1
        x[FIELD_INDEX["has_context"]] = 1
        x[FIELD_INDEX["minimal_lookup"]] = 1
        x[FIELD_INDEX["exact_small_need"]] = 1
        res = compute_apc_v4(x)
        self.assertEqual(res["candidate_level"], "L6")


if __name__ == "__main__":
    unittest.main()
