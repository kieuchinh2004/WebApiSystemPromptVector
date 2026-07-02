import unittest

from classifier import VectorParseError, parse_vector_from_content


class TestClassifierParser(unittest.TestCase):
    def test_parse_binary_vector_without_explanation(self):
        vector = "10" + "0" * 58
        values, explanation, payload = parse_vector_from_content('{"vector":"' + vector + '"}')
        self.assertEqual(len(values), 60)
        self.assertEqual(values[:2], [1.0, 0.0])
        self.assertEqual(explanation, "Vector extracted without explanation.")
        self.assertEqual(payload["vector"], vector)

    def test_parse_binary_vector(self):
        vector = "10" + "0" * 24
        values, explanation, payload = parse_vector_from_content(
            '{"vector":"' + vector + '","explanation":"context missing"}'
        )
        self.assertEqual(len(values), 26)
        self.assertEqual(values[:2], [1.0, 0.0])
        self.assertEqual(explanation, "context missing")
        self.assertEqual(payload["vector"], vector)

    def test_parse_float_features(self):
        features = [0.5] * 26
        body = '{"features":[' + ",".join(str(v) for v in features) + '],"explanation":"partial"}'
        values, explanation, _ = parse_vector_from_content(body)
        self.assertEqual(values, features)
        self.assertEqual(explanation, "partial")

    def test_reject_short_vector(self):
        with self.assertRaises(VectorParseError):
            parse_vector_from_content('{"vector":"101","explanation":"bad"}')

    def test_reject_long_vector(self):
        with self.assertRaises(VectorParseError):
            parse_vector_from_content('{"vector":"' + "1" * 27 + '","explanation":"bad"}')

    def test_reject_feature_out_of_range(self):
        features = [0.0] * 26
        features[7] = 1.2
        body = '{"features":[' + ",".join(str(v) for v in features) + '],"explanation":"bad"}'
        with self.assertRaises(VectorParseError):
            parse_vector_from_content(body)

    def test_reject_non_json_text(self):
        with self.assertRaises(VectorParseError):
            parse_vector_from_content("vector: 11000000000000000000000000")


if __name__ == "__main__":
    unittest.main()
