# Test libs
from unittest import TestCase
from uuid import uuid4

# Wrapper
from platiagro.metrics_nlp.wrapper import MetricsCalculator

# Test values
from platiagro.metrics_nlp.utils import (
    SAMPLE_HYPS,
    SAMPLE_REFS_SINGLE,
    SAMPLE_REFS_MULT,
)
from platiagro.metrics_nlp.utils import (
    SAMPLE_HYPS_TK,
    SAMPLE_REFS_SINGLE_TK,
    SAMPLE_REFS_MULT_TK,
)

# List Metrics
from platiagro.metrics_nlp.metrics import get_metrics_data

# Typing
from typing import List


# Mock tokenizer
class MockTokenizer(object):
    """A mock tokenizer for testing"""

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        """Decodes list of token_ids to list of strings"""

        for i, sample_hyp_tk in enumerate(SAMPLE_HYPS_TK):
            if token_ids == sample_hyp_tk:
                return SAMPLE_HYPS[i]

        for i, sample_ref_single_tk in enumerate(SAMPLE_REFS_SINGLE_TK):
            if token_ids == sample_ref_single_tk:
                return SAMPLE_REFS_SINGLE[i]

        for i, sample_ref_mult_tk in enumerate(SAMPLE_REFS_MULT_TK):
            for j, sample_ref_tk in enumerate(sample_ref_mult_tk):
                if token_ids == SAMPLE_REFS_MULT_TK[i][j]:
                    return SAMPLE_REFS_MULT[i][j]

        raise ValueError("Unknown token_ids_batch")


RUN_ID = str(uuid4())


class TestMetricsNLP(TestCase):
    def setUp(self):
        """Don't need a setup"""
        pass

    def test_metrics_call(self):

        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())

        # Check metrics
        for metric in metrics_name:

            # Initialize metric component
            metric_component = metrics_data[metric]["component"]()

            # Get test sample
            hypothesis = SAMPLE_HYPS[0]
            references_single = SAMPLE_REFS_SINGLE[0]
            references_mult = SAMPLE_REFS_MULT[0]

            # Call metric

            # Single reference
            metric_value = metric_component(
                hypothesis=hypothesis, references=references_single
            )
            self.assertIsInstance(metric_value, float)

            # Multiple reference
            if not metrics_data[metric]["single_ref_only"]:
                metric_value = metric_component(
                    hypothesis=hypothesis, references=references_mult
                )
                self.assertIsInstance(metric_value, float)

    def test_metrics_calculate(self):

        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())

        # Check metrics
        for metric in metrics_name:

            # Initialize metric component
            metric_component = metrics_data[metric]["component"]()

            # Call metric

            # Single reference
            metric_value = metric_component.calculate(
                batch_hypotheses=SAMPLE_HYPS, batch_references=SAMPLE_REFS_SINGLE
            )
            self.assertIsInstance(metric_value, float)

            # Multiple reference
            if not metrics_data[metric]["single_ref_only"]:
                metric_value = metric_component.calculate(
                    batch_hypotheses=SAMPLE_HYPS, batch_references=SAMPLE_REFS_MULT
                )
                self.assertIsInstance(metric_value, float)

    def test_metrics_wrapper_all(self):

        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())

        # Initializate wrappers
        wrapper_all = MetricsCalculator()  # default == all metrics
        wrapper_all_params = MetricsCalculator(
            metric_params={"gleu": {"min_len": 1, "max_len": 3}}
        )  # default == all metrics

        # Check documentation
        self.assertIsInstance(wrapper_all.__str__(), str)
        self.assertIsInstance(wrapper_all_params.__str__(), str)

        # Check wrappers #

        # All, single reference, text
        values = wrapper_all.calculate_from_texts(
            hypothesis=SAMPLE_HYPS, references=SAMPLE_REFS_SINGLE
        )
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(metrics_name))

        # All, single reference, tokens
        values = wrapper_all.calculate_from_tokens(
            hypothesis_tokens=SAMPLE_HYPS_TK,
            references_tokens=SAMPLE_REFS_SINGLE_TK,
            tokenizer=MockTokenizer(),
        )
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(metrics_name))

        # All, single reference, text
        values = wrapper_all_params.calculate_from_texts(
            hypothesis=SAMPLE_HYPS, references=SAMPLE_REFS_SINGLE
        )
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(metrics_name))

        # All, single reference, tokens
        values = wrapper_all_params.calculate_from_tokens(
            hypothesis_tokens=SAMPLE_HYPS_TK,
            references_tokens=SAMPLE_REFS_SINGLE_TK,
            tokenizer=MockTokenizer(),
        )
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(metrics_name))

    def test_metrics_wrapper_mult(self):

        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())
        mult_metrics_name = [
            metric
            for metric in metrics_name
            if not metrics_data[metric]["single_ref_only"]
        ]

        # Initializate wrappers
        wrapper_mult = MetricsCalculator(metrics=mult_metrics_name)

        # Check documentation
        self.assertIsInstance(wrapper_mult.__str__(), str)

        # Check wrappers #

        # Mult, multiple reference, text
        values = wrapper_mult.calculate_from_texts(
            hypothesis=SAMPLE_HYPS, references=SAMPLE_REFS_MULT
        )
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(mult_metrics_name))

        # Mult, multiple reference, token
        values = wrapper_mult.calculate_from_tokens(
            hypothesis_tokens=SAMPLE_HYPS_TK,
            references_tokens=SAMPLE_REFS_MULT_TK,
            tokenizer=MockTokenizer(),
        )
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(mult_metrics_name))

    def test_metrics_invalid(self):

        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())

        # Check metrics
        for metric in metrics_name:

            # Initialize metric component
            metric_component = metrics_data[metric]["component"]()

            # Call metric

            with self.assertRaises(ValueError):
                metric_component.calculate(batch_hypotheses=[""], batch_references=[0])

            with self.assertRaises(ValueError):
                metric_component.calculate(batch_hypotheses=[0], batch_references=[""])

        if "rouge" in metrics_name:

            rouge = metrics_data["rouge"]["component"]()

            # Invalid rouge metric
            with self.assertRaises(ValueError):
                rouge("a", "b", metric="")

            # Invalid rouge method
            with self.assertRaises(ValueError):
                rouge("a", "b", method="")

    def test_metrics_empty_string(self):

        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())

        # Check metrics
        for metric in metrics_name:

            # Initialize metric component
            metric_component = metrics_data[metric]["component"]()

            # Call metric
            value = metric_component(hypothesis="", references="")
            self.assertIsInstance(value, float)

            value = metric_component(hypothesis="a", references="")
            self.assertIsInstance(value, float)

            value = metric_component(hypothesis="", references="a")
            self.assertIsInstance(value, float)
