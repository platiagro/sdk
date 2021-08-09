# Test libs
from unittest import TestCase
from uuid import uuid4

# Math/Alg. Lin.
import numpy as np

# Metrics
from platiagro.metrics_nlp import Bleu, Gleu, Rouge, BertScore, Bleurt, Meteor, Accuracy
from platiagro.metrics_nlp import FScore, Precision, Recall, WER, MER, WIP, WIL
from platiagro.metrics_nlp import JaroWinkler, Jaro, EditDistance

# Wrapper
from platiagro.metrics_nlp import MetricsCalculator

# Test values
from platiagro.metrics_nlp import SAMPLE_HYPS, SAMPLE_REFS_SINGLE, SAMPLE_REFS_MULTI
from platiagro.metrics_nlp import SAMPLE_HYPS_TK, SAMPLE_REFS_SINGLE_TK, SAMPLE_REFS_MULTI_TK

# List Metrics
from platiagro.metrics_nlp import get_metrics_data

# Mock tokenizer
class MockTokenizer(object):
    """A mock tokenizer for testing"""

    def decode(self, token_ids_batch: List[List[int]]) -> List[str]:
        """Decodes list of token_ids to list of strings"""

        if token_ids_batch == SAMPLE_HYPS_TK:
            return SAMPLE_HYPS
        
        if token_ids_batch == SAMPLE_REFS_SINGLE_TK:
            return SAMPLE_REFS_SINGLE

        if token_ids_batch == SAMPLE_REFS_MULTI_TK:
            return SAMPLE_REFS_MULTI

        raise ValueError('Unknown token_ids_batch')
 
RUN_ID = str(uuid4())


class TestPlotting(TestCase):

    def setUp(self):
        """Don't need a setup"""
        pass

    def test_metrics_call(self):
        
        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())

        # Check metrics
        for metric in metrics_name:

            metric_component = metrics_data[metric]['component']()

            hypothesis = SAMPLE_HYPS[0]
            references_single = SAMPLE_REFS_SINGLE[0]
            references_mult = SAMPLE_REFS_MULT[0]

            # Call metric

            # Single reference
            metric_value = metric_component(hypothesis=hypothesis, references=references_single)
            self.assertIsInstance(metric_value, float)

            # Multiple reference
            if not metric_data[metric]['single_ref_only']:
                metric_value = metric_component(hypothesis=hypothesis, references=references_mult)
                self.assertIsInstance(metric_value, float)
    
    def test_metrics_calculate(self):
        
        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())

        # Check metrics
        for metric in metrics_name:

            metric_component = metrics_data[metric]['component']()

            hypothesis = SAMPLE_HYPS_TK
            references_single = SAMPLE_REFS_SINGLE
            references_mult = SAMPLE_REFS_MULT

            # Call metric

            # Single reference
            metric_value = metric_component.calculate(hypothesis=hypothesis, references=references_single)
            self.assertIsInstance(metric_value, float)

            # Multiple reference
            if not metric_data[metric]['single_ref_only']:
                metric_value = metric_component.calculate(hypothesis=hypothesis, references=references_mult)
                self.assertIsInstance(metric_value, float)

    def test_metrics_wrapper(self):

        # Get metrics data
        metrics_data = get_metrics_data()
        metrics_name = list(metrics_data.keys())
        mult_metrics_name = [metric for metric in metrics_name if not metrics_data[metric]['single_ref_only']]

        # Initializate wrappers
        wrapper_all = MetricsCalculator(metrics=metrics_name)
        wrapper_mult = MetricsCalculator(metrics=mult_metrics_name)

        # Check wrappers #

        # All, single reference, text
        values = wrapper_all.calculate_from_texts(hypothesis=SAMPLE_HYPS, references=SAMPLE_REFS_SINGLE)
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(metrics_name))

        # All, single reference, tokens
        values = wrapper_all.calculate_from_tokens(hypothesis=SAMPLE_HYPS_TK, references=SAMPLE_REFS_SINGLE_TK, tokenizer=MockTokenizer())
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(metrics_name))

        # Mult, multiple reference, text
        values = wrapper_mult.calculate_from_tokens(hypothesis=SAMPLE_HYPS, references=SAMPLE_REFS_MULT)
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(mult_metrics_name))

        # Mult, multiple reference, token
        values = wrapper_mult.calculate_from_tokens(hypothesis=SAMPLE_HYPS_TK, references=SAMPLE_REFS_MULT_TK, tokenizer=MockTokenizer())
        self.assertIsInstance(values, dict)
        self.assertEqual(len(values), len(mult_metrics_name))



        




