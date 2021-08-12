#############
## IMPORTS ##
#############

# ROUGE
from rouge import Rouge as RougeRaw

# typing
from typing import Union, List, Callable, Dict, Any

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_MULT, SAMPLE_REFS_SINGLE
from platiagro.metrics_nlp.utils import _hyp_typo_validator, _ref_typo_validator, _empty_values_score

# base class
from platiagro.metrics_nlp.base import BaseMetric

# numpy
import numpy as np

# validator

def _rouge_validator(rouge_method: str, rouge_metric: str):
    '''Validates rouge params'''

    try: assert rouge_method in ['l', '1', '2']
    except AssertionError:
        raise ValueError(f'"{rouge_method}" not implemented. Method must be "l", "1" or "2"')
    
    try: assert rouge_metric in ['f1', 'precision', 'recall']
    except AssertionError:
        raise ValueError(f'"{rouge_metric}" not implemented. Metric must be "f1", "precision" or "recall"')

#################
## ROUGE CLASS ##
#################

class Rouge(BaseMetric):
    """Rouge metric class"""

    def __init__(self):
        
        self.metric_map = {
            'f1': 'f',
            'precision': 'p',
            'recall': 'r',
        }

        self.rouge = RougeRaw()

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_MULT[0])
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def _get_score(self,
                   rouge_scores: List,
                   method: str,
                   metric: str) -> float:

        '''Get the ROUGE score of a specific method and metric.

            Params:
                rouge_scores (list): list of ROUGE scores
                method (str): 'l' or '1' or '2'
                metric (str): 'f1' or 'precision' or 'recall'

            Returns:
                ROUGE score (float)
        '''
        
        # Get first element that represents the scores
        scores = rouge_scores[0]

        # Get method scores
        method_scores = scores[f'rouge-{method}']

        # Get metric score
        score = method_scores[self.metric_map[metric]]

        return score

    def __call__(self,
                 hypothesis: str, 
                 references: Union[List[str], str],
                 method: str = 'l', 
                 metric: str = 'f1',
                 average: Callable = np.max) -> float:

        '''Compute ROUGE-N score of a hypothesis and a reference.

            Params:
                hypothesis (str): hypothesis sentence
                reference (list[str] or str): list of reference sentences or a reference sentence
                method (str): 'l' or '1' or '2'
                metric (str): 'f1' or 'precision' or 'recall'
                average (function): function used to average the scores of multiple references if applicable
                

            Returns:
                ROUGE-N score (float) from a hypothesis and reference(s)
        '''

        # empty values
        if hypothesis == '' or references == '':
            return _empty_values_score(hypothesis, references, min_val = 0.0, max_val = 1.0)

        # Validate method and metric rouge params
        _rouge_validator(method, metric)

        if isinstance(references, str):
            references = [references]

        all_scores = list()
        # Iterate over each reference and apply average func
        for reference in references:

            # Calculate all Rouge scores
            scores = self.rouge.get_scores(hypothesis, reference)

            # Take specific score
            score = self._get_score(scores, method, metric)

            # Append score
            all_scores.append(score)
        
        # Apply average func
        score = average(all_scores)

        return float(score)

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[Union[List[str], str]],
                  method: str = 'l', 
                  metric: str = 'f1',
                  average: Callable = np.max) -> float:

        '''Compute ROUGE-N score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[list[str] or str]): list of list of reference sentences or a list of reference sentence
                method (str): 'l' or '1' or '2'
                metric (str): 'f1' or 'precision' or 'recall'
                average (function): function used to average the scores of multiple references if applicable
                

            Returns:
                Mean ROUGE-N score (float) from a batch_hypotheses and batch_references
        '''

        scores = []

        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

            # Calculate score
            score = self(hyp, ref, method, metric, average)
            scores.append(score)
        
        return float(np.mean(scores))
