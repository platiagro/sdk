## IMPORTS ##

# Bleurt (pip install git+https://github.com/google-research/bleurt.git)
from bleurt.score import BleurtScorer

# typing
from typing import Union, List, Callable, Dict, Any

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_MULT, SAMPLE_REFS_SINGLE
from platiagro.metrics_nlp.utils import _hyp_typo_validator, _ref_typo_validator

# base class
from platiagro.metrics_nlp.base import BaseMetric

# numpy
import numpy as np

## BLEURT CLASS ##

class Bleurt(BaseMetric):
    """BLEURT metric class"""

    def __init__(self):
        
        self.bleurt = BleurtScorer()

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_MULT[0])
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def __call__(self,
                 hypothesis: Union[List[str], str], 
                 references: Union[List[str], str, List[List[str]]],
                 is_batch: bool = False,
                 average: Callable = np.max,
                 **kwargs) -> float:

        '''Compute BLEURT score of a hypothesis and a reference.
           See more at: https://github.com/google-research/bleurt

            Params:
                hypothesis (list[str] or str): list of hypothesis sentence or a hypothesis sentence
                reference (list[list[str]] or list[str] or str): list of reference sentences or a reference sentence
                is_batch (bool): whether or not to return BLEURT score for a batch of hypothesis and references or a single value
                average (function): function used to average the scores of multiple references if applicable
                kwargs: see complete list at: https://github.com/google-research/bleurt/blob/master/bleurt/score.py#L143
                

            Returns:
                BLEURT score (float) from a hypothesis and reference(s)
        '''

        if isinstance(references, list) and isinstance(references[0], str) and isinstance(hypothesis, str):
            references = [references]

        if isinstance(references, str):
            references = [[references]]
        
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]

        scores = []
        for hyp, ref in zip(hypothesis, references):
            
            int_scores = []
            if isinstance(ref, list):
                for ref_sent in ref:
                    int_scores.append(self.bleurt.score(references=[ref_sent], candidates=[hyp], **kwargs))
                score = average(int_scores)
            else:
                score = self.bleurt.score(references=[ref], candidates=[hyp], **kwargs)
            
            # Add score
            scores.append(score)

        if is_batch:
            return float(average(scores))
        else:
            return float(scores[0])

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[Union[List[str], str]],
                  average: Callable = np.max,
                  **kwargs) -> float:

        '''Compute BLEURT score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[list[str] or str]): list of list of reference sentences or a list of reference sentence
                average (function): function used to average the scores of multiple references if applicable
                kwargs: see complete list at: https://github.com/google-research/bleurt/blob/master/bleurt/score.py#L143
                

            Returns:
                Mean BLEURT score (float) from a batch_hypotheses and batch_references
        '''

        scores = []

        # Validates hypothesis and references
        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

        scores = self(batch_hypotheses, batch_references, is_batch=True, average=average, **kwargs)

        return np.mean(scores)
