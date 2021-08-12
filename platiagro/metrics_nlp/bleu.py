#############
## IMPORTS ##
#############

# NLTK
from nltk.translate.bleu_score import sentence_bleu

# typing
from typing import Union, List, Callable, Dict, Any

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_MULT, SAMPLE_REFS_SINGLE
from platiagro.metrics_nlp.utils import _hyp_typo_validator, _ref_typo_validator

# base class
from platiagro.metrics_nlp.base import BaseMetric

# numpy
import numpy as np

################
## BLEU CLASS ##
################

class Bleu(BaseMetric):
    """BLEU metric class"""

    def __init__(self):

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_MULT[0])
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def __call__(self,
                 hypothesis: str, 
                 references: Union[List[str], str],
                 weights: List[float] = [0.25, 0.25, 0.25, 0.25],
                 smoothing_function: Callable = None,
                 auto_reweigh: bool = False) -> float:

        '''Compute BLEU score of a hypothesis and a reference.

            Params:
                hypothesis (str): hypothesis sentence
                reference (list[str] or str): list of reference sentences or a reference sentence
                weights (list[float]): weights for unigrams, bigrams, trigrams and so on
                smoothing_function (function):
                    function used to smooth/weight the scores of sub-sequences.
                    Default: SmoothingFunction().
                auto_reweigh (bool): option to re-normalize the weights uniformly
                

            Returns:
                BLEU score (float) from a hypothesis and reference(s)
        '''

        if isinstance(references, str):
            references = [references]

        return float(sentence_bleu(references, hypothesis, weights, smoothing_function, auto_reweigh))

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[Union[List[str], str]],
                  weights: List[float] = [0.25, 0.25, 0.25, 0.25],
                  smoothing_function: Callable = None,
                  auto_reweigh: bool = False) -> float:

        '''Compute BLEU score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[list[str] or str]): list of list of reference sentences or a list of reference sentence
                weights (list[float]): weights for unigrams, bigrams, trigrams and so on
                smoothing_function (function):
                    function used to smooth/weight the scores of sub-sequences.
                    Default: SmoothingFunction().
                auto_reweigh (bool): option to re-normalize the weights uniformly
                

            Returns:
                Mean BLEU score (float) from a batch_hypotheses and batch_references
        '''

        scores = []

        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

            # Calculate score
            score = self(hyp, ref, weights, smoothing_function, auto_reweigh)
            scores.append(score)
        
        return float(np.mean(scores))
