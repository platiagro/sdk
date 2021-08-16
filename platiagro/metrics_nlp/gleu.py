# IMPORTS #


# NLTK
from nltk.translate.gleu_score import sentence_gleu

# typing
from typing import Union, List

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_MULT, SAMPLE_REFS_SINGLE
from platiagro.metrics_nlp.utils import _hyp_typo_validator, _ref_typo_validator

# base class
from platiagro.metrics_nlp.base import BaseMetric

# numpy
import numpy as np

# GLEU CLASS #


class Gleu(BaseMetric):
    """GLEU metric class"""

    def __init__(self):

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_MULT[0])
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def __call__(self,
                 hypothesis: str,
                 references: Union[List[str], str],
                 min_len: int = 1,
                 max_len: int = 4) -> float:

        '''Compute GLEU score (Google-BLEU) of a hypothesis and a reference.

            Params:
                hypothesis (str): hypothesis sentence
                reference (list[str] or str): list of reference sentences or a reference sentence
                min_len (int): the minimum order of n-gram this function should extract
                max_len (int): the maximum order of n-gram this function should extract


            Returns:
                GLEU score (float) from a hypothesis and reference(s)
        '''

        if isinstance(references, str):
            references = [references]

        return float(sentence_gleu(references, hypothesis, min_len, max_len))

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[Union[List[str], str]],
                  min_len: int = 1,
                  max_len: int = 4) -> float:

        '''Compute GLEU score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[list[str] or str]): list of list of reference sentences or a list of reference sentence
                min_len (int): the minimum order of n-gram this function should extract
                max_len (int): the maximum order of n-gram this function should extract


            Returns:
                Mean GLEU score (float) from a batch_hypotheses and batch_references
        '''

        scores = []

        for hyp, ref in zip(batch_hypotheses, batch_references):

            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

            # Calculate score
            score = self(hyp, ref, min_len, max_len)
            scores.append(score)

        return float(np.mean(scores))
