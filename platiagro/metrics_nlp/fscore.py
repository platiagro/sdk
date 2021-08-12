# IMPORTS #


# NLTK score
from nltk.metrics.scores import f_measure

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_SINGLE

# base class
from platiagro.metrics_nlp.base import NLTKScore

# FSCORE CLASS #


class FScore(NLTKScore):
    """FSCORE metric class: inherits from NLTKScore class"""

    def __init__(self):

        self.metric = f_measure

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])
