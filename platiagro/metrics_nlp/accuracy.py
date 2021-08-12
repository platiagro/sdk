# IMPORTS #


# NLTK score
from nltk.metrics.scores import accuracy

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_SINGLE

# base class
from platiagro.metrics_nlp.base import NLTKScore

# ACCURACY CLASS #


class Accuracy(NLTKScore):
    """ACCURACY metric class: inherits from NLTKScore class"""

    def __init__(self):

        self.metric = accuracy
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])
