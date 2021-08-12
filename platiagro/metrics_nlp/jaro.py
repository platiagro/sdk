# IMPORTS #


# NLTK
from nltk.metrics.distance import jaro_similarity

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_SINGLE

# base class
from platiagro.metrics_nlp.base import NLTKDistance

# JARO SIMILARITY CLASS #


class Jaro(NLTKDistance):
    """JARO metric class: inherits from NLTKDistance class"""

    def __init__(self):

        self.metric = jaro_similarity

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])
