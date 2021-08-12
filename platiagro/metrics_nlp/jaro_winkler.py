#############
## IMPORTS ##
#############

# NLTK
from nltk.metrics.distance import jaro_winkler_similarity

# typing
from typing import Union, List, Callable, Dict, Any

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_SINGLE

# base class
from platiagro.metrics_nlp.base import NLTKDistance

###################################
## JARO WINKLER SIMILARITY CLASS ##
###################################

class JaroWinkler(NLTKDistance):
    """JARO_WINKLER metric class: inherits from NLTKDistance class"""

    def __init__(self):

        self.metric = jaro_winkler_similarity

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])
