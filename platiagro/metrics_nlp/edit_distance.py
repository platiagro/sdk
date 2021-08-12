#############
## IMPORTS ##
#############

# NLTK
from nltk.metrics.distance import edit_distance

# typing
from typing import Union, List, Callable, Dict, Any

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_SINGLE

# base class
from platiagro.metrics_nlp.base import NLTKDistance

#########################
## EDIT DISTANCE CLASS ##
#########################

class EditDistance(NLTKDistance):
    """EDIT_DISTANCE metric class: inherits from NLTKDistance class"""

    def __init__(self):

        self.metric = edit_distance

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])
