#############
## IMPORTS ##
#############

# NLTK score
from nltk.metrics.scores import precision

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_MULT, SAMPLE_REFS_SINGLE

# base class
from platiagro.metrics_nlp.base import NLTKScore

# typing
from typing import Union, List, Callable, Dict, Any

#####################
## PRECISION CLASS ##
#####################

class Precision(NLTKScore):
    """PRECISION metric class: inherits from NLTKScore class"""

    def __init__(self):

        self.metric = precision
        
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])
