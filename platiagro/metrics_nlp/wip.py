## IMPORTS ##

# Jiwer (pip install jiwer)
import jiwer.transforms as tr
import jiwer

# typing
from typing import Union, List, Callable, Dict, Any

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_SINGLE

# base class
from platiagro.metrics_nlp.base import JIWERScore

## WIP CLASS ##

class WIP(JIWERScore):
    """WIP metric class: inherits from JIWERScore class"""

    def __init__(self,
                 truth_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform,
                 hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform):

        self.metric = jiwer.wip
        self.truth_transform = truth_transform
        self.hypothesis_transform = hypothesis_transform

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])
