## IMPORTS ##

# Class
from abc import ABC, abstractmethod

# typing
from typing import Union, List, Callable, Dict, Any

# samples and validators
from platiagro.metrics_nlp.utils import SAMPLE_HYPS, SAMPLE_REFS_MULT, SAMPLE_REFS_SINGLE
from platiagro.metrics_nlp.utils import _hyp_typo_validator, _ref_typo_validator, _mult_references_validator, _empty_values_score

# numpy
import numpy as np

## BASE CLASS (TEMPLATE) ##

class BaseMetric(ABC):
    """Abstract Model class that is inherited to all NLP metrics"""

    @abstractmethod
    def __call__(self,**kwargs):
        pass

    @abstractmethod
    def calculate(self,**kwargs):
        pass

    def _health_validation(self, **kwargs):
        '''Validates health of metric'''

        _ = self(**kwargs)


## NLTK SCORES CLASS (TEMPLATE) ##

class NLTKScore(BaseMetric):
    """NLTKScore template metric class"""

    def __call__(self,
                 hypothesis: Union[str, List[str]], 
                 references: Union[str,  List[str]],
                 **kwargs) -> float:

        '''Compute NLTKScore score of a hypothesis and a reference.

            Params:
                hypothesis (str): a hypothesis sentence or a list of hypothesis sentences
                reference (str): a reference sentence or a list of reference sentences
                kwargs: see complete list at: https://www.nltk.org/_modules/nltk/metrics/scores.html

            Returns:
                NLTKScore score (float) from a hypothesis and a reference
        '''

        if isinstance(hypothesis, str):
            hypothesis = {hypothesis}
            references = {references}
        
        else:
            hypothesis = set(hypothesis)
            references = set(references)

        assert len(hypothesis) == len(references), "Hypothesis and reference lists must have the same length"

        score = self.metric(test=hypothesis, reference=references, **kwargs)

        return float(score)

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[str],
                  **kwargs) -> float:

        '''Compute NLTKScore score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[str]): list of reference sentence
                kwargs: see complete list at: https://www.nltk.org/_modules/nltk/metrics/scores.html
                

            Returns:
                NLTKScore score (float) from a batch_hypotheses and batch_references
        '''

        assert not _mult_references_validator(batch_references), _MULT_REF_ERROR_MSG

        # Validates hypothesis and references
        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

        score = self(batch_hypotheses, batch_references, **kwargs)

        return float(score)

## JIWER SCORES CLASS (TEMPLATE) ##

class JIWERScore(BaseMetric):
    """JIWERScore template metric class"""

    def __call__(self,
                 hypothesis: Union[str, List[str]], 
                 references: Union[str,  List[str]],
                 **kwargs) -> float:

        '''Compute JIWERScore score of a hypothesis and a reference.

            Params:
                hypothesis (str): a hypothesis sentence or a list of hypothesis sentences
                reference (str): a reference sentence or a list of reference sentences
                kwargs: see complete list at: https://github.com/jitsi/jiwer/blob/1fd2a161fd21296640c655da0786e94ea0f5df77/jiwer/measures.py#L65

            Returns:
                JIWERScore score (float) from a hypothesis and a reference
        '''

        if hypothesis == '' or references == '':

            if self.metric.__name__ == 'wip':
                max_val = 0.0
                min_val = 1.0
            else:
                max_val = 1.0
                min_val = 0.0

            return _empty_values_score(hypothesis, references, min_val=max_val, max_val=min_val)

        score = self.metric(truth=references, 
                            hypothesis=hypothesis,
                            truth_transform=self.truth_transform,
                            hypothesis_transform=self.hypothesis_transform)

        return float(score)

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[str],
                  **kwargs) -> float:

        '''Compute JIWERScore score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[str]): list of reference sentence
                kwargs: see complete list at: https://github.com/jitsi/jiwer/blob/1fd2a161fd21296640c655da0786e94ea0f5df77/jiwer/measures.py#L65
                

            Returns:
                JIWERScore score (float) from a batch_hypotheses and batch_references
        '''

        assert not _mult_references_validator(batch_references), _MULT_REF_ERROR_MSG

        # Validates hypothesis and references
        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

        score = self(batch_hypotheses, batch_references, **kwargs)

        return float(score)

## NLTK DISTANCE CLASS (TEMPLATE) ##

class NLTKDistance(BaseMetric):
    """NLTKDistance template metric class"""

    def __call__(self,
                 hypothesis: str, 
                 references: str,
                 **kwargs) -> float:

        '''Compute NLTKDistance score of a hypothesis and a reference.

            Params:
                hypothesis (str): a hypothesis sentence
                reference (str): a reference sentence
                kwargs: see complete list at: https://www.nltk.org/api/nltk.metrics.html

            Returns:
                NLTKDistance score (float) from a hypothesis and a reference
        '''

        score = self.metric(s1=hypothesis, s2=references, **kwargs)

        return float(score)

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[str],
                  **kwargs) -> float:

        '''Compute NLTKDistance score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[str]): list of reference sentence
                kwargs: see complete list at: https://www.nltk.org/api/nltk.metrics.html
                

            Returns:
                Mean NLTKDistance score (float) from a batch_hypotheses and batch_references
        '''

        assert not _mult_references_validator(batch_references), _MULT_REF_ERROR_MSG

        scores = []

        # Validates hypothesis and references
        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

            scores.append(self(hyp, ref, **kwargs))

        return float(np.mean(scores))
