#############
## IMPORTS ##
#############

# NLTK
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.metrics.scores import accuracy, f_measure, precision, recall
from nltk.metrics.distance import jaro_winkler_similarity, jaro_similarity, edit_distance

# ROUGE
from rouge import Rouge as RougeRaw

# typing
from typing import Union, List, Callable, Dict, Any

# Numpy
import numpy as np

# Class
from abc import ABC, abstractmethod

# Hugging Face Metrics
from datasets import load_metric

# Bleurt (pip install git+https://github.com/google-research/bleurt.git)
from bleurt.score import BleurtScorer

# Jiwer (pip install jiwer)
import jiwer.transforms as tr
import jiwer

# Warnings
import warnings

###########
## UTILS ##
###########

SAMPLE_HYPS = ["Hello there general kenobi", "foo bar foobar!"]
SAMPLE_REFS_SINGLE = ["hello there general kenobi", "foo bar foobar !"]
SAMPLE_REFS_MULT = [["hello there general kenobi", "hello there !"], ["Foo bar foobar", "foo bar foobar."]]

SAMPLE_HYPS_TK = [[9308, 43, 356, 214, 1841, 3090, 4505, 719, 1], [3167, 43, 2752, 3167, 43, 2227, 1310, 1]]
SAMPLE_REFS_SINGLE_TK = [[13902, 124, 356, 214, 1841, 3090, 4505, 719, 1], [3167, 43, 2752, 3167, 43, 2227, 5727, 1]]
SAMPLE_REFS_MULT_TK = [
    [[13902, 124, 356, 214, 1841, 3090, 4505, 719, 1], [13902, 124, 356, 214, 5727, 1]],
    [[1800, 43, 2752, 3167, 43, 2227, 1], [3167, 43, 2752, 3167, 43, 2227, 5, 1]]
]

_MULT_REF_ERROR_MSG = "The metric operates only with single reference."


def _hyp_typo_validator(hyp: str):
    '''Validates hypothesis string'''

    is_valid = isinstance(hyp, str)

    try: assert is_valid
    except AssertionError:
        raise ValueError("Hypothesis must be a string")

def _ref_typo_validator(ref: Union[List[str], str]):
    '''Validates references string'''

    is_valid = False

    if isinstance(ref, str):
        # If it's a string
        is_valid = True
    
    elif isinstance(ref, list):
        # If it's a list and all elements are strings
        is_valid = all(isinstance(r, str) for r in ref)

    else:
        is_valid = False
    
    try: assert is_valid
    except AssertionError:
        raise ValueError("References must be a list of strings or a string")

def _rouge_validator(rouge_method: str, rouge_metric: str):
    '''Validates rouge params'''

    try: assert rouge_method in ['l', '1', '2']
    except AssertionError:
        raise ValueError(f'"{rouge_method}" not implemented. Method must be "l", "1" or "2"')
    
    try: assert rouge_metric in ['f1', 'precision', 'recall']
    except AssertionError:
        raise ValueError(f'"{rouge_metric}" not implemented. Metric must be "f1", "precision" or "recall"')

def _bert_score_validator(bert_lang: str, bert_metric: str):
    '''Validates bert score params'''

    try: assert bert_lang in ['en', 'en-sci', 'zh', 'tr', 'others']
    except AssertionError:
        raise ValueError(f'"{bert_lang}" not implemented. Lang must be "en", "en-sci", "tr", "others" or "zh"')
    
    try: assert bert_metric in ['f1', 'precision', 'recall']
    except AssertionError:
        raise ValueError(f'"{bert_metric}" not implemented. Metric must be "f1", "precision" or "recall"')

def _mult_references_validator(refs: Union[List[List[str]], List[str], str]):
    '''Validates if it's multiple references'''

    return isinstance(refs, list) and all(isinstance(r, list) for r in refs)

###########################
## BASE CLASS (TEMPLATE) ##
###########################

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

        return sentence_bleu(references, hypothesis, weights, smoothing_function, auto_reweigh)

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
        
        return np.mean(scores)

################
## GLEU CLASS ##
################

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

        return sentence_gleu(references, hypothesis, min_len, max_len)

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
        
        return np.mean(scores)

#################
## ROUGE CLASS ##
#################

class Rouge(BaseMetric):
    """Rouge metric class"""

    def __init__(self):
        
        self.metric_map = {
            'f1': 'f',
            'precision': 'p',
            'recall': 'r',
        }

        self.rouge = RougeRaw()

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_MULT[0])
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def _get_score(self,
                   rouge_scores: List,
                   method: str,
                   metric: str) -> float:

        '''Get the ROUGE score of a specific method and metric.

            Params:
                rouge_scores (list): list of ROUGE scores
                method (str): 'l' or '1' or '2'
                metric (str): 'f1' or 'precision' or 'recall'

            Returns:
                ROUGE score (float)
        '''
        
        # Get first element that represents the scores
        scores = rouge_scores[0]

        # Get method scores
        method_scores = scores[f'rouge-{method}']

        # Get metric score
        score = method_scores[self.metric_map[metric]]

        return score

    def __call__(self,
                 hypothesis: str, 
                 references: Union[List[str], str],
                 method: str = 'l', 
                 metric: str = 'f1',
                 average: Callable = np.max) -> float:

        '''Compute ROUGE-N score of a hypothesis and a reference.

            Params:
                hypothesis (str): hypothesis sentence
                reference (list[str] or str): list of reference sentences or a reference sentence
                method (str): 'l' or '1' or '2'
                metric (str): 'f1' or 'precision' or 'recall'
                average (function): function used to average the scores of multiple references if applicable
                

            Returns:
                ROUGE-N score (float) from a hypothesis and reference(s)
        '''

        # Validate method and metric rouge params
        _rouge_validator(method, metric)

        if isinstance(references, str):
            references = [references]

        all_scores = list()
        # Iterate over each reference and apply average func
        for reference in references:

            # Calculate all Rouge scores
            scores = self.rouge.get_scores(hypothesis, reference)

            # Take specific score
            score = self._get_score(scores, method, metric)

            # Append score
            all_scores.append(score)
        
        # Apply average func
        score = float(average(all_scores))

        return score

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[Union[List[str], str]],
                  method: str = 'l', 
                  metric: str = 'f1',
                  average: Callable = np.max) -> float:

        '''Compute ROUGE-N score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[list[str] or str]): list of list of reference sentences or a list of reference sentence
                method (str): 'l' or '1' or '2'
                metric (str): 'f1' or 'precision' or 'recall'
                average (function): function used to average the scores of multiple references if applicable
                

            Returns:
                Mean ROUGE-N score (float) from a batch_hypotheses and batch_references
        '''

        scores = []

        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

            # Calculate score
            score = self(hyp, ref, method, metric, average)
            scores.append(score)
        
        return np.mean(scores)

######################
## BERT_SCORE CLASS ##
######################

class BertScore(BaseMetric):
    """BERT_SCORE metric class"""

    def __init__(self):
        
        self.bert_score = load_metric("bertscore")

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_MULT[0])
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def __call__(self,
                 hypothesis: Union[List[str], str], 
                 references: Union[List[str], str, List[List[str]]],
                 lang: str = 'en', 
                 metric: str = 'f1',
                 is_batch: bool = False,
                 **kwargs) -> Union[List[float], float]:

        '''Compute BERT_SCORE score of a hypothesis and a reference.
           See more at: https://github.com/Tiiiger/bert_score

            Params:
                hypothesis (list[str] or str): list of hypothesis sentence or a hypothesis sentence
                reference (list[list[str]] or list[str] or str): list of reference sentences or a reference sentence
                lang (str): 'en' or 'en-sci' or 'zh' or 'tr' or 'others'
                metric (str): 'f1' or 'precision' or 'recall'
                is_batch (bool): whether or not to return BERT_SCORE score for a batch of hypothesis and references or a single value
                kwargs: see complete list at: https://github.com/Tiiiger/bert_score/blob/master/bert_score/scorer.py#L29
                

            Returns:
                BERT_SCORE score (float) from a hypothesis and reference(s) or a list of BERT_SCORE score(s) for a batch of hypothesis and references
        '''

        # Validate lang and metric BERT_SCORE params
        _bert_score_validator(lang, metric)

        if isinstance(references, list) and isinstance(references[0], str) and isinstance(hypothesis, str):
            references = [references]

        if isinstance(references, str):
            references = [[references]]
        
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]

        scores = self.bert_score.compute(references=references, predictions=hypothesis, lang=lang, **kwargs)

        # Get score
        score = scores[metric]

        if is_batch:
            return score
        else:
            return score[0]

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[Union[List[str], str]],
                  lang: str = 'en', 
                  metric: str = 'f1',
                  **kwargs) -> float:

        '''Compute BERT_SCORE score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[list[str] or str]): list of list of reference sentences or a list of reference sentence
                lang (str): 'en' or 'en-sci' or 'zh' or 'tr' or 'others'
                metric (str): 'f1' or 'precision' or 'recall'
                kwargs: see complete list at: https://github.com/Tiiiger/bert_score/blob/master/bert_score/scorer.py#L29
                

            Returns:
                Mean BERT_SCORE score (float) from a batch_hypotheses and batch_references
        '''

        scores = []

        # Validates hypothesis and references
        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

        scores = self(batch_hypotheses, batch_references, lang, metric, is_batch=True, **kwargs)

        return np.mean(scores)

##################
## BLEURT CLASS ##
##################

class Bleurt(BaseMetric):
    """BLEURT metric class"""

    def __init__(self):
        
        self.bleurt = BleurtScorer()

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_MULT[0])
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def __call__(self,
                 hypothesis: Union[List[str], str], 
                 references: Union[List[str], str, List[List[str]]],
                 is_batch: bool = False,
                 average: Callable = np.max,
                 **kwargs) -> Union[List[float], float]:

        '''Compute BLEURT score of a hypothesis and a reference.
           See more at: https://github.com/google-research/bleurt

            Params:
                hypothesis (list[str] or str): list of hypothesis sentence or a hypothesis sentence
                reference (list[list[str]] or list[str] or str): list of reference sentences or a reference sentence
                is_batch (bool): whether or not to return BLEURT score for a batch of hypothesis and references or a single value
                average (function): function used to average the scores of multiple references if applicable
                kwargs: see complete list at: https://github.com/google-research/bleurt/blob/master/bleurt/score.py#L143
                

            Returns:
                BLEURT score (float) from a hypothesis and reference(s) or a list of BLEURT score(s) for a batch of hypothesis and references
        '''

        if isinstance(references, list) and isinstance(references[0], str) and isinstance(hypothesis, str):
            references = [references]

        if isinstance(references, str):
            references = [[references]]
        
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]

        scores = []
        for hyp, ref in zip(hypothesis, references):
            
            int_scores = []
            if isinstance(ref, list):
                for ref_sent in ref:
                    int_scores.append(self.bleurt.score(references=[ref_sent], candidates=[hyp], **kwargs))
                score = average(int_scores)
            else:
                score = self.bleurt.score(references=[ref], candidates=[hyp], **kwargs)
            
            # Add score
            scores.append(score)

        if is_batch:
            return scores
        else:
            return scores[0]

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[Union[List[str], str]],
                  average: Callable = np.max,
                  **kwargs) -> float:

        '''Compute BLEURT score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[list[str] or str]): list of list of reference sentences or a list of reference sentence
                average (function): function used to average the scores of multiple references if applicable
                kwargs: see complete list at: https://github.com/google-research/bleurt/blob/master/bleurt/score.py#L143
                

            Returns:
                Mean BLEURT score (float) from a batch_hypotheses and batch_references
        '''

        scores = []

        # Validates hypothesis and references
        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

        scores = self(batch_hypotheses, batch_references, is_batch=True, average=average, **kwargs)

        return np.mean(scores)
    
##################
## METEOR CLASS ##
##################

class Meteor(BaseMetric):
    """METEOR metric class"""

    def __init__(self):
        
        self.meteor = load_metric("meteor")

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

    def __call__(self,
                 hypothesis: Union[List[str], str], 
                 references: Union[List[str], str],
                 alpha: float = 0.9,
                 beta: float = 3,
                 gamma: float = 0.5) -> Union[List[float], float]:

        '''Compute METEOR score of a hypothesis and a reference.
           See more at: https://huggingface.co/metrics/meteor

            Params:
                hypothesis (list[str] or str): list of hypothesis sentence or a hypothesis sentence
                reference (or list[str] or str): list of reference sentences or a reference sentence
                alpha (float): Parameter for controlling relative weights of precision and recall
                beta (float): Parameter for controlling shape of penalty as a function of fragmentation
                gamma (float): Relative weight assigned to fragmentation penalty
                

            Returns:
                METEOR score (float) from a hypothesis and reference(s) or a list of METEOR score(s) for a batch of hypothesis and references
        '''

        if isinstance(references, str):
            references = [references]
        
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]

        score = self.meteor.compute(references=references, 
                                    predictions=hypothesis, 
                                    alpha=alpha,
                                    beta=beta,
                                    gamma=gamma)['meteor']

        return score

    def calculate(self,
                  batch_hypotheses: List[str],
                  batch_references: List[str],
                  alpha: float = 0.9,
                  beta: float = 3,
                  gamma: float = 0.5) -> float:

        '''Compute BLEURT score of a batch of hypothesis and references.

            Params:
                batch_hypotheses (list[str]): list of hypothesis sentences
                batch_references (list[str]): list of reference sentence
                alpha (float): Parameter for controlling relative weights of precision and recall
                beta (float): Parameter for controlling shape of penalty as a function of fragmentation
                gamma (float): Relative weight assigned to fragmentation penalty
                

            Returns:
                METEOR score (float) from a batch_hypotheses and batch_references
        '''

        assert not _mult_references_validator(batch_references), _MULT_REF_ERROR_MSG

        # Validates hypothesis and references
        for hyp, ref in zip(batch_hypotheses, batch_references):
            
            # Typo validations
            _hyp_typo_validator(hyp)
            _ref_typo_validator(ref)

        score = self(batch_hypotheses, batch_references, alpha=alpha, beta=beta, gamma=gamma)

        return score

##################################
## NLTK SCORES CLASS (TEMPLATE) ##
##################################

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

        return score

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

        return score

####################
## ACCURACY CLASS ##
####################

class Accuracy(NLTKScore):
    """ACCURACY metric class: inherits from NLTKScore class
    """

    def __init__(self):

        self.metric = accuracy
        
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

##################
## FSCORE CLASS ##
##################

class FScore(NLTKScore):
    """F1SCORE metric class: inherits from NLTKScore class
    """

    def __init__(self):

        self.metric = f_measure
        
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

#####################
## PRECISION CLASS ##
#####################

class Precision(NLTKScore):
    """PRECISION metric class: inherits from NLTKScore class
    """

    def __init__(self):

        self.metric = precision
        
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

##################
## RECALL CLASS ##
##################

class Recall(NLTKScore):
    """RECALL metric class: inherits from NLTKScore class
    """

    def __init__(self):

        self.metric = recall
        
        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

###################################
## JIWER SCORES CLASS (TEMPLATE) ##
###################################

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

        score = self.metric(truth=references, 
                            hypothesis=hypothesis,
                            truth_transform=self.truth_transform,
                            hypothesis_transform=self.hypothesis_transform)

        return score

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

        return score

###############
## WER CLASS ##
###############

class WER(JIWERScore):
    """WER metric class: inherits from JIWERScore class"""

    def __init__(self,
                 truth_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform,
                 hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform):

        self.metric = jiwer.wer
        self.truth_transform = truth_transform
        self.hypothesis_transform = hypothesis_transform

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

###############
## MER CLASS ##
###############

class MER(JIWERScore):
    """MER metric class: inherits from JIWERScore class"""

    def __init__(self,
                 truth_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform,
                 hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform):

        self.metric = jiwer.mer
        self.truth_transform = truth_transform
        self.hypothesis_transform = hypothesis_transform

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

###############
## WIP CLASS ##
###############

class WIP(JIWERScore):
    """WIP metric class: inherits from JIWERScore class"""

    def __init__(self,
                 truth_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform,
                 hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform):

        self.metric = jiwer.wip
        self.truth_transform = truth_transform
        self.hypothesis_transform = hypothesis_transform

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

###############
## WIL CLASS ##
###############

class WIL(JIWERScore):
    """WIL metric class: inherits from JIWERScore class"""

    def __init__(self,
                 truth_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform,
                 hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = jiwer.measures._default_transform):

        self.metric = jiwer.wil
        self.truth_transform = truth_transform
        self.hypothesis_transform = hypothesis_transform

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

####################################
## NLTK DISTANCE CLASS (TEMPLATE) ##
####################################

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

        return score

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

        return np.mean(scores)

###################################
## JARO WINKLER SIMILARITY CLASS ##
###################################

class JaroWinkler(NLTKDistance):
    """JARO_WINKLER metric class: inherits from NLTKDistance class"""

    def __init__(self):

        self.metric = jaro_winkler_similarity

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

###########################
## JARO SIMILARITY CLASS ##
###########################

class Jaro(NLTKDistance):
    """JARO metric class: inherits from NLTKDistance class"""

    def __init__(self):

        self.metric = jaro_similarity

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

#########################
## EDIT DISTANCE CLASS ##
#########################

class EditDistance(NLTKDistance):
    """EDIT_DISTANCE metric class: inherits from NLTKDistance class"""

    def __init__(self):

        self.metric = edit_distance

        self._health_validation(hypothesis=SAMPLE_HYPS[0], references=SAMPLE_REFS_SINGLE[0])

################
# METRICS LIST #
################

_METRICS = {
    'bleu': {
        'component': Bleu,
        'single_ref_only': False,
    },
    'gleu': {
        'component': Gleu,
        'single_ref_only': False,
    },
    'rouge': {
        'component': Rouge,
        'single_ref_only': False,
    },
    'bertscore': {
        'component': BertScore,
        'single_ref_only': False,
    },
    'bleurt': {
        'component': Bleurt,
        'single_ref_only': False,
    },
    'meteor': {
        'component': Meteor,
        'single_ref_only': True,
    },
    'accuracy': {
        'component': Accuracy,
        'single_ref_only': True,
    },
    'precision': {
        'component': Precision,
        'single_ref_only': True,
    },
    'recall': {
        'component': Recall,
        'single_ref_only': True,
    },
    'fscore': {
        'component': FScore,
        'single_ref_only': True,
    },
    'wer': {
        'component': WER,
        'single_ref_only': True,
    },
    'mer': {
        'component': MER,
        'single_ref_only': True,
    },
    'wip': {
        'component': WIP,
        'single_ref_only': True,
    },
    'wil': {
        'component': WIL,
        'single_ref_only': True,
    },
    'jaro': {
        'component': Jaro,
        'single_ref_only': True,
    },
    'jarowinkler': {
        'component': JaroWinkler,
        'single_ref_only': True,
    },
    'editdistance': {
        'component': EditDistance,
        'single_ref_only': True,
    },
}

_NAMES = list(_METRICS.keys())

def get_metrics_names():
    """Get metrics names"""
    return _NAMES

def get_metrics_data():
    """Get metrics data
    
        Returns:
            metrics_data (dict): metrics data (eg. {'bleu': {'component': object, 'single_ref_only': bool}, ...}) 
        """
    return _METRICS


###################
# METRICS WRAPPER #
###################

class MetricsCalculator():
    '''Metrics Calculator Wrapper'''

    def __str__(self):

        description = "Metrics Calculator Wrapper\n"
        description += "Metrics: {}\n".format(', '.join(_NAMES))
        description += "Computes multiple scores of text segments against one or more references.\n"
        description += "Usage:\n"
        description += "Params:\n"
        description += "\tmetrics: list of metrics to compute. Default: all\n"
        description += "\tmetric_params: dictionary of metric parameters. Default: metric's default parameters\n"
        description += "Returns:\n"
        description += "\tscores: dictionary of metrics and their scores\n"
        description += "Exemples:\n"
        description += ">>> metrics = ['gleu', 'wer']\n"
        description += ">>> metric_params = {'gleu': {'max_len':4}}\n"
        description += ">>> metrics_calculator = MetricsCalculator(metrics, metric_params)\n"
        description += ">>> hypothesis = [\"Hello there general kenobi\", \"foo bar foobar!\"]\n"
        description += ">>> references = [\"hello there general kenobi\", \"foo bar foobar !\"]\n"
        description += ">>> scores = metrics_calculator.compute_metrics(hypothesis, references)\n"
        description += ">>> print(scores)\n"
        description += "{'gleu': 0.919247009148487, 'wer': 0.375}"
        
        return description

    def __init__(self, 
                 metrics: List[str] = None,
                 metric_params: Dict[str, Any] = None):
        '''
        Params:
            metrics (list[str]): list of metrics to compute. Default: all
            metric_params (dict): dictionary of metric parameters. Default: metric's default parameters
                Example:
                    metric_params = {
                        'gleu': {
                            'min_len': 1, 
                            'max_len': 4
                            }
                        }
                            
        '''
        
        if metrics is None:
            self.metrics_names = _NAMES
        
        else:
            # Verify if metrics are valid
            assert all(metric in _NAMES for metric in metrics), f"Invalid metric. All the available metrics: {_NAMES}"
            self.metrics_names = metrics

        if metric_params is None:
            self.metric_params = {}
        else:
            # Verify if metrics are valid
            assert all(metric in _NAMES for metric in metric_params.keys()), f"Invalid metric for metric params. All the available metrics: {_NAMES}"
            self.metric_params = metric_params

        self.loaded = False
        self._load_metrics()

    def _load_metrics(self):
        '''Loads all metrics from the metrics module'''
        
        self.metrics = []
        for metric in self.metrics_names:
            self.metrics.append(_METRICS[metric]['component']())

        self.loaded = True

    def calculate_from_texts(self, 
                             hypothesis: List[str], 
                             references: Union[List[List[str]], List[str]], 
                             **kwargs) -> Dict[str, float]:
        '''Compute metrics from hypothesis and references texts.

            Params:
                hypothesis (list[str]): list of hypothesis
                references (list[list[str]] or list[str]): list of references

            Returns:
                Dictionary of metric names and scores
        '''

        # Scores dict
        scores = {}

        # Compute metrics
        for metric_class, metric_name in zip(self.metrics, self.metrics_names):

            # Selects metric params
            metric_params = None
            if self.metric_params is not None:
                metric_params = self.metric_params.get(metric_name, {})

            # Compute metric
            scores[metric_name] = metric_class.calculate(batch_hypotheses=hypothesis, batch_references=references, **metric_params)

        return scores

    def calculate_from_tokens(self, 
                             hypothesis_tokens: List[Any], 
                             references_tokens: Union[List[List[Any]], List[Any]],
                             tokenizer: Any = None, 
                             **kwargs) -> Dict[str, float]:
        '''Compute metrics from hypothesis and references texts.

            Params:
                hypothesis_tokens (list[Any]): list of hypothesis
                references_tokens (list[list[Any]] or list[Any]): list of references
                tokenizer (Any): tokenizer to use, must have a decoder method. Default: None
                kwargs: additional parameters to pass to the tokenizer

            Returns:
                Dictionary of metric names and scores
        '''

        # Assertions for the tokenizer
        assert tokenizer is not None, "'tokenizer' parameter cannot be None. Use a valid tokenizer."
        assert callable(getattr(tokenizer, 'decode', None)), "'tokenizer' must have a 'decode' method."

        # Detokenize hypothesis
        hypothesis = [tokenizer.decode(tokens, **kwargs) for tokens in hypothesis_tokens]

        # Detokenize references
        references = []
        for ref_tokens in references_tokens:
            if isinstance(ref_tokens[0], list):
                references.append([tokenizer.decode(tokens, **kwargs) for tokens in ref_tokens])
            else:
                references.append(tokenizer.decode(ref_tokens, **kwargs))

        # Calculate scores from texts
        return self.calculate_from_texts(hypothesis, references)