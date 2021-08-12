#############
## IMPORTS ##
#############

# typing
from typing import Union, List, Callable, Dict, Any

# metrics
from platiagro.metrics_nlp.metrics import get_metrics_names, get_metrics_data

_NAMES = get_metrics_names()
_METRICS = get_metrics_data()

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
