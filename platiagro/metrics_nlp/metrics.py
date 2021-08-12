## AVAILABLE METRICS ##

from platiagro.metrics_nlp.accuracy import Accuracy
from platiagro.metrics_nlp.precision import Precision
from platiagro.metrics_nlp.recall import Recall
from platiagro.metrics_nlp.fscore import FScore

from platiagro.metrics_nlp.bleu import Bleu
from platiagro.metrics_nlp.bleurt import Bleurt
from platiagro.metrics_nlp.gleu import Gleu
from platiagro.metrics_nlp.rouge import Rouge

from platiagro.metrics_nlp.edit_distance import EditDistance
from platiagro.metrics_nlp.jaro_winkler import JaroWinkler
from platiagro.metrics_nlp.jaro import Jaro

from platiagro.metrics_nlp.mer import MER
from platiagro.metrics_nlp.wer import WER
from platiagro.metrics_nlp.wil import WIL
from platiagro.metrics_nlp.wip import WIP

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
    'bleurt': {
        'component': Bleurt,
        'single_ref_only': False,
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