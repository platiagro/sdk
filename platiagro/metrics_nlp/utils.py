###########
## UTILS ##
###########

# typing
from typing import Union, List, Callable, Dict, Any

# Samples

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

# Validators

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

def _mult_references_validator(refs: Union[List[List[str]], List[str], str]):
    '''Validates if it's multiple references'''

    return isinstance(refs, list) and all(isinstance(r, list) for r in refs)

def _empty_values_score(hyp: str, refs: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    '''Returns max_val if both hyp and refs are empty, otherwise returns min_val'''
    
    if hyp == '' and refs == '':
        return max_val
    
    return min_val