# -*- coding: utf-8 -*-
"""
docstring, to write
"""
from typing import Union, Tuple, List, Sequence, NoReturn
from numbers import Number

import difflib
from fuzzysearch import find_near_matches
from fuzzywuzzy import process, fuzz


__all__ = [
    "LCSubStr",
    "dict_depth", "dict_to_str",
    "str2bool",
    "printmd",
    "local_fuzzy_match_1", "local_fuzzy_match_2",
]


def LCSubStr(X:str, Y:str) -> Tuple[int, List[str]]:
    """ finished, checked,

    find the longest common sub-strings of two strings,
    with complexity O(mn), m=len(X), n=len(Y)

    Parameters:
    -----------
    X, Y: str,
        the two strings to extract the longest common sub-strings

    Returns:
    --------
    lcs_len, lcs: int, list of str,
        the longest length, and the list of longest common sub-strings

    Reference:
    ----------
    https://www.geeksforgeeks.org/longest-common-substring-dp-29/
    """
    m, n = len(X), len(Y)
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]

    # To store the length of
    # longest common substring
    lcs_len = 0
    lcs = []
    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                if LCSuff[i][j] > lcs_len:
                    lcs_len = LCSuff[i][j]
                    lcs = [Y[j-lcs_len:j]]
                elif LCSuff[i][j] == lcs_len:
                    lcs_len = LCSuff[i][j]
                    lcs.append(Y[j-lcs_len:j])
            else:
                LCSuff[i][j] = 0
    return lcs_len, lcs


def dict_depth(d:dict) -> int:
    """ finished, checked,

    find the 'depth' of a (possibly) nested dict

    Parameters:
    -----------
    d: dict,
        a (possibly) nested dict
    
    Returns:
    --------
    depth: int,
        the 'depth' of `d`
    """
    try:
        depth = 1+max([dict_depth(v) for _,v in d.items() if isinstance(v, dict)])
    except:
        depth = 1
    return depth


def dict_to_str(d:Union[dict, list, tuple], current_depth:int=1, indent_spaces:int=4) -> str:
    """ finished, checked,

    convert a (possibly) nested dict into a `str` of json-like formatted form,
    this nested dict might also contain lists or tuples of dict (and of str, int, etc.)

    Parameters:
    -----------
    d: dict, or list, or tuple,
        a (possibly) nested `dict`, or a list of `dict`
    current_depth: int, default 1,
        depth of `d` in the (possible) parent `dict` or `list`
    indent_spaces: int, default 4,
        the indent spaces of each depth

    Returns:
    --------
    s: str,
        the formatted string
    """
    assert isinstance(d, (dict, list, tuple))
    if len(d) == 0:
        s = f"{{}}" if isinstance(d, dict) else f"[]"
        return s
    # flat_types = (Number, bool, str,)
    flat_types = (Number, bool,)
    flat_sep = ", "
    s = "\n"
    unit_indent = " "*indent_spaces
    prefix = unit_indent*current_depth
    if isinstance(d, (list, tuple)):
        if all([isinstance(v, flat_types) for v in d]):
            len_per_line = 110
            current_len = len(prefix) + 1  # + 1 for a comma 
            val = []
            for idx, v in enumerate(d):
                add_v = f"\042{v}\042" if isinstance(v, str) else str(v)
                add_len = len(add_v) + len(flat_sep)
                if current_len + add_len > len_per_line:
                    val = ", ".join([item for item in val])
                    s += f"{prefix}{val},\n"
                    val = [add_v]
                    current_len = len(prefix) + 1 + len(add_v)
                else:
                    val.append(add_v)
                    current_len += add_len
            if len(val) > 0:
                val = ", ".join([item for item in val])
                s += f"{prefix}{val}\n"
        else:
            for idx, v in enumerate(d):
                if isinstance(v, (dict, list, tuple)):
                    s += f"{prefix}{dict_to_str(v, current_depth+1)}"
                else:
                    val = f"\042{v}\042" if isinstance(v, str) else v
                    s += f"{prefix}{val}"
                if idx < len(d) - 1:
                    s += ",\n"
                else:
                    s += "\n"
    elif isinstance(d, dict):
        for idx, (k, v) in enumerate(d.items()):
            key = f"\042{k}\042" if isinstance(k, str) else k
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{key}: {dict_to_str(v, current_depth+1)}"
            else:
                val = f"\042{v}\042" if isinstance(v, str) else v
                s += f"{prefix}{key}: {val}"
            if idx < len(d) - 1:
                s += ",\n"
            else:
                s += "\n"
    s += unit_indent*(current_depth-1)
    s = f"{{{s}}}" if isinstance(d, dict) else f"[{s}]"
    return s


def str2bool(v:Union[str, bool]) -> bool:
    """ finished, checked,

    converts a 'boolean' value possibly in the format of str to bool

    Parameters:
    -----------
    v: str or bool,
        the 'boolean' value

    Returns:
    --------
    b: bool,
        `v` in the format of bool

    References:
    -----------
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       b = v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        b = True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        b = False
    else:
        raise ValueError('Boolean value expected.')
    return b


def printmd(md_str:str) -> NoReturn:
    """ finished, checked,

    printing bold, colored, etc., text

    Parameters:
    -----------
    md_str: str,
        string in the markdown style

    References:
    -----------
    [1] https://stackoverflow.com/questions/23271575/printing-bold-colored-etc-text-in-ipython-qtconsole
    """
    try:
        from IPython.display import Markdown, display
        display(Markdown(md_str))
    except:
        print(md_str)


def local_fuzzy_match_1(query_string:str, large_string:str, threshold:float=0.8, best_only:bool=True) -> list:
    """ finished, checked,

    fuzzy matches `query_string` in `large_string`, using `fuzzywuzzy` and `fuzzysearch`

    Parameters:
    -----------
    query_string: str,
        the query string to find fuzzy matches in `large_string`
    large_string: str,
        the large string which contains potential fuzzy matches of `query_string`
    threshold: float, default 0.8,
        threshold of fuzzy matching
    best_only: bool, default True,
        if True, only the best match will be returned

    Returns:
    --------
    result: list,
        3-element list (if `best_only` is True):
            - matched text in `large_string`,
            - start index of the matched text in `large_string`
            - end index of the matched text in `large_string`
        or list of such 3-element list (`best_only` is False)

    Reference:
    ----------
    https://stackoverflow.com/questions/17740833/checking-fuzzy-approximate-substring-existing-in-a-longer-string-in-python
    """
    result = []
    local_scores = []
    for word, _ in process.extractBests(query_string, (large_string,), score_cutoff=threshold):
        max_l_dist = max(1, int(len(query_string)*(1-threshold)))
        for match in find_near_matches(query_string, word, max_l_dist=max_l_dist):
            match = word[match.start:match.end]
            start = large_string.find(match)
            result.append([match, start, start+len(match)])
            local_scores.append(fuzz.ratio(match, query_string))
    if best_only and len(result) > 0:
        best_idx = np.argmax(local_scores)
        result = result[best_idx]
    return result


def local_fuzzy_match_2(query_string:str, large_string:str, threshold:float=0.8, best_only:bool=True) -> list:
    """ finished, checked,

    fuzzy matches 'query_string' in 'large_string', using `difflib`

    Parameters:
    -----------
    query_string: str,
        the query string to find fuzzy matches in `large_string`
    large_string: str,
        the large string which contains potential fuzzy matches of `query_string`
    threshold: float, default 0.8,
        threshold of fuzzy matching
    best_only: bool, default True,
        if True, only the best match will be returned

    Returns:
    --------
    result: list,
        3-element list (if `best_only` is True):
            - matched text in `large_string`,
            - start index of the matched text in `large_string`
            - end index of the matched text in `large_string`
        or list of such 3-element list (`best_only` is False)

    Reference:
    ----------
    https://stackoverflow.com/questions/17740833/checking-fuzzy-approximate-substring-existing-in-a-longer-string-in-python
    """
    words = large_string.split()
    result = []
    local_scores = []
    for word in words:
        s = difflib.SequenceMatcher(None, word, query_string)
        word_res = []
        starts = []
        ends = []
        for i, j, n in s.get_matching_blocks():
            if not n:
                continue
            word_res.append(word[i:i+n])
            starts.append(i+large_string.find(word))
            ends.append(i+n+large_string.find(word))
        match = ''.join(word_res)
        if len(match) / float(len(query_string)) >= threshold:
            start = min(starts)
            end = max(ends)
            match = large_string[start:end]
            result.append([match, start, end])
            local_scores.append(fuzz.ratio(match, query_string))
    if best_only and len(result) > 0:
        best_idx = np.argmax(local_scores)
        result = result[best_idx]
    return result
