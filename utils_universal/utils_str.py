# -*- coding: utf-8 -*-
"""
docstring, to write
"""
import re
from typing import Union, Tuple, List, Sequence, NoReturn
from numbers import Number

import difflib
from fuzzysearch import find_near_matches, LevenshteinSearchParams
from fuzzywuzzy import process, fuzz


__all__ = [
    "LCSubStr",
    "dict_depth", "dict_to_str",
    "str2bool",
    "printmd",
    "local_fuzzy_match_1", "local_fuzzy_match_2", "local_fuzzy_match",
    "extract_chinese",
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


def local_fuzzy_match_1(query_string:str, large_string:str, threshold:float=0.8, best_only:bool=True, verbose:int=0) -> list:
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
    verbose: int, default 0,
        print verbosity

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
    if verbose >= 2:
        print("-"*100)
        print("-"*100)
        print("in local_fuzzy_match_1 (using `fuzz`)")
        print(f"input query_string = \042{query_string}\042")
        print(f"result of `process.extractBests` is {process.extractBests(query_string, (large_string,), score_cutoff=threshold)}")
    result = []
    local_scores = []
    for word, _ in process.extractBests(query_string, (large_string,), score_cutoff=threshold):
        max_l_dist = max(2, int(len(query_string)*(1-threshold)))
        search_params = \
            LevenshteinSearchParams(None,None,None,max_l_dist)
        if verbose >= 2:
            print("-"*100)
            print(f"search_params for `find_near_matches` are\n(max_substitutions = {search_params.unpacked[0]}, max_insertions = {search_params.unpacked[1]}, max_deletions = {search_params.unpacked[2]}, max_l_dist = {search_params.unpacked[3]}),")
            print(f"hence the method for `find_near_matches` is {_choose_search_class(search_params)}")
        for match in find_near_matches(query_string, word, max_l_dist=max_l_dist):
            match = word[match.start: match.end]
            start = large_string.find(match)
            result.append([match, start, start+len(match)])
            score = _chn_set_ratio(match, query_string)
            if verbose >= 2:
                print(f"match = \042{match}\042 for word = \042{word}\042, with score = {score}")
            local_scores.append(score)
    if best_only and len(result) > 0:
        best_idx = np.argmax(local_scores)
        result = result[best_idx]
    elif best_only and len(result) == 0:
        result = ["", -1, -1]
    if verbose >= 2:
        print(f"final result = {result}")
        print("-"*100)
        print("-"*100)
    return result


def local_fuzzy_match_2(query_string:str, large_string:str, threshold:float=0.8, best_only:bool=True, verbose:int=0) -> list:
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
    verbose: int, default 0,
        print verbosity

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
    _puncs_ = ["；", "。"]
    # words = large_string.split()
    words = []
    start = 0
    for s in re.finditer("|".join(_puncs_), large_string):
        end = s.start()
        words.append(large_string[start:end])
        start = s.end()
    if start < len(large_string):
        end = len(large_string)
        words.append(large_string[start:end])
    if verbose >= 2:
        print("-"*100)
        print("-"*100)
        print("in _local_fuzzy_match_2 (using `difflib`)")
        print(f"input query_string = \042{query_string}\042")
        print(f"`words` = {words}")
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
        ratio = len(extract_chinese(match)) / len(extract_chinese(query_string))
        if verbose >= 2:
            print("-"*100)
            print(f"match = \042{match}\042 for word = \042{word}\042, with ratio = {ratio}")
            print(f"the detailed matching blocks are {word_res}")
        if ratio >= threshold:
            start = min(starts)
            end = max(ends)
            match = large_string[start: end]
            result.append([match, start, end])
            score = _chn_set_ratio(match, query_string)
            local_scores.append(score)
    if best_only and len(result) > 0:
        best_idx = np.argmax(local_scores)
        result = result[best_idx]
    elif best_only and len(result) == 0:
        result = ["", -1, -1]
    if verbose >= 2:
        print(f"final result = {result}")
        print("-"*100)
        print("-"*100)
    return result


def local_fuzzy_match(query_string:str, large_string:str, threshold:float=0.8, verbose:int=0) -> list:
    """ finished, checked,

    fuzzy matches 'query_string' in 'large_string',
    merged from results obtained using `difflib` and from results using `fuzzywuzzy` and `fuzzysearch`

    Parameters:
    -----------
    query_string: str,
        the query string to find fuzzy matches in `large_string`
    large_string: str,
        the large string which contains potential fuzzy matches of `query_string`
    threshold: float, default 0.8,
        threshold of fuzzy matching
    verbose: int, default 0,
        print verbosity

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
    if threshold < 0.7:
        match, start, end = local_fuzzy_match_2(
            query_string, large_string, threshold,
            best_only=True, verbose=verbose,
        )
        return [match, start, end]
    match_1, start_1, end_1 = local_fuzzy_match_1(
        query_string, large_string, threshold,
        best_only=True, verbose=verbose,
    )
    match_2, start_2, end_2 = local_fuzzy_match_2(
        query_string, large_string, threshold,
        best_only=True, verbose=verbose,
    )
    if len(match_1) == 0:
        match, start, end = match_2, start_2, end_2
    elif len(match_2) == 0:
        match, start, end = match_1, start_1, end_1
    else:
        start = max(start_1, start_2)
        end = min(end_1, end_2)
        if start < end:
            match = large_string[start:end]
        else:
            score1 = \
                fuzz.token_set_ratio(match_1, large_string, force_ascii=False)
            score2 = \
                fuzz.token_set_ratio(match_2, large_string, force_ascii=False)
            if score1 > score2:
                match, start, end = match_1, start_1, end_1
            else:
                match, start, end = match_2, start_2, end_2
    result = [match, start, end]
    return result


def extract_chinese(texts:str) -> str:
    """ finished, checked,

    extract all Chinese characters (and arabic numbers) in `texts`

    Parameters:
    -----------
    texts: str,
        a string which contains Chinese characters and other characters

    Returns:
    --------
    pure_chinese_texts: str,
        the string which contains all Chinese characters (and arabic numbers) in `texts`,
        with ordering preserved
    """
    pure_chinese_texts = "".join(re.findall("[\u4e00-\u9FFF0-9]", texts))
    return pure_chinese_texts


def _chn_set_ratio(s1:str, s2:str) -> int:
    """
    """
    # consider fuzz.partial_token_set_ratio ?
    csr = fuzz.token_set_ratio(
        extract_chinese(s1),
        extract_chinese(s2),
        force_ascii=False,
    )
    return csr


def _choose_search_class(search_params:LevenshteinSearchParams) -> str:
    """
    modified from `fuzzysearch.choose_search_class`
    """
    max_substitutions, max_insertions, max_deletions, max_l_dist = search_params.unpacked

    # if the limitations are so strict that only exact matches are allowed,
    # use search_exact()
    if max_l_dist == 0:
        return "ExactSearch"

    # if only substitutions are allowed, use find_near_matches_substitutions()
    elif max_insertions == 0 and max_deletions == 0:
        return "SubstitutionsOnlySearch"

    # if it is enough to just take into account the maximum Levenshtein
    # distance, use find_near_matches_levenshtein()
    elif max_l_dist <= min(
        (max_substitutions if max_substitutions is not None else (1 << 29)),
        (max_insertions if max_insertions is not None else (1 << 29)),
        (max_deletions if max_deletions is not None else (1 << 29)),
    ):
        return "LevenshteinSearch"

    # if none of the special cases above are met, use the most generic version
    else:
        return "GenericSearch"
