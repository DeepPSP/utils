# -*- coding: utf-8 -*-
"""
commonly used utilities, that do not belong to a particular category
"""
import os
import subprocess
import collections
import time
import re
import math
import json
from functools import reduce
from glob import glob
from copy import deepcopy
from logging import Logger
from datetime import datetime, timedelta
from typing import (
    Union, Optional, Any, NoReturn,
    Iterable, List, Tuple, Sequence, Dict, Callable,
)
from numbers import Real, Number

import numpy as np
from wfdb.io import _header
from wfdb import Record, MultiRecord


__all__ = [
    "ArrayLike", "ArrayLike_Float", "ArrayLike_Int",
    "DEFAULT_FIG_SIZE_PER_SEC",
    "idx2ts", "ms2samples", "samples2ms",
    "timestamp_to_local_datetime_string",
    "list_sum",
    "modulo", "gcd",
    "angle_d2r",
    "execute_cmd",
    "get_record_list_recursive",
    "get_record_list_recursive2",
    "get_record_list_recursive3",
    "clear_jupyter_notebook_outputs",
]


ArrayLike = Union[Sequence, np.ndarray]
ArrayLike_Float = Union[Sequence[float], np.ndarray]
ArrayLike_Int = Union[Sequence[int], np.ndarray]


DEFAULT_FIG_SIZE_PER_SEC = 4.8


def idx2ts(idx:int, start_ts:int, fs:int) -> int:
    """ finished, checked,
    
    Parameters:
    -----------
    idx, int,
        the index to be converted into timestamp
    start_ts, int,
        the timestamp of the point at index 0
    fs: int,
        sampling frequency

    Returns:
    --------
    int, the timestamp of the point at index `idx`
    """
    return int(start_ts + idx * 1000 // fs)


def ms2samples(t:Real, fs:Real) -> int:
    """ finished, checked,

    convert time `t` with units in ms to number of samples

    Parameters:
    -----------
    t: real number,
        time with units in ms
    fs: real number,
        sampling frequency of a signal

    Returns:
    --------
    n_samples: int,
        number of samples corresponding to time `t`
    """
    n_samples = t * fs // 1000
    return n_samples


def samples2ms(n_samples:int, fs:Real) -> Real:
    """ finished, checked,

    inverse function of `ms2samples`

    Parameters:
    -----------
    n_samples: int,
        number of sample points
    fs: real number,
        sampling frequency of a signal

    Returns:
    --------
    t: real number,
        time duration correponding to `n_samples`
    """
    t = n_samples * 1000 / fs
    return t


def timestamp_to_local_datetime_string(ts:int, ts_in_second:bool=False, fmt:str="%Y-%m-%d %H:%M:%S") -> str:
    """ finished, checked,

    Parameters:
    -----------
    ts: int,
        timestamp, in second or millisecond
    ts_in_second, bool, default False,
        if Ture, `ts` is in second, otherwise in millisecond
    fmt, str, default "%Y-%m-%d %H:%M:%S",
        the format of the output string

    Returns:
    --------
    str, the string form of `ts` in the form of `fmt`
    """
    from dateutil import tz

    if ts_in_second:
        utc = datetime.utcfromtimestamp(ts)
    else:
        utc = datetime.utcfromtimestamp(ts // 1000)
    
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    # Tell the datetime object that it's in UTC time zone since 
    # datetime objects are 'naive' by default
    utc = utc.replace(tzinfo=from_zone)

    # Convert time zone
    return utc.astimezone(to_zone).strftime(fmt)


def time_string_to_timestamp(time_string:str, fmt:str="%Y-%m-%d %H:%M:%S", return_second:bool=False) -> int:
    """ finished, checked,

    Parameters:
    -----------
    time_string: str,
        the time in the string format to be converted
    fmt: str, default "%Y-%m-%d %H:%M:%S",
        the format of `time_string`
    return_second: bool, default False,
        if True, the output is in second, otherwise in millisecond

    Returns:
    --------
    ts: int,
        timestamp, in second or millisecond, corr. to `time_string`
    """
    if return_second:
        ts = int(round(datetime.strptime(time_string, fmt).timestamp()))
    else:
        ts = int(round(datetime.strptime(time_string, fmt).timestamp()*1000))
    return ts


def list_sum(l:Sequence[list]) -> list:
    """ finished, checked,

    Parameters:
    -----------
    l: sequence of list,
        the sequence of lists to obtain the summation

    Returns:
    --------
    l_sum: list,
        sum of `l`,
        i.e. if l = [list1, list2, ...], then l_sum = list1 + list2 + ...
    """
    l_sum = reduce(lambda a,b: a+b, l, [])
    return l_sum


def modulo(val:Real, dividend:Real, val_range_start:Real=0) -> Real:
    """ finished, checked,

    find the value
        val mod dividend, within interval [val_range_start, val_range_start+abs(dividend)]

    Parameters:
    -----------
    val: real number,
        the number to be moduloed
    dividend: real number,
        the dividend
    returns:
    --------
    mod_val: real number,
        equals val mod dividend, within interval [val_range_start, val_range_start+abs(dividend)]
    """
    _dividend = abs(dividend)
    mod_val = val - val_range_start - _dividend*int((val-val_range_start)/_dividend)
    mod_val = mod_val + val_range_start if mod_val >= 0 else _dividend + mod_val + val_range_start
    return mod_val
    # alternatively
    # return (val-val_range_start)%_dividend + val_range_start


def gcd(l:Sequence[int]) -> int:
    """ finished, checked,

    greatest common divisor of a sequence of integers

    Parameters:
    -----------
    l: sequence of int,
        a sequence of integers
    
    Returns:
    --------
    int, the greatest common divisor of the integers in l;
    if l is empty, returns 0
    """
    assert all([isinstance(i, int) for i in l])
    return reduce(math.gcd, l, 0)


def angle_d2r(angle:Union[Real,np.ndarray]) -> Union[Real,np.ndarray]:
    """ finished, checked,
    
    Parameters:
    -----------
    angle: real number or ndarray,
        the angle(s) in degrees

    Returns:
    --------
    to writereal number or ndarray, the angle(s) in radians
    """
    return np.pi*angle/180.0


def execute_cmd(cmd:str, logger:Optional[Logger]=None, raise_error:bool=True) -> Tuple[int, List[str]]:
    """ finished, checked,

    execute shell command using `Popen`

    Parameters:
    -----------
    cmd: str,
        the shell command to be executed
    logger: Logger, optional,
    raise_error: bool, default True,
        if True, error will be raised when occured

    Returns:
    --------
    exitcode, output_msg: int, list of str,
        exitcode: exit code returned by `Popen`
        output_msg: outputs from `stdout` of `Popen`
    """
    shell_arg, executable_arg = True, None
    s = subprocess.Popen(
        cmd,
        shell=shell_arg,
        executable=executable_arg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    debug_stdout = collections.deque(maxlen=1000)
    if logger:
        logger.info("\n"+"*"*10+"  execute_cmd starts  "+"*"*10+"\n")
    while 1:
        line = s.stdout.readline().decode("utf-8", errors="replace")
        if line.rstrip():
            debug_stdout.append(line)
            if logger:
                logger.debug(line)
        exitcode = s.poll()
        if exitcode is not None:
            for line in s.stdout:
                debug_stdout.append(line.decode("utf-8", errors="replace"))
            if exitcode is not None and exitcode != 0:
                error_msg = " ".join(cmd) if not isinstance(cmd, str) else cmd
                error_msg += "\n"
                error_msg += "".join(debug_stdout)
                s.communicate()
                s.stdout.close()
                if logger:
                    logger.info("\n"+"*"*10+"  execute_cmd failed  "+"*"*10+"\n")
                if raise_error:
                    raise subprocess.CalledProcessError(exitcode, error_msg)
                else:
                    output_msg = list(debug_stdout)
                    return exitcode, output_msg
            else:
                break
    s.communicate()
    s.stdout.close()
    output_msg = list(debug_stdout)

    if logger:
        logger.info("\n"+"*"*10+"  execute_cmd succeeded  "+"*"*10+"\n")

    exitcode = 0

    return exitcode, output_msg


def get_record_list_recursive(db_dir:str, rec_ext:str) -> List[str]:
    """ finished, checked,

    get the list of records in `db_dir` recursively,
    for example, there are two folders 'patient1', 'patient2' in `db_dir`,
    and there are records 'A0001', 'A0002', ... in 'patient1'; 'B0001', 'B0002', ... in 'patient2',
    then the output would be 'patient1{sep}A0001', ..., 'patient2{sep}B0001', ...,
    sep is determined by the system

    Parameters:
    -----------
    db_dir: str,
        the parent (root) path of the whole database
    rec_ext: str,
        extension of the record files

    Returns:
    --------
    res: list of str,
        list of records, in lexicographical order
    """
    res = []
    db_dir = os.path.join(db_dir, "tmp").replace("tmp", "")  # make sure `db_dir` ends with a sep
    roots = [db_dir]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            res += [item for item in tmp if os.path.isfile(item)]
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    res = [os.path.splitext(item)[0].replace(db_dir, "") for item in res if item.endswith(rec_ext)]
    res = sorted(res)

    return res


def get_record_list_recursive2(db_dir:str, rec_pattern:str) -> List[str]:
    """ finished, checked,

    get the list of records in `db_dir` recursively,
    for example, there are two folders 'patient1', 'patient2' in `db_dir`,
    and there are records 'A0001', 'A0002', ... in 'patient1'; 'B0001', 'B0002', ... in 'patient2',
    then the output would be 'patient1{sep}A0001', ..., 'patient2{sep}B0001', ...,
    sep is determined by the system

    Parameters:
    -----------
    db_dir: str,
        the parent (root) path of the whole database
    rec_pattern: str,
        pattern of the record filenames, e.g. 'A*.mat'

    Returns:
    --------
    res: list of str,
        list of records, in lexicographical order
    """
    res = []
    db_dir = os.path.join(db_dir, "tmp").replace("tmp", "")  # make sure `db_dir` ends with a sep
    roots = [db_dir]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            # res += [item for item in tmp if os.path.isfile(item)]
            res += glob(os.path.join(r, rec_pattern), recursive=False)
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    res = [os.path.splitext(item)[0].replace(db_dir, "") for item in res]
    res = sorted(res)

    return res


def get_record_list_recursive3(db_dir:str, rec_patterns:Union[str,Dict[str,str]]) -> Union[List[str], Dict[str, List[str]]]:
    """ finished, checked,

    get the list of records in `db_dir` recursively,
    for example, there are two folders 'patient1', 'patient2' in `db_dir`,
    and there are records 'A0001', 'A0002', ... in 'patient1'; 'B0001', 'B0002', ... in 'patient2',
    then the output would be 'patient1{sep}A0001', ..., 'patient2{sep}B0001', ...,
    sep is determined by the system

    Parameters:
    -----------
    db_dir: str,
        the parent (root) path of the whole database
    rec_patterns: str or dict,
        pattern of the record filenames, e.g. "A(?:\d+).mat",
        or patterns of several subsets, e.g. `{"A": "A(?:\d+).mat"}`

    Returns:
    --------
    res: list of str,
        list of records, in lexicographical order
    """
    if isinstance(rec_patterns, str):
        res = []
    elif isinstance(rec_patterns, dict):
        res = {k:[] for k in rec_patterns.keys()}
    db_dir = os.path.join(db_dir, "tmp").replace("tmp", "")  # make sure `db_dir` ends with a sep
    roots = [db_dir]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            # res += [item for item in tmp if os.path.isfile(item)]
            if isinstance(rec_patterns, str):
                res += list(filter(re.compile(rec_patterns).search, tmp))
            elif isinstance(rec_patterns, dict):
                for k in rec_patterns.keys():
                    res[k] += list(filter(re.compile(rec_patterns[k]).search, tmp))
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    if isinstance(rec_patterns, str):
        res = [os.path.splitext(item)[0].replace(db_dir, "") for item in res]
        res = sorted(res)
    elif isinstance(rec_patterns, dict):
        for k in rec_patterns.keys():
            res[k] = [os.path.splitext(item)[0].replace(db_dir, "") for item in res[k]]
            res[k] = sorted(res[k])
    return res


def wfdb_rdheader(header_data:List[str]) -> Union[Record, MultiRecord]:
    """ finished, checked,
    
    modified from `wfdb.rdheader`

    Parameters
    ----------
    head_data: list of str,
        lines of the .hea header file
    """
    # Read the header file. Separate comment and non-comment lines
    header_lines, comment_lines = [], []
    for line in header_data:
        striped_line = line.strip()
        # Comment line
        if striped_line.startswith('#'):
            comment_lines.append(striped_line)
        # Non-empty non-comment line = header line.
        elif striped_line:
            # Look for a comment in the line
            ci = striped_line.find('#')
            if ci > 0:
                header_lines.append(striped_line[:ci])
                # comment on same line as header line
                comment_lines.append(striped_line[ci:])
            else:
                header_lines.append(striped_line)

    # Get fields from record line
    record_fields = _header._parse_record_line(header_lines[0])

    # Single segment header - Process signal specification lines
    if record_fields['n_seg'] is None:
        # Create a single-segment WFDB record object
        record = Record()

        # There are signals
        if len(header_lines)>1:
            # Read the fields from the signal lines
            signal_fields = _header._parse_signal_lines(header_lines[1:])
            # Set the object's signal fields
            for field in signal_fields:
                setattr(record, field, signal_fields[field])

        # Set the object's record line fields
        for field in record_fields:
            if field == 'n_seg':
                continue
            setattr(record, field, record_fields[field])
    # Multi segment header - Process segment specification lines
    else:
        # Create a multi-segment WFDB record object
        record = MultiRecord()
        # Read the fields from the segment lines
        segment_fields = _header._read_segment_lines(header_lines[1:])
        # Set the object's segment fields
        for field in segment_fields:
            setattr(record, field, segment_fields[field])
        # Set the objects' record fields
        for field in record_fields:
            setattr(record, field, record_fields[field])

        # Determine whether the record is fixed or variable
        if record.seg_len[0] == 0:
            record.layout = 'variable'
        else:
            record.layout = 'fixed'

    # Set the comments field
    record.comments = [line.strip(' \t#') for line in comment_lines]

    return record


def clear_jupyter_notebook_outputs(fp:str, dst:Optional[str]=None) -> NoReturn:
    """ finished, checked,

    clear outputs of a jupyter notebook,
    in cases where it is not able to be opened via jupyter

    Parameters:
    -----------
    fp: str,
        path of the jupyter notebook
    dst: str, optional,
        save destination of the contents of `fp` with outputs cleared
        if is None, `fp` will be used
    """
    try:
        with open(fp, "r") as f:
            contents = json.load(f)
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8") as f:
            contents = json.load(f)
    for cell in contents["cells"]:
        if "outputs" in cell.keys():
            cell["outputs"] = []
    new_fp = dst if dst else fp
    with open(new_fp, "w") as f:
        json.dump(contents, f)
