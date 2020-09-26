# -*- coding: utf-8 -*-
"""
utilities for visualization
"""
from numbers import Real
from typing import Union, Optional, List, Tuple, Sequence, NoReturn, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation, rc
from IPython.display import display
from IPython.display import HTML
import ipywidgets as W
from easydict import EasyDict as ED

from ..common import ArrayLike


__all__ = [
    "plot_single_lead_ecg",
    "plot_hypnogram",
    "plot_confusion_matrix",
]


def plot_single_lead_ecg(s:ArrayLike, freq:Real, use_idx:bool=False, **kwargs) -> NoReturn:
    """ not finished,

    single lead ECG plot,

    Parameters:
    -----------
    s: array_like,
        the single lead ECG signal
    freq: real,
        sampling frequency of `s`
    use_idx: bool, default False,
        use idx instead of time for the x-axis
    kwargs: dict,
        keyword arguments, including
        - "waves": Dict[str, np.ndarray], consisting of
            "ppeaks", "qpeaks", "rpeaks", "speaks", "tpeaks",
            "ponsets", "poffsets", "qonsets", "soffsets", "tonsets", "toffsets"

    contributors: Jeethan, WEN Hao
    """
    default_fig_sz = 120
    line_len = freq * 25  # 25 seconds
    nb_lines, residue = divmod(len(s), line_len)
    waves = ED(kwargs.get("waves", ED()))
    if residue > 0:
        nb_lines += 1
    for idx in range(nb_lines):
        idx_start = idx*line_len
        idx_end = min((idx+1)*line_len, len(s))
        c = s[idx_start:idx_end]
        secs = np.arange(idx_start, idx_end)
        if not use_idx:
            secs = secs / freq
        mvs = np.array(c) * 0.001
        fig_sz = int(round(default_fig_sz * (idx_end-idx_start)/line_len))
        fig, ax = plt.subplots(figsize=(fig_sz, 6))
        ax.plot(secs, mvs, c='black')

        ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-1.5, 1.5)
        if waves:
            for w, w_indices in waves.items():
                epoch_w = [wi-idx_start for wi in w_indices if idx_start <= wi < idx_end]
                for wi in epoch_w:
                    ax.axvline(wi, linestyle='dashed', linewidth=0.7, color='magenta')
        if use_idx:
            plt.xlabel('Samples')
        else:
            plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        plt.show()


def plot_hypnogram(sleep_stage_curve:ArrayLike, style:str='original', **kwargs) -> NoReturn:
    """

    plot the hypnogram

    Parameters:
    -----------
    sleep_stage_curve: array_like,
        the sleep stage curve, each element is of the form 't, val',
        allowed stages are (case insensitive)
        - awake
        - REM
        - NREM1, NREM2, NREM3, NREM4
    style: str, default 'original'
        style of the hypnogram, can be the original style, or 'vspan'
    kwargs: dict,
        other key word arguments, including
        - ax: the axis to plot
    """
    all_stages = ['NREM4', 'NREM3', 'NREM2', 'NREM1', 'REM', 'awake',]
    all_stages = [item for item in all_stages if item.lower() in set([p[1].lower() for p in sleep_stage_curve])]
    all_stages = {all_stages[idx]:idx for idx in range(1,len(all_stages)+1)}

    palette = {
        'awake': 'orange',
        'REM': 'yellow',
        'NREM1': 'green',
        'NREM2': 'cyan',
        'NREM3': 'blue',
        'NREM4': 'purple',
    }
    patches = {k: mpatches.Patch(color=c, label=k) for k,c in palette.items()}

    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,12))

    if style == 'original':
        pass
    elif style == 'vspan':
        pass

    raise NotImplementedError


def plot_confusion_matrix(y_true:ArrayLike, y_pred:ArrayLike, classes:Sequence[str], normalize:Optional[str]=None, title:Optional[str]=None, save_path:Optional[str]=None, cmap:Optional[Any]=None) -> Any:
    """ finished, not checked,

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    In case `sklearn` has too low version to have this function

    Parameters:
    -----------
    to write
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)

    # try:
    #     from sklearn.metrics import ConfusionMatrixDisplay
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    #     disp = disp.plot(
    #         include_values=True,
    #         cmap=(cmap or plt.cm.Blues),
    #         xticks_rotation=30,
    #         values_format=('.2f' if normalize else 'd'),
    #     )
    #     return disp
    # except:
    #     pass

    if not title:
        if normalize:
            title = f'Normalized confusion matrix (along {normalize})'
        else:
            title = 'Confusion matrix'

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=(cmap or plt.cm.Blues))
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        # title=title,
        # ylabel='True label',
        # xlabel='Predicted label',
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("True severity",fontsize=14)
    ax.set_ylabel("Predicted severity",fontsize=14)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", 
                fontsize=18
            )
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

    return ax


class EcgAnimation(object):
    """ NOT finished,

    References:
    -----------
    [1] http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/
    [2] https://physionet.org/lightwave/
    """
    __name__ = "EcgAnimation"
    __SIGNAL_FORMATS__ = ["lead_first", "channel_first", "lead_last", "channel_last",]

    def __init__(self, signal:ArrayLike, freq:Real, fmt:Optional[str]=None) -> NoReturn:
        """ NOT finished,

        Parameters:
        -----------
        to write
        """
        self.signal = np.array(signal)
        self.freq = freq
        self.fmt = fmt.lower() if isinstance(fmt, str) else fmt
        assert fmt is None or fmt in self.__SIGNAL_FORMATS__

        self._fig, self._ax, self._line = None, None, None
        self._create_background()

        # self.goto_button = W.Button(description="refresh signal window")
        # self.Wout = W.Output()

    def _create_background(self) -> NoReturn:
        """
        """
        default_fig_sz = 120
        line_len = freq * 25  # 25 seconds
        fig, ax = plt.subplots(figsize=(fig_sz, 6))
        # ax.plot(secs, mvs, c='black')

        ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        # ax.set_ylim(-1.5, 1.5)
        raise NotImplementedError
