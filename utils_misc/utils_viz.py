# -*- coding: utf-8 -*-
"""
utilities for visualization
"""
import math, time
from numbers import Real
from typing import Union, Optional, List, Tuple, Sequence, NoReturn, Any

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation, rc
from IPython.display import display
from IPython.display import HTML
import ipywidgets as W
from easydict import EasyDict as ED

from ..common import ArrayLike, DEFAULT_FIG_SIZE_PER_SEC


__all__ = [
    "plot_single_lead_ecg",
    "plot_hypnogram",
    "plot_confusion_matrix",
    "EcgAnimation"，
]


def plot_single_lead_ecg(s:ArrayLike, fs:Real, use_idx:bool=False, **kwargs:Any) -> NoReturn:
    """ NOT finished, NOT checked,

    single lead ECG plot,

    Parameters:
    -----------
    s: array_like,
        the single lead ECG signal
    fs: real,
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
    line_len = fs * 25  # 25 seconds
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
            secs = secs / fs
        mvs = np.array(c) * 0.001
        fig_sz = int(round(default_fig_sz * (idx_end-idx_start)/line_len))
        fig, ax = plt.subplots(figsize=(fig_sz, 6))
        ax.plot(secs, mvs, color="black")

        ax.axhline(y=0, linestyle="-", linewidth="1.0", color="red")
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")
        ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-1.5, 1.5)
        if waves:
            for w, w_indices in waves.items():
                epoch_w = [wi-idx_start for wi in w_indices if idx_start <= wi < idx_end]
                for wi in epoch_w:
                    ax.axvline(wi, linestyle="dashed", linewidth=0.7, color="magenta")
        if use_idx:
            plt.xlabel("Samples")
        else:
            plt.xlabel("Time [s]")
        plt.ylabel("Voltage [mV]")
        plt.show()


def plot_hypnogram(sleep_stage_curve:ArrayLike, style:str="original", **kwargs:Any) -> NoReturn:
    """ NOT finished, NOT checked,

    plot the hypnogram

    Parameters:
    -----------
    sleep_stage_curve: array_like,
        the sleep stage curve, each element is of the form "t, val",
        allowed stages are (case insensitive)
        - awake
        - REM
        - NREM1, NREM2, NREM3, NREM4
    style: str, default "original"
        style of the hypnogram, can be the original style, or "vspan"
    kwargs: dict,
        other key word arguments, including
        - ax: the axis to plot
    """
    all_stages = ["NREM4", "NREM3", "NREM2", "NREM1", "REM", "awake",]
    all_stages = [item for item in all_stages if item.lower() in set([p[1].lower() for p in sleep_stage_curve])]
    all_stages = {all_stages[idx]:idx for idx in range(1,len(all_stages)+1)}

    palette = {
        "awake": "orange",
        "REM": "yellow",
        "NREM1": "green",
        "NREM2": "cyan",
        "NREM3": "blue",
        "NREM4": "purple",
    }
    patches = {k: mpatches.Patch(color=c, label=k) for k,c in palette.items()}

    ax = kwargs.get("ax", None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,12))

    if style == "original":
        pass
    elif style == "vspan":
        pass

    raise NotImplementedError


def plot_confusion_matrix(y_true:ArrayLike,
                          y_pred:ArrayLike,
                          classes:Sequence[str],
                          normalize:Optional[str]=None,
                          title:Optional[str]=None,
                          save_path:Optional[str]=None,
                          cmap:Optional[Any]=None) -> Any:
    """ finished, NOT checked,

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    In case `sklearn` has too low version to have this function

    Parameters:
    -----------
    y_true: array_like,
        array of ground truths
    y_pred: array_like,
        array of predictions
    classes: sequence of str,
        sequence of names of the classes
    normalize: str, optional, case insensitive,
        can be one of "true", "pred", "all", or None
        if not None, normalizes confusion matrix over the true (rows),
        predicted (columns) conditions or all the population
    title: str, optional,
        title of the plot of confusion matrix
    save_path: str, optional,
        path to save the plot of the confusion matrix
    cmap: optional,
        colormap,
        if not specified, defaults to `plt.cm.Blues`

    Returns:
    --------
    ax: `plt.axes.Axes`
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
    #         values_format=(".2f" if normalize else "d"),
    #     )
    #     return disp
    # except:
    #     pass

    if not title:
        if normalize:
            title = f"Normalized confusion matrix (along {normalize})"
        else:
            title = "Confusion matrix"

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=(cmap or plt.cm.Blues))
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        # title=title,
        # ylabel="True label",
        # xlabel="Predicted label",
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("True severity",fontsize=14)
    ax.set_ylabel("Predicted severity",fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
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
        plt.savefig(save_path, bbox_inches="tight", transparent=True)

    return ax



class EcgAnimation(object):
    """ NOT finished, currently usable but two slow and might have bugs,

    TODO: add more widgets and implement for multi-lead signals; acceleration!!!

    References:
    -----------
    [1] http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/
    [2] https://physionet.org/lightwave/
    [3] https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html
    [4] https://kapernikov.com/ipywidgets-with-matplotlib/
    """
    __name__ = "EcgAnimation"
    __SIGNAL_FORMATS__ = ["lead_first", "channel_first", "lead_last", "channel_last",]
    __default_duration_anim__ = 10  # 10 seconds

    def __init__(self, signal:ArrayLike, fs:Real, fmt:Optional[str]=None, ticks_granularity:int=0) -> NoReturn:
        """ NOT finished,

        Parameters:
        -----------
        signal: array_like,
            the input signal (1d for single lead signal, 2d for multi-lead signal)
        fs: real number,
            sampling frequency of the input signal
        fmt: str, optional,
            format of the input signal,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks),
            the higher this value is, the slower the animator is
        """
        rc("animation", html="jshtml")

        timer = time.time()

        self.signal = np.array(signal)
        if self._auto_infer_units() == "mV":
            self.signal = 1000 * self.signal
        self.fs = fs
        self.dt = 0.05  # second per frame
        self.frame_fs = int(1/self.dt)
        self.fmt = fmt.lower() if isinstance(fmt, str) else fmt
        self.ticks_granularity = ticks_granularity
        assert (self.signal.ndim == 1 and fmt is None) \
            or (self.signal.ndim == 2 and fmt in self.__SIGNAL_FORMATS__)

        if self.signal.ndim == 2:
            raise NotImplementedError("not implemented for multi-lead signals currently")

        if self.signal.ndim==1 or (self.signal.ndim==2 and self.fmt in ["lead_last", "channel_last"]):
            self.siglen = self.signal.shape[0]
            self.duration = self.siglen / self.fs
        elif self.signal.ndim==2 and self.fmt in ["lead_first", "channel_first"]:
            self.siglen = self.signal.shape[1]
            self.duration = self.siglen / self.fs
        
        self._n_frames = max(
            1, math.ceil((self.duration-self.__default_duration_anim__)*self.frame_fs)
        )

        default_fig_sz = DEFAULT_FIG_SIZE_PER_SEC * self.__default_duration_anim__
        self._fig, self._ax = plt.subplots(figsize=(default_fig_sz, 4))
        self._line = None
        self._create_background()

        print(f"_create_background cost {time.time() - timer:.5f} seconds")
        timer = time.time()

        self._anim = animation.FuncAnimation(
            fig=self._fig,
            func=self._animate,
            # frames=self._n_frames,
            interval=1000*self.dt,
            repeat=False,
            cache_frame_data=False,
            blit=True,
            # save_count=20,
        )

        print(f"instantiation of FuncAnimation cost {time.time() - timer:.5f} seconds")


    def display(self):
        """
        """
        timer = time.time()
        w = HTML(self._anim.to_jshtml())
        print(f"instantiation of FuncAnimation cost {time.time() - timer:.5f} seconds")
        return w

    def _create_background(self,) -> NoReturn:
        """
        """
        self._ax.axhline(y=0, linestyle="-", linewidth="1.0", color="red")
        if self.ticks_granularity >= 1:
            self._ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
            self._ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            self._ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")
        if self.ticks_granularity >= 2:
            self._ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
            self._ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))
            self._ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
        self._ax.set_xlim(0, self.__default_duration_anim__)
        self._ax.set_ylim(-np.max(np.abs(self.signal))*1.2, np.max(np.abs(self.signal))*1.2)
        self._ax.set_xlabel("Time [s]")
        self._ax.set_ylabel("Voltage [μV]")
        self._line, = self._ax.plot([], [])

    def _animate(self, frame:int) -> Tuple[mpl.artist.Artist]:
        """
        The required signature is::

            def func(frame, *fargs) -> iterable_of_artists
        """
        x_start = frame * self.frame_fs
        x_end = int(x_start + self.fs*self.__default_duration_anim__)
        x = np.linspace(
            start=x_start/self.fs,
            stop=x_end/self.fs,
            num=int(self.__default_duration_anim__*self.fs),
        )
        self._ax.set_xlim(x_start/self.fs, x_end/self.fs)
        if x_end <= self.siglen:
            y = self.signal[x_start: x_end]
        else:
            y = np.append(
                self.signal[x_start: min(self.siglen,x_end)],
                np.full(shape=(x_end-self.siglen,), fill_value=np.nan)
            )
        self._line.set_data(x, y)
        return (self._line,)

    def _auto_infer_units(self) -> str:
        """ finished, checked,

        automatically infer the units of `data`,

        Returns:
        --------
        units: str,
            units of `data`, "μV" or "mV"
        """
        _MAX_mV = 20  # 20mV, seldom an ECG device has range larger than this value
        max_val = np.max(self.signal) - np.min(self.signal)
        if max_val > _MAX_mV:
            units = "μV"
        else:
            units = "mV"
        return units
