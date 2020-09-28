# -*- coding: utf-8 -*-
"""
utilities for visualization
"""
import math
from numbers import Real
from typing import Union, Optional, List, Tuple, Sequence, NoReturn, Any

import numpy as np
import matplotlib as mpl
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



class EcgAnimation(animation.FuncAnimation):
    """ NOT finished,

    References:
    -----------
    [1] http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/
    [2] https://physionet.org/lightwave/
    [3] https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html
    [4] https://kapernikov.com/ipywidgets-with-matplotlib/
    """
    __name__ = "EcgAnimation"
    __SIGNAL_FORMATS__ = ["lead_first", "channel_first", "lead_last", "channel_last",]
    __default_duration_anim__ = 5  # 5 seconds

    def __init__(self, signal:ArrayLike, freq:Real, fmt:Optional[str]=None) -> NoReturn:
        """ NOT finished,

        Parameters:
        -----------
        to write
        """
        rc('animation', html='jshtml')

        self.signal = np.array(signal)
        if self._auto_infer_units() == "mV":
            self.signal = 1000 * self.signal
        self.freq = freq
        self.fmt = fmt.lower() if isinstance(fmt, str) else fmt
        assert (self.signal.ndim == 1 and fmt is None) \
            or (self.signal.ndim == 2 and fmt in self.__SIGNAL_FORMATS__)

        if self.signal.ndim == 2:
            raise NotImplementedError("not implemented for multi-lead signals currently")

        if self.signal.ndim==1 or (self.signal.ndim==2 and self.fmt in ["lead_last", "channel_last"]):
            self.siglen = self.signal.shape[0]
            self.duration = self.siglen / self.freq
        elif self.signal.ndim==2 and self.fmt in ["lead_first", "channel_first"]:
            self.siglen = self.signal.shape[1]
            self.duration = self.siglen / self.freq
        
        self._n_frames = max(
            1, math.ceil((self.duration-self.__default_duration_anim__)*self.freq)
        )

        default_fig_sz = 20
        self._fig, self._ax = plt.subplots(figsize=(default_fig_sz, 4))
        self._line = None
        self._create_background()
        
        super().__init__(
            fig=self._fig,
            func=self._animate,
            frames=self._n_frames,
            repeat=False,
            cache_frame_data=False,
            blit=True,
        )
        self._display()
        # self.anim = HTML(animation.FuncAnimation(
        #     fig=self._fig,
        #     func=self._animate,
        #     frames=self._n_frames,
        #     repeat=False,
        #     blit=True,
        # ).to_jshtml())

    def _display(self):
        """
        """
        return HTML(self.to_jshtml())

    def _create_background(self,) -> NoReturn:
        """
        """
        self._ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        self._ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        self._ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        self._ax.yaxis.set_major_locator(plt.MultipleLocator(500))
        self._ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))
        self._ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        self._ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        self._ax.set_xlim(0, self.__default_duration_anim__)
        self._ax.set_ylim(-np.max(np.abs(self.signal))*1.2, np.max(np.abs(self.signal))*1.2)
        self._ax.set_xlabel('Time [s]')
        self._ax.set_ylabel('Voltage [μV]')
        self._line, = self._ax.plot([], [])

    def _animate(self, frame:int) -> Tuple[mpl.artist.Artist]:
        """
        The required signature is::

            def func(frame, *fargs) -> iterable_of_artists
        """
        x_start = frame
        x_end = int(x_start + self.freq*self.__default_duration_anim__)
        x = np.linspace(
            start=x_start/self.freq,
            stop=x_end/self.freq,
            num=int(self.__default_duration_anim__*self.freq),
        )
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
            units of `data`, 'μV' or 'mV'
        """
        _MAX_mV = 20  # 20mV, seldom an ECG device has range larger than this value
        max_val = np.max(self.signal) - np.min(self.signal)
        if max_val > _MAX_mV:
            units = 'μV'
        else:
            units = 'mV'
        return units



# class EcgAnimation(W.VBox):
#     """ NOT finished,

#     References:
#     -----------
#     [1] http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/
#     [2] https://physionet.org/lightwave/
#     [3] https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html
#     [4] https://kapernikov.com/ipywidgets-with-matplotlib/
#     """
#     __name__ = "EcgAnimation"
#     __SIGNAL_FORMATS__ = ["lead_first", "channel_first", "lead_last", "channel_last",]
#     __default_duration_anim__ = 5  # 5 seconds

#     def __init__(self, signal:ArrayLike, freq:Real, fmt:Optional[str]=None) -> NoReturn:
#         """ NOT finished,

#         Parameters:
#         -----------
#         to write
#         """
#         super().__init__()
#         output = W.Output()
#         rc('animation', html='jshtml')

#         self.signal = np.array(signal)
#         if self._auto_infer_units() == "mV":
#             self.signal = 1000 * self.signal
#         self.freq = freq
#         self.fmt = fmt.lower() if isinstance(fmt, str) else fmt
#         assert (self.signal.ndim == 1 and fmt is None) \
#             or (self.signal.ndim == 2 and fmt in self.__SIGNAL_FORMATS__)

#         if self.signal.ndim == 2:
#             raise NotImplementedError("not implemented for multi-lead signals currently")

#         if self.signal.ndim==1 or (self.signal.ndim==2 and self.fmt in ["lead_last", "channel_last"]):
#             self.siglen = self.signal.shape[0]
#             self.duration = self.siglen / self.freq
#         elif self.signal.ndim==2 and self.fmt in ["lead_first", "channel_first"]:
#             self.siglen = self.signal.shape[1]
#             self.duration = self.siglen / self.freq
        
#         self._frame_freq = max(25, self.freq//5)
#         self._n_frames = max(
#             1, math.ceil((self.duration-self.__default_duration_anim__)/self._frame_freq)
#         )

#         with output:
#             default_fig_sz = 20
#             self._fig, self._ax = plt.subplots(figsize=(default_fig_sz, 4))
#             self._line = None
#             self._create_background()
#             self.anim = HTML(animation.FuncAnimation(
#                 fig=self._fig,
#                 func=self._animate,
#                 frames=self._n_frames,
#                 repeat=False,
#                 blit=True,
#             ).to_jshtml())

#         # # define (init) widgets
#         # color_picker = W.ColorPicker(
#         #     value="blue", 
#         #     description="pick a color"
#         # )
#         # time_choice = W.FloatSlider(
#         #     value=0,
#         #     min=0,
#         #     max=self.duration,
#         #     description="s",
#         # )
#         # controls = W.VBox([
#         #     color_picker,
#         #     time_choice,
#         # ])
#         # controls.layout = self._make_box_layout()

#         # out_box = W.Box([output])
#         # output.layout = self._make_box_layout()

#         # # observe stuff
#         # time_choice.observe(self._update_start_time, "value")  # "value": param of `W.FloatSlider`
#         # color_picker.observe(self._update_line_color, "value")  # "value": param of `W.ColorPicker`

#         out_box = W.Box([output])
#         output.layout = self._make_box_layout()
#         # add to children
#         self.children = [output]


#     # def _update_start_time(self, change:dict) -> NoReturn:
#     #     """
#     #     """
#     #     self._start_time = change.new

#     # def _update_line_color(self, change:dict) -> NoReturn:
#     #     """
#     #     """
#     #     self._line.set_color(change.new)

#     def _create_background(self,) -> NoReturn:
#         """
#         """
#         self._ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
#         self._ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
#         self._ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
#         self._ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
#         self._ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
#         self._ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
#         self._ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#         self._ax.set_xlim(0, self.__default_duration_anim__)
#         # ax.set_ylim(-np.max(np.abs(self.signal))*1.2, np.max(np.abs(self.signal))*1.2)
#         self._ax.set_xlabel('Time [s]')
#         self._ax.set_ylabel('Voltage [μV]')
#         self._line, = self._ax.plot([], [])

#     def _animate(self, frame_idx:int) -> Tuple[mpl.artist.Artist]:
#         """
#         The required signature is::

#             def func(frame, *fargs) -> iterable_of_artists
#         """
#         # x_start = int(self._start_time*self.freq + frame_idx*self._frame_freq)
#         # x_start = int(start_time*self.freq + frame_idx*self._frame_freq)
#         x_start = int(frame_idx*self._frame_freq)
#         x_end = int(x_start + self.freq*self.__default_duration_anim__)
#         x = np.linspace(
#             start=x_start/self.freq,
#             end=x_end/self.freq,
#             num=int(self.__default_duration_anim__*self.freq),
#         )
#         y = np.append(
#             self.signal[x_start: min(self.siglen,x_end)],
#             np.full(shape=(x_end-self.siglen,), fill_value=np.nan)
#         )
#         self._line.set_data(x, y)
#         return (self._line,)

#     def _make_box_layout(self):
#         """
#         """
#         l = W.Layout(
#             border='solid 1px black',
#             margin='0px 10px 10px 0px',
#             padding='5px 5px 5px 5px'
#         )
#         return l

#     def _auto_infer_units(self) -> str:
#         """ finished, checked,

#         automatically infer the units of `data`,

#         Returns:
#         --------
#         units: str,
#             units of `data`, 'μV' or 'mV'
#         """
#         _MAX_mV = 20  # 20mV, seldom an ECG device has range larger than this value
#         max_val = np.max(self.signal) - np.min(self.signal)
#         if max_val > _MAX_mV:
#             units = 'μV'
#         else:
#             units = 'mV'
#         return units
