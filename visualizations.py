import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numba.core import config

from utils import df_columns_to_arrays
from statistics import calculate_QQplot

def configure_axis(ax):
    if ax == None:
        ax = plt.gca()
    return ax

def scaled_tick_format(scale):
    tick_format = ticker.FuncFormatter(lambda x, pos: f"{x/scale :0g}")
    return tick_format

class SciNotationFormatter(ticker.ScalarFormatter):
    def __init__(self, exponent=0, fformat="%1.1f", offset=False, mathText=True):
        self.exponent = exponent
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.exponent
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format

def plot_histogram(dataframe, ax = None, **hist_kwargs):
    counts = df_columns_to_arrays(dataframe)
    labels = list(dataframe.columns)
    ax = configure_axis(ax)
    
    ax.hist(counts, label = labels, density = True, color = ["navy", "orangered"], edgecolor = "black", **hist_kwargs)

    ax.xaxis.set_major_formatter(scaled_tick_format(scale = 100))
    ax.yaxis.set_major_formatter(SciNotationFormatter(-4, "%1.1f"))

    return ax

def plot_experiment_histograms(experiment_dataframes):
    titles = ["Manual", "Imaris - No corrections", "Imaris - Corrected"]
    fig, axes = plt.subplots(3, 1, figsize = (6, 8), sharex = True)

    for i, (df, title, ax) in enumerate(zip(experiment_dataframes, titles, axes)):
        plot_histogram(df, ax, bins = "doane")
        ax.set_title(title, x = 0.97, y = 0.75, fontweight = "bold", fontsize = "medium", horizontalalignment = "right")
        ax.grid(alpha = 0.4)
        # Remove the scientific exponent x10^-4 for all subplots except the first
        if i!=0:
            plt.setp(ax.get_yaxis().get_offset_text(), visible = False)
    
    axes[0].legend(loc = "center right", fontsize = "small")
    fig.subplots_adjust(hspace = 0.07)
    fig.supxlabel("Fiber size ($\mu m^2 \\times 100$)", fontsize = "medium", y = 0.05)
    fig.supylabel("Fiber number density", fontsize = "medium", x = 0.01)

    return fig

def plot_QQ(sample_array1, sample_array2, ax = None, **scatter_kwargs):
    ax = configure_axis(ax)
    Q1, Q2, Q2_linearfit, r = calculate_QQplot(sample_array1, sample_array2)

    ax.scatter(Q1, Q2, color = "darkslateblue", label = "Q-Q plot", alpha = 0.8, **scatter_kwargs)
    ax.plot(Q1, Q2_linearfit, 'r-', label = "linear fit", linewidth = 3, alpha = 0.6)
    ax.plot(Q1, Q1, 'k--', label = "$y = x$", alpha = 0.7)

    ax.text(0.1, 0.8, f"PPCC: $r = {r :.3f}$", fontsize = "large", color = "firebrick", transform = ax.transAxes)

    ax.xaxis.set_major_formatter(scaled_tick_format(scale = 100))
    ax.yaxis.set_major_formatter(scaled_tick_format(scale = 100))

    major_tick_loc = ticker.MultipleLocator(base = 2000)
    ax.xaxis.set_major_locator(major_tick_loc)
    ax.yaxis.set_major_locator(major_tick_loc)
    ax.grid(alpha = 0.7)
    
    return ax

def plot_QQplot_comparison(experiment_dataframes, experiment_labels):
    group_labels = [name.capitalize() for name in experiment_dataframes[0].columns]
    exp_arrays = [df_columns_to_arrays(df) for df in experiment_dataframes]
    
    fig, axes = plt.subplots(2, 2, figsize = (9, 9))
    for i in range(2):
        for j, group_name in enumerate(group_labels):
            ax = axes[i, j]
            plot_QQ(exp_arrays[i+1][j], exp_arrays[0][j], ax)
            ax.set_xlabel(f"Quantiles: {experiment_labels[i+1]} / {group_name}")
            ax.set_ylabel(f"Quantiles: {experiment_labels[0]} / {group_name}")
            ax.set_aspect("equal")
    axes[0, 0].legend(loc = "lower right", fontsize = "small")
    fig.tight_layout()
    return fig