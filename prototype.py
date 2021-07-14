#%%
%reload_ext autoreload
%autoreload 2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from utils import *
from statistics import descr_stat_func, A_statistic, KS_statistic, calculate_QQplot
from bootstrap import bootstrap_confidence_interval, bootstrap_confidence_interval_2samp
from visualizations import plot_experiment_histograms, plot_histogram, plot_QQ, plot_QQplot_comparison

plt.rcParams["font.size"] = 14
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.labelsize"] = "medium"

# %%
fed_fast_filepaths = ["./data/fed-fast manual.xlsx", "./data/fed-fast automated.xlsx", "./data/fed-fast corrected.xlsx"]
fed_fast_dataset = import_data_list(fed_fast_filepaths)

shLacz_filepaths = ["./data/shLacz manual.xlsx", "./data/shLacz automated.xlsx", "./data/shLacz corrected Aviv.xlsx"]
shLacz_dataset = import_data_list(shLacz_filepaths)

shCAPN1_filepaths = ["./data/shCAPN1 manual Aviv.xlsx", "./data/shCAPN1 automated.xlsx", "./data/shCAPN1 corrected.xlsx"]
shCAPN1_dataset = import_data_list(shCAPN1_filepaths)
# %%
fig = plot_experiment_histograms(fed_fast_dataset)
#plt.savefig("./figures/figure3B.png")
# %%
fig = plot_experiment_histograms(shLacz_dataset)
#plt.savefig("./figures/figure5B.png")
# %%
fig = plot_experiment_histograms(shCAPN1_dataset)
#plt.savefig("./figures/figure4B.png")
# %%
sample1, sample2 = df_columns_to_arrays(fed_fast_dataset[0])

# %%
import scipy.stats as scstats
scstats.ks_2samp(sample1, sample2)
# %%
KS_statistic(sample1, sample2)
# %%
output = bootstrap_confidence_interval(sample1, descr_stat_func["mean"], 20000, CI_algorithm = "BCa")
# %%
stat = Statistic(*output, "mean")
print(stat)

# %%
def _calculate_stat_diff(statistic1, statistic2):
    val1 = statistic1.value
    val2 = statistic2.value
    return 100*(val2 - val1)/val1

def descr_stat_comparison_table(dataframe, statistic = "mean", bootstrap_sample_nb = 20000, random_seed = 10):
    # Assume first column corresponds to the control
    sample1 = np.array(dataframe.iloc[:,0], dtype = np.float32)
    sample2 = np.array(dataframe.iloc[:,1], dtype = np.float32)
    col_names = list(dataframe.columns)

    np.random.seed(random_seed)
    stat_func = descr_stat_func[statistic]

    out1 = bootstrap_confidence_interval(sample1, stat_func, 20000, CI_algorithm = "BCa", random_seed = random_seed)
    out2 = bootstrap_confidence_interval(sample2, stat_func, 20000, CI_algorithm = "BCa", random_seed = random_seed)
    stat1, stat2 = Statistic(*out1, statistic), Statistic(*out2, statistic)
    diff = _calculate_stat_diff(stat1, stat2)
    
    col_names.append("Change (%)")
    result_df = pd.DataFrame([[str(stat1), str(stat2), f"{diff :.2f}"]])
    result_df.columns = col_names
    result_df.index = [statistic.capitalize()]
    return result_df

# %%
from matplotlib.patches import Rectangle
def plot_experiment_comparision_histogram(ax, dataframe):
 
    ax = plot_histogram(dataframe, ax, bins = "doane")
    ax.legend()
    ax.set_xlabel("Fiber size ($\mu m^2 \\times 100$)")
    ax.set_ylabel("Fiber number probability")

    statistic_names = ["median", "mean", "stdev", "skewness"]
    df = pd.concat([descr_stat_comparison_table(dataframe, stat) for stat in statistic_names])
    tbl = pd.plotting.table(ax, df, loc = "bottom", cellLoc = "center", bbox=[0.0, -0.6, 1.0, 0.4])
    
    tbl.auto_set_font_size(False)
    #tbl.set_fontsize(14)
    tbl.scale(1, 5)
    tbl.auto_set_column_width([0, 1, 2, 3])
    return ax
# %%
def descriptive_stat_table(dataframe):
    statistic_names = ["median", "mean", "stdev", "skewness"]
    descr_stat_table = pd.concat([descr_stat_comparison_table(dataframe, stat) for stat in statistic_names])
    return descr_stat_table
# %%
out = descriptive_stat_table(shLacz_dataset[2])
out.to_csv("./figures/shLacz_corrected.csv")
# %%
shLacz_dataset[2].shape
# %%
dataset = fed_fast_dataset[0]
fig, ax = plt.subplots(figsize = (10, 10))
ax = plot_experiment_comparision_histogram(ax, dataset)
ax.set_title("Manual analysis")
plt.tight_layout()
#plt.savefig("./figures/fed-fast_manual.png")
# %%
dataset = shCAPN1_dataset[0]
fig, ax = plt.subplots(figsize = (10, 10))
ax = plot_experiment_comparision_histogram(ax, dataset)
ax.set_title("Manual analysis")
plt.tight_layout()
plt.savefig("./figures/shCAPN1_manualAviv.png")
# %%
dataset = shLacz_dataset[3]
fig, ax = plt.subplots(figsize = (10, 10))
ax = plot_experiment_comparision_histogram(ax, dataset)
ax.set_title("Imaris - corrected (Nadav)")
plt.tight_layout()
plt.savefig("./figures/shLacz_correctedAviv.png")
# %%
def plot_empirical_distribution(dataframe_list, labels, density = True):
    x_scale = 100
    x_ticks = ticker.FuncFormatter(lambda x, pos: f"{x/x_scale :0g}")
    hist_kwargs = {"edgecolor": "black"}
    experiment_names = list(map(str.capitalize, dataframe_list[0].columns))

    fig, axes = plt.subplots(1, 2, figsize = (16, 6))
    for i, ax in enumerate(axes):
        hist_data = list(map(lambda x: x.iloc[:, i], dataframe_list))
        ax.hist(hist_data, bins = 18, density = density, rwidth = 0.7, label = labels, **hist_kwargs)
        ax.xaxis.set_major_formatter(x_ticks)
        ax.ticklabel_format(axis = "y", style = "sci", scilimits = (0, 0))
        ax.set_xlim((0, 9000))
        ax.legend()

        ax.set_xlabel("Fiber size ($\mu m^2 \\times 100$)")
        if density:
            ax.set_ylabel("Fiber number probability")
        else:
            ax.set_ylabel("Fiber number counts")
        ax.set_title(experiment_names[i])

    fig.suptitle("Empirical distributions of the mouse muscle fiber sizes")
    plt.tight_layout()

    return fig, axes
# %%
fig, axes = plot_empirical_distribution(fed_fast_dataset, labels = ["manual", "automated", "corrected"], density = False)
#plt.savefig("./figures/fed_fast - empirical count distribution.png")
# %%
fig, axes = plot_empirical_distribution(fed_fast_dataset, labels = ["manual", "automated", "corrected"])
#plt.savefig("./figures/fed_fast - empirical probability distribution.png")

# %%
# %%
fig = plot_QQplot_comparison(fed_fast_dataset, ["Manual", "Automated", "Corrected"])
##plt.savefig("./figures/figure3C.png")

# %%
fig, axes = plot_QQplot(fed_fast_dataset[0], fed_fast_dataset[2], ["Manual", "Corrected"])
#plt.savefig("./figures/fed_fast - corrected QQplot.png")
# %%
def bootstrap_A_from_df(dataframe, random_seed = 10):
    data_arrays = df_columns_to_arrays(dataframe)
    bootstrap_result = bootstrap_confidence_interval_2samp(data_arrays[1], data_arrays[0], A_statistic, B = 20000, CI_algorithm = "BCa", random_seed = random_seed)
    return bootstrap_result
# %%
dataset = shLacz_dataset
A_stats = [bootstrap_A_from_df(df) for df in dataset]
exp_labels = ["Manual", "Automated", "Corrected"]
# %%
A_val = np.array([A[0] for A in A_stats])
A_dist = np.stack([A[1] for A in A_stats])
A_err = np.array([A[2] for A in A_stats])
# %%
A_val
# %%
from matplotlib import ticker
def plot_column_scatter(means, values, CIs, experiment_labels, experiment_name, statistic_name, p_values):
    fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    
    x_means = np.arange(len(means))
    errs = np.abs(CIs.T - means)
    ax.errorbar(x_means, means, errs, linestyle = '', marker = '.', color = 'black', capsize = 4.0)
    
    Dx_scatter = np.random.uniform(low = -0.2, high = 0.2, size = values.shape)
    for i, x_mean in enumerate(x_means):
        ax.scatter(x_mean+Dx_scatter[i], values[i], s = 6.0, facecolors = None, alpha = 0.3)
    ax.axhline(y = 0.5, linestyle = "--", color = "firebrick", linewidth = 2.0, alpha = 0.8)

    ax.set_xticks(x_means)
    ax.set_xticklabels(experiment_labels, fontsize = "small")
    ax.set_ylabel(f"{statistic_name}\n({experiment_name} vs non-transfected)", fontsize = "small")

    x_margin = 0.4
    ax.set_xlim((x_means[0]-x_margin, x_means[-1]+x_margin))
    y_lim = (0, 1)
    ax.set_ylim(y_lim)

    for i, p in enumerate(p_values):
        if p < 0.001:
            ax.text(i, 0.87*y_lim[1], "***", horizontalalignment = "center")
    ax.text(2.3, 0.04*y_lim[1], "***: $p < 0.001$ for the\nBrunner-Munzel test", fontsize = "small", horizontalalignment = "right", bbox = {"facecolor": "white", "edgecolor": "grey", "alpha": 0.85})
    ax.grid(alpha = 0.5)

    fig.tight_layout()
    return fig
# %%
p_vals = [0.94, 0, 0.33]
fig = plot_column_scatter(A_val, A_dist, A_err, exp_labels, "shLacz", "A statistic", p_vals )
#plt.savefig("./figures/figure5C.png")
# %%
# %%
p_vals = [0, 0.86, 0]
fig = plot_column_scatter(A_val, A_dist, A_err, exp_labels, "shCAPN1", "A statistic", p_vals )
#plt.savefig("./figures/figure4C.png")
# %%
def plot_statistics_bar(values, errors, experiment_labels, experiment_name, statistic_name, p_values):
    x = np.arange(len(A_stats))
    
    fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.bar(x, values, yerr = errors, color = "palegreen", edgecolor = "black", capsize = 4.0)
    #ax.axhline(y = 0.5, linestyle = "--", color = "firebrick", linewidth = 2.0, alpha = 0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_labels, fontsize = "small")
    ax.set_ylabel(f"{statistic_name}\n({experiment_name} vs non-transfected)", fontsize = "small")
    y_lim = (0, 0.2)
    for i, p in enumerate(p_values):
        if p < 0.001:
            ax.text(i, 0.85*y_lim[1], "***", horizontalalignment = "center")
    ax.grid(alpha = 0.3)
    ax.set_ylim(y_lim)
    ax.text(0.02, 0.04*y_lim[1], "***: $p < 0.001$ for the\nKolmogorov-Smirnoff test", fontsize = "small", horizontalalignment = "left", bbox = {"facecolor": "white", "edgecolor": "grey", "alpha": 0.85})
    major_tick_loc = ticker.MultipleLocator(base = 0.05)
    ax.yaxis.set_major_locator(major_tick_loc)

    fig.tight_layout()
    return fig
# %%
p_vals = [0, 0.86, 0]
fig = plot_statistics_bar(A_val, A_err, exp_labels, "shCAPN1", "A statistic", p_vals)
#plt.savefig("column ./figures/figure4C.png")
# %%
p_vals = [0.94, 0, 0.33]
fig = plot_statistics_bar(A_val, A_err, exp_labels, "shLacz", "A statistic", p_vals)
#plt.savefig("./figures/figure5C.png")
# %%
import scipy.stats as scstats
dataset = shLacz_dataset
KS_test_results = np.array([scstats.ks_2samp(df_columns_to_arrays(df)[1], df_columns_to_arrays(df)[0]) for df in dataset])
KS_vals = KS_test_results[:,0]
p_vals = KS_test_results[:,1]

fig = plot_statistics_bar(KS_vals, None, exp_labels, "shLacz", "KS statistic", p_vals)
#plt.savefig("./figures/figure5D.png")
# %%
p_vals
# %%
sample1 = np.array(fed_fast_dataset[0].iloc[:,0], dtype = float)
sample2 = np.array(fed_fast_dataset[0].iloc[:,1], dtype = float)
ks, p = scstats.ks_2samp(sample1, sample2)
# %%
es, p = scstats.epps_singleton_2samp(sample1, sample2)
# %%
p_vals
# %%
p
# %%
scstats.mstats.spearmanr(sample1, sample2)
# %%
data_ind = 2
sample1, sample2 = shLacz_dataset[data_ind].iloc[:,1], shLacz_dataset[data_ind].iloc[:,0]
sample1, sample2 = np.array(sample1, np.float32), np.array(sample2, np.float32)
scstats.mstats.brunnermunzel(sample1, sample2, alternative = "two-sided", distribution = "normal")
#A_statistic(sample1, sample2)
# %%
output = bootstrap_confidence_interval_2samp(sample1, sample2, A_statistic, B = 20000, CI_algorithm = "BCa")
#%%
stat = Statistic(*output, "A_statistic")
print(stat)
# %%
data_ind = 2
sample1, sample2 = fed_fast_dataset[data_ind].iloc[:,0], fed_fast_dataset[data_ind].iloc[:,1]
sample1, sample2 = np.array(sample1, np.float32), np.array(sample2, np.float32)
scstats.mstats.brunnermunzel(sample1, sample2, alternative = "greater", distribution = "normal")
# %%
A_statistic(sample1, sample2)
# %%
output = bootstrap_confidence_interval_2samp(sample1, sample2, A_statistic, B = 20000, CI_algorithm = "BCa")
#%%
stat = Statistic(*output, "A_statistic")
print(stat)
# %%
import scipy.stats as scstats
data_ind = 2
sample1, sample2 = shCAPN1_dataset[data_ind].iloc[:,1], shCAPN1_dataset[data_ind].iloc[:,0]
sample1, sample2 = np.array(sample1, np.float32), np.array(sample2, np.float32)
scstats.mstats.brunnermunzel(sample1, sample2, alternative = "greater", distribution = "normal")
# %%
A_statistic(sample1, sample2)
# %%
output = bootstrap_confidence_interval_2samp(sample1, sample2, A_statistic, B = 20000, CI_algorithm = "BCa")
#%%
stat = Statistic(*output, "A_statistic")
print(stat)

# %%


# %%
# %%

# %%
col_ind = 1
sample0 = np.array(fed_fast_dataset[0].iloc[:, col_ind], dtype = float)
sample1 = np.array(fed_fast_dataset[1].iloc[:, col_ind], dtype = float)
sample2 = np.array(fed_fast_dataset[2].iloc[:, col_ind], dtype = float)

wdist1 = scstats.wasserstein_distance(sample0, sample1)
wdist2 = scstats.wasserstein_distance(sample0, sample2)


# %%
edist1 = scstats.energy_distance(sample0, sample1)
edist2 = scstats.energy_distance(sample0, sample2)
# %%
edist1
# %%
edist2
# %%
kolmogorov_distance(sample0, sample1)
# %%
def calculate_distance_metrics(dataframe1, dataframe2, column_index, metric_functions, metric_names):
    data1 = np.array(dataframe1.iloc[column_index], float)
    data2 = np.array(dataframe2.iloc[column_index], float)

    metrics_dict = {name: f(data1, data2) for f, name in zip(metric_functions, metric_names)}
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient = "index")
    return metrics_df

def kolmogorov_distance(u_values, v_values):
    ks, _ = scstats.ks_2samp(u_values, v_values)
    return ks

metric_functions = [scstats.wasserstein_distance, scstats.energy_distance, kolmogorov_distance]
metric_names = ["Wasserstein", "Energy", "Kolmogorov"]

# %%
def plot_metrics_table(dataframe_list, label_list, metric_functions, metric_names):
    fig, axes = plt.subplots(1, 2, figsize = (9, 3))
    exp_names = dataframe_list[0].columns

    for i, ax in enumerate(axes):
        output = pd.concat([calculate_distance_metrics(dataframe_list[0], df, i, metric_functions, metric_names) for df in dataframe_list[1:]], axis = 1) 
        data = output.values
        table = ax.table(cellText = np.around(data, 2), colLabels = label_list[1:], rowLabels = metric_names, loc = "center", cellLoc = "center")
        table.scale(1, 2)

        ax.axis("off")
        ax.axis("tight")
        ax.set_title(exp_names[i])
    
    return fig, axes
# %%
scstats.wasserstein_distance(sample0, sample1)
# %%
metrics = calculate_distance_metrics(fed_fast_dataset[0], fed_fast_dataset[1], 0, metric_functions, metric_names)
# %%
metrics
# %%
fig, axes = plot_metrics_table(fed_fast_dataset, labels, metric_functions, metric_names)
fig.suptitle("Distance metrics between the manual distribution")
plt.tight_layout()
# %%

# %%


# %%
fig, ax = plt.subplots(figsize = (5, 5))
ax = plot_QQ(*df_columns_to_arrays(fed_fast_dataset[0]))
ax.set_title("Q-Q plot for the manual analysis")
ax.set_xlabel("Quantiles: Fed")
ax.set_ylabel("Quantiles: Fasted")
plt.tight_layout()
#plt.savefig("./figures/fed_fast - QQplot manual.png")
# %%
k, _ = scstats.ks_2samp(sample0, sample1)
k
# %%
kolmogorov_distance(sample0, sample2)
# %%
Dx = 3.0
x = np.linspace(-4, 4, 100)
norm1 = scstats.norm()
norm2 = scstats.norm(loc = Dx)
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.plot(x, norm1.pdf(x), label = 'before')
ax.plot(x+Dx, norm2.pdf(x+Dx), label = 'after')
ax.set_xlim((-5, 5+Dx))
ax.set_ylim((-0.01, 1))
ax.legend()
plt.savefig("./figures/distribution_shift.png")
# %%
x = np.linspace(-4, 4, 100)
dist1 = scstats.norm()
dist2 = scstats.gumbel_l(loc = 1.5)
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.plot(x, dist1.pdf(x), label = 'before')
ax.plot(x, dist2.pdf(x), label = 'after')
ax.set_xlim((-5, 5))
ax.set_ylim((-0.01, 1))
ax.legend()
plt.savefig("./figures/distribution_skew.png")
# %%
import matplotlib.pyplot as plt
import scipy.stats as scstats

def Supplementary_Figure1A():
    x = np.linspace(-4, 4, 100)
    norm1 = scstats.norm(scale = 0.5)
    norm2 = scstats.norm(scale = 1.0)
    
    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    ax.plot(x, norm1.pdf(x), linewidth = 2.0)
    ax.fill_between(x, norm1.pdf(x), alpha = 0.5, label = "Method 1: $N(0, 0.5^2)$")
    ax.plot(x, norm2.pdf(x), linewidth = 2.0)
    ax.fill_between(x, norm2.pdf(x), alpha = 0.5, label = "Method 2: $N(0, 1^2)$")
    
    #ax.axvline(x = 0.0, color = "black", linewidth = 2.0, linestyle = "--", alpha = 0.4)
    ax.text(0.0, 0.9, "Identical means (0.0)\nDifferent stdev (0.5 vs 1.0)", horizontalalignment = "center", fontsize = "small")
    
    ax.set_ylim((0, 1.5))
    ax.set_xlim((-5, 5))
    
    ax.grid(alpha = 0.4)
    ax.legend(fontsize = "small")
    
    return fig
fig = Supplementary_Figure1A()
#plt.savefig("./figures/supplementary_figure1A.png")
# %%
def Supplementary_Figure1B():
    x = np.linspace(-5, 5, 100)
    sigma1, sigma2 = 0.5, 2.0
    ratio = 0.4
    norm1_1 = scstats.norm(scale = sigma1)
    norm1_2 = scstats.norm(scale = sigma2)
    var = ratio*sigma1**2+(1-ratio)*sigma2**2
    sigma = np.sqrt(var)
    norm2 = scstats.norm(scale = sigma)
    y1 = ratio*norm1_1.pdf(x)+(1-ratio)*norm1_2.pdf(x)
    y2 = norm2.pdf(x)
    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    ax.plot(x, y1, linewidth = 2.0)
    ax.fill_between(x, y1, alpha = 0.5, label = f"Method 1: ${ratio} \cdot N(0, {sigma1}^2)+{1-ratio} \cdot N(0, {sigma2}^2)$")
    ax.plot(x, y2, linewidth = 2.0)
    ax.fill_between(x, y2, alpha = 0.5, label = f"Method 2: $N(0, {var})$")
    
    #ax.axvline(x = 0.0, color = "black", linewidth = 2.0, linestyle = "--", alpha = 0.4)
    text = f"Identical means (0.0)\nIdentical medians (0.0)\nIdentical stdev ({sigma :.2f})\nIdentical skewness (0.0)"
    ax.text(0.0, 0.5, text, horizontalalignment = "center", fontsize = "small")
    
    ax.set_ylim((0, 1.0))
    ax.set_xlim((-6, 6))
    
    ax.grid(alpha = 0.4)
    ax.legend(fontsize = "small")
    
    return fig
fig = Supplementary_Figure1B()
#plt.savefig("./figures/supplementary_figure1B.png")
# %%
def Supplementary_Figure2():
    fig, axes = plt.subplots(1, 2, figsize = (8, 3))
    x = np.linspace(-4, 4, 100)

    def _shifted_distributions(ax):
        Dx = 3.0
        norm1 = scstats.norm()
        norm2 = scstats.norm(loc = Dx)
        y1 = norm1.pdf(x)
        y2 = norm2.pdf(x+Dx)
        ax.plot(x, y1)
        ax.fill_between(x, y1, alpha = 0.5, label = 'Before')
        ax.plot(x+Dx, y2)
        ax.fill_between(x+Dx, y2, alpha = 0.5, label = 'After')
        ax.set_xlim((-5, 5+Dx))
        ax.legend()
        return ax

    def _skewed_distributions(ax):
        dist1 = scstats.norm()
        dist2 = scstats.gumbel_l(loc = 2.0)
        y1 = dist1.pdf(x)
        y2 = dist2.pdf(x)
        ax.plot(x, y1)
        ax.fill_between(x, y1, alpha = 0.5, label = 'Before')
        ax.plot(x, y2)
        ax.fill_between(x, y2, alpha = 0.5, label = 'After')
        ax.set_xlim((-5, 5))
        ax.legend()
        return ax

    axes[0] = _shifted_distributions(axes[0])
    axes[0].set_title("Distribution shifted", fontsize = "medium")
    axes[1] = _skewed_distributions(axes[1])
    axes[1].set_title("Skewness change induced", fontsize = "medium")
    for ax in axes:
        ax.set_ylim((-0.01, 0.7))
        ax.grid(alpha = 0.4)
    return fig

fig = Supplementary_Figure2()
fig.tight_layout()
#plt.savefig("./figures/supplementary_figure2.png")
# %%
def Supplementary_Figure3():
    fig, axes = plt.subplots(1, 3, figsize = (12, 3))
    x = np.linspace(-4, 4, 100)

    sigma1, sigma2 = 1.0, 0.7
    def _plot_double_gaussians(ax, mu1, mu2):
        norm1 = scstats.norm(loc = mu1, scale = sigma1)
        norm2 = scstats.norm(loc = mu2, scale = sigma2)
        x1, x2 = x+mu1, x+mu2
        y1, y2 = norm1.pdf(x1), norm2.pdf(x2)
        ax.plot(x1, y1)
        ax.fill_between(x1, y1, alpha = 0.5, label = 'Distribution 1')
        ax.plot(x2, y2)
        ax.fill_between(x2, y2, alpha = 0.5, label = 'Distribution 2')
        ax.legend(fontsize = "small", loc = "upper right")
        ax.set_ylim((0, 1.0))
        ax.grid(alpha = 0.6)
        return ax
    
    A_text_loc = (0.2, 0.8)
    axes[0] = _plot_double_gaussians(axes[0], 0.0, 1.0)
    axes[0].text(*A_text_loc, "$A < 0.5$", horizontalalignment = "center", fontweight = "bold", transform = axes[0].transAxes)
    axes[0].set_title("1 is stochastically less than 2", fontsize = "small")

    axes[1] = _plot_double_gaussians(axes[1], 0.0, 0.0)
    axes[1].text(*A_text_loc, "$A = 0.5$", horizontalalignment = "center", fontweight = "bold", transform = axes[1].transAxes)
    axes[1].set_title("1 is stochastically equal to 2", fontsize = "small")

    axes[2] = _plot_double_gaussians(axes[2], 0.0, -1.0)
    axes[2].text(*A_text_loc, "$A > 0.5$", horizontalalignment = "center", fontweight = "bold", transform = axes[2].transAxes)
    axes[2].set_title("1 is stochastically greater than 2", fontsize = "small")
    
    axes[2].text(-1, 0.01, u"\u2193", color = "mediumblue", fontweight = "bold", fontsize = "large")
    axes[2].text(0.7, 0.01, u"\u2193", color = "saddlebrown", fontweight = "bold", fontsize = "large")
    return fig

fig = Supplementary_Figure3()
fig.tight_layout()
#plt.savefig("./figures/supplementary_figure3.png")
# %%
