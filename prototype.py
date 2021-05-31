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
    bootstrap_result = bootstrap_confidence_interval_2samp(*data_arrays, A_statistic, B = 20000, CI_algorithm = "BCa", random_seed = random_seed)
    return Statistic(*bootstrap_result, "A statistic")
# %%
dataset = shCAPN1_dataset
A_stats = [bootstrap_A_from_df(df) for df in dataset]
exp_labels = ["Manual", "Automated", "Corrected"]
A_val = np.array([A.value for A in A_stats])
A_err = np.array([A.errorbar() for A in A_stats]).T

# %%
def plot_statistics_bar(values, errors, experiment_labels, statistic_name, p_values, star_offset = 0.4):
    x = np.arange(len(A_stats))

    fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    ax.bar(x, values, yerr = errors, color = "mediumslateblue", edgecolor = "black", capsize = 4.0)
    #ax.axhline(y = 0.5, linestyle = "--", color = "firebrick", linewidth = 2.0, alpha = 0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_labels, fontsize = "small")
    ax.set_ylabel(statistic_name)
    for i, p in enumerate(p_values):
        if p < 0.001:
            ax.text(i, values[i]+star_offset, "***", horizontalalignment = "center")
    ax.grid(alpha = 0.3)
    ax.set_ylim((0, 0.25))
    fig.tight_layout()
    return fig
# %%
p_vals = [0, 0.86, 0]
fig = plot_statistics_bar(A_val, A_err, exp_labels, "A statistic", p_vals)
#plt.savefig("./figures/figure4C.png")
# %%
p_vals = [0.94, 0, 0.33]
fig = plot_statistics_bar(A_val, A_err, exp_labels, "A statistic", p_vals)
#plt.savefig("./figures/figure5C.png")
# %%
import scipy.stats as scstats
dataset = shLacz_dataset
KS_test_results = np.array([scstats.ks_2samp(*df_columns_to_arrays(df)) for df in dataset])
KS_vals = KS_test_results[:,0]
p_vals = KS_test_results[:,1]

fig = plot_statistics_bar(KS_vals, None, exp_labels, "KS statistic", p_vals, 0.1)
plt.savefig("./figures/figure5D.png")
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
data_ind = 3
sample1, sample2 = shLacz_dataset[data_ind].iloc[:,0], shLacz_dataset[data_ind].iloc[:,1]
sample1, sample2 = np.array(sample1, np.float32), np.array(sample2, np.float32)
scstats.mstats.brunnermunzel(sample1, sample2, alternative = "two-sided", distribution = "normal")
A_statistic(sample1, sample2)
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
data_ind = 3
sample1, sample2 = shCAPN1_dataset[data_ind].iloc[:,0], shCAPN1_dataset[data_ind].iloc[:,1]
sample1, sample2 = np.array(sample1, np.float32), np.array(sample2, np.float32)
scstats.mstats.brunnermunzel(sample1, sample2, alternative = "less", distribution = "normal")
# %%
A_statistic(sample1, sample2)
# %%
output = bootstrap_confidence_interval_2samp(sample1, sample2, A_statistic, B = 20000, CI_algorithm = "BCa")
#%%
stat = Statistic(*output, "A_statistic")
print(stat)

# %%
sample1, sample2 = shLacz_dataset[0].iloc[:,0], shLacz_dataset[0].iloc[:,1]
sample1, sample2 = np.array(sample1, dtype = float), np.array(sample2, dtype = float)
scstats.epps_singleton_2samp(sample1, sample2)

# %%
data_ind = 1
sample1, sample2 = shLacz_dataset[data_ind].iloc[:,0], shLacz_dataset[data_ind].iloc[:,1]
scstats.ks_2samp(sample1, sample2)
# %%

import statsmodels.stats as ststats

def calculate_descriptive_stats(dataframe, column_index):
    data_array = np.array(dataframe.iloc[:, column_index], dtype = float)
    des_stats = ststats.descriptivestats.describe(data_array)
    return des_stats

def plot_descr_stats_table(dataframe_list, label_list, statistics_names):
    fig, axes = plt.subplots(1, 2, figsize = (10, 3.5))
    exp_names = dataframe_list[0].columns

    for i, ax in enumerate(axes):
        output = pd.concat([calculate_descriptive_stats(df, i) for df in dataframe_list], axis = 1) 
        data = output.loc[statistics_names].values
        table = ax.table(cellText = np.around(data, 2), colLabels = label_list, rowLabels = statistics_names, loc = "center", cellLoc = "center")
        table.scale(1, 2)

        ax.axis("off")
        ax.axis("tight")
        ax.set_title(exp_names[i])
    
    return fig, axes
# %%
ststats.descriptivestats.describe(sample0)
# %%

# %%
labels = ["Manual", "Automated", "Corrected"]
stat_names = ["median", "mean", "std", "skew"]
fig, axes = plot_descr_stats_table(fed_fast_dataset, labels, stat_names)
fig.suptitle("Descriptive statistics for mouse muscle fiber size distribution")
plt.tight_layout()
#plt.savefig("./figures/fed_fast - descrptive statistics.png")
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
