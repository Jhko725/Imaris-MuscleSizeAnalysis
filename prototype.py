#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
import numpy as np


plt.rcParams["font.size"] = 14
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.labelsize"] = "medium"

def import_data(filepath):
    df_raw = pd.read_excel(filepath, skiprows = None)
    labels = df_raw.columns[[0, 3]]
    df = df_raw.iloc[1:, [1, 4]]
    df.columns = labels
    return df

def import_data_list(filepath_list):
    return list(map(import_data, filepath_list))

# %%
fed_fast_filepaths = ["./fed-fast manual.xlsx", "./fed-fast automated.xlsx", "./fed-fast corrected.xlsx"]
fed_fast_dataset = import_data_list(fed_fast_filepaths)

shLacz_filepaths = ["./shLacz manual.xlsx", "./shLacz automated.xlsx", "./shLacz corrected Aviv.xlsx", "./shLacz corrected Nadav.xlsx"]
shLacz_dataset = import_data_list(shLacz_filepaths)

shCAPN1_filepaths = ["./shCAPN1 manual Aviv.xlsx", "./shCAPN1 manual Nadav.xlsx", "./shCAPN1 automated.xlsx", "./shCAPN1 corrected.xlsx"]
shCAPN1_dataset = import_data_list(shCAPN1_filepaths)
# %%
fed_fast_dataset[2].shape
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
from statsmodels.graphics import gofplots
import statsmodels.tools as smtools
import scipy.stats as scstats

def calculate_QQplot(data1, data2, a = 0):
    def _sample_quantiles(data):
        probplot = gofplots.ProbPlot(np.array(data, dtype = float), a = a)
        return probplot.sample_quantiles
    
    def _match_quantile_probabilities(quantiles1, quantiles2):
        if len(quantiles1) > len(quantiles2):
            quantiles2, quantiles1 = _match_quantile_probabilities(quantiles2, quantiles1)
        else:
            N_obs = len(quantiles1)
            probs = gofplots.plotting_pos(N_obs, a)
            quantiles2 = scstats.mstats.mquantiles(quantiles2, probs)
        
        return quantiles1, quantiles2

    s1, s2 = _sample_quantiles(data1), _sample_quantiles(data2)
    s1, s2 = _match_quantile_probabilities(s1, s2)
    
    linreg_result = sm.OLS(s1, smtools.tools.add_constant(s2)).fit()
    s2_fitted = linreg_result.fittedvalues
    r = np.sqrt(linreg_result.rsquared)
    
    return s1, s2, s2_fitted, r
# %%
def plot_QQplot(dataframe1, dataframe2, labels):
    experiment_names = list(map(str.capitalize, dataframe1.columns))
    fig, axes = plt.subplots(1, 2, figsize = (16, 6))

    for i, ax in enumerate(axes):
        s1, s2, s2_fitted, r = calculate_QQplot(dataframe1.iloc[:, i], dataframe2.iloc[:, i])
        
        ax.scatter(s2, s1, label = "Q-Q plot")
        ax.plot(s2, s2_fitted, 'r-', label = f"linear fit: $r = {r :.3f}$")
        ax.plot(s2, s2, 'k--', label = "$y = x$")

        x_scale = 100
        x_ticks = ticker.FuncFormatter(lambda x, pos: f"{x/x_scale :0g}")
        ax.xaxis.set_major_formatter(x_ticks)
        ax.yaxis.set_major_formatter(x_ticks)
        #ax.set_xlim((150, 7500))
        #ax.set_ylim((0, 7500))
        
        ax.legend()
        ax.set_xlabel(f"{labels[0]} quantile ($\mu m^2 \\times 100$)")
        ax.set_ylabel(f"{labels[1]} quantile ($\mu m^2 \\times 100$)")
        ax.set_title(f"{experiment_names[i]}")

        fig.suptitle(f"Q-Q plot between the {labels[0]} and {labels[1]} distributions")
        plt.tight_layout()

    return fig, axes


# %%
fig, axes = plot_QQplot(fed_fast_dataset[0], fed_fast_dataset[1], ["Manual", "Automated"])
#plt.savefig("./figures/fed_fast - automated QQplot.png")

# %%
fig, axes = plot_QQplot(fed_fast_dataset[0], fed_fast_dataset[2], ["Manual", "Corrected"])
#plt.savefig("./figures/fed_fast - corrected QQplot.png")
# %%
sample1 = np.array(fed_fast_dataset[0].iloc[:,0], dtype = float)
sample2 = np.array(fed_fast_dataset[0].iloc[:,1], dtype = float)
ks, p = scstats.ks_2samp(sample1, sample2)
# %%
es, p = scstats.epps_singleton_2samp(sample1, sample2)
# %%
ks
# %%
p
# %%
scstats.mstats.spearmanr(sample1, sample2)
# %%

# %%

# %%
def calculate_effect_size(sample1, sample2):
    sample1 = np.array(sample1, dtype = float)
    sample2 = np.array(sample2, dtype = float)

    N_total = len(sample1)*len(sample2)
    s1gts2 = np.sum(sample1.reshape(-1, 1) > sample2.reshape(1, -1))/N_total
    s1eqs2 = np.sum(sample1.reshape(-1, 1) == sample2.reshape(1, -1))/N_total
    return s1gts2+0.5*s1eqs2
# %%
data_ind = 0
sample1, sample2 = shLacz_dataset[data_ind].iloc[:,0], shLacz_dataset[data_ind].iloc[:,1]
scstats.mstats.brunnermunzel(sample1, sample2, alternative = "two-sided", distribution = "normal")
# %%
calculate_effect_size(sample1, sample2)
# %%
sample1, sample2 = shCAPN1_dataset[0].iloc[:,0], shCAPN1_dataset[0].iloc[:,1]
scstats.mstats.brunnermunzel(sample1, sample2, alternative = "less", distribution = "normal")
# %%
calculate_effect_size(sample1, sample2)
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
np.mean(sample0)
# %%
np.median(sample0)
# %%
scstats.tstd(sample0)
# %%
scstats.skew(sample0)
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
from numpy.random import default_rng
def draw_bootstrap_samples(sample_array, N_bootstrap = 10000):
    rng = default_rng()
    bootstrap_samples = rng.choice(sample_array, size = (N_bootstrap, len(sample_array)))
    return bootstrap_samples

def calculate_bootstrap_CI(statistics_function, sample_array, confidence_level = 0.95, N_bootstrap = 10000):
    sample_stat = statistics_function(sample_array)
    bootstrap_samples = draw_bootstrap_samples(sample_array, N_bootstrap = N_bootstrap)
    bootstrap_stats = statistics_function(bootstrap_samples, axis = 1)
    bootstrap_differences = bootstrap_stats - sample_stat

    alpha = 1-confidence_level
    difference_CI = np.quantile(bootstrap_differences, (0.5*alpha, 1-0.5*alpha))
    print(difference_CI)
    CI = np.flip(sample_stat - difference_CI)
    return sample_stat, CI

# %%
sample = fed_fast_dataset[1].iloc[:,0]
sample = np.array(sample, dtype = np.float32)
# %%
calculate_bootstrap_CI(np.mean, sample, N_bootstrap = 100000)
# %%
calculate_bootstrap_CI(np.median, sample, N_bootstrap = 100000)
# %%
calculate_bootstrap_CI(scstats.tstd, sample, N_bootstrap = 100000)
# %%
calculate_bootstrap_CI(scstats.skew, sample, N_bootstrap = 20000)
# %%
true_mean = np.mean(sample0)
mean_dist = np.mean(boot0, axis = 1)
# %%
print(true_mean)
print(np.mean(mean_dist))
# %%
print(np.std(sample0)/np.sqrt(len(sample0)))
print(scstats.tstd(mean_dist))
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
kde0 = sm.nonparametric.KDEUnivariate(sample0)
kde0.fit(bw = "silverman")
kde1 = sm.nonparametric.KDEUnivariate(sample1)
kde1.fit(bw = "silverman")
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.hist(sample1, bins = 50, density = True)
ax.plot(kde1.support, kde1.density, lw = 3)
# %%
len(kde0.support)
# %%
from matplotlib.patches import Rectangle
def plot_experiment_comparision_histogram(ax, dataframe):
    x_scale = 100
    x_ticks = ticker.FuncFormatter(lambda x, pos: f"{x/x_scale :0g}")
    hist_kwargs = {"edgecolor": "black"}

    #ax.plot([], [], ' ', label = f"""{"" :8s} Skewness""")
    #label = _make_label(dataframe)
    ax.hist(dataframe.values, label = dataframe.columns, bins = 18, density = True, rwidth = 0.7, **hist_kwargs)
    
    ax.xaxis.set_major_formatter(x_ticks)
    ax.ticklabel_format(axis = "y", style = "sci", scilimits = (0, 0))
    #ax.set_xlim((0, 9000))
    ax.legend()

    ax.set_xlabel("Fiber size ($\mu m^2 \\times 100$)")
    ax.set_ylabel("Fiber number probability")

    return ax

def plot_descr_stats_comparison_table(ax, dataframe, statistics_names):
   
    output = pd.concat([calculate_descriptive_stats(dataframe, i) for i in range(dataframe.shape[1])], axis = 1) 
    data = output.loc[statistics_names].values
        
    table = ax.table(cellText = np.around(data, 2), colLabels = dataframe.columns, rowLabels = statistics_names, loc = "center", cellLoc = "center")
    table.scale(1, 2.5)
    table.set_fontsize(16)

    ax.axis("off")
    #ax.axis("tight")
    
    return ax

def plot_experiment_comparision(dataframe, statistics_names, figsize = (10, 5)):
    fig, axes = plt.subplots(1, 2, figsize = figsize, gridspec_kw = {"width_ratios": [1.5, 1]})
    
    axes[0] = plot_experiment_comparision_histogram(axes[0], dataframe)
    axes[1] = plot_descr_stats_comparison_table(axes[1], dataframe, statistics_names)
    return fig, axes

def _make_label(dataframe):
    labels = []
    for i, exp_name in enumerate(dataframe.columns):
        skew = calculate_descriptive_stats(dataframe, i).loc["skew"]
        label_string = f"{str.capitalize(exp_name) :9s}  {skew[0] :.2f}"
        labels.append(label_string)
    return labels
# %%
fig, ax = plot_experiment_comparision_histogram(fed_fast_dataset[0])
ax.set_title("Distributions for the manual analysis")
plt.tight_layout()
# %%
calculate_descriptive_stats(fed_fast_dataset[0], 0).loc["skew"].values
# %%
fig, ax = plot_descr_stats_comparison_table(fed_fast_dataset[0], stat_names)
plt.tight_layout()
# %%
fig, axes = plot_experiment_comparision(fed_fast_dataset[0], stat_names)
fig.suptitle("Comparison of distributions for the manual analysis of the fed-fast experiment")
plt.tight_layout()
#plt.savefig("./figures/fed-fast_manual.png")
# %%
fig, axes = plot_experiment_comparision(fed_fast_dataset[1], stat_names)
fig.suptitle("Comparison of distributions for the automated analysis of the fed-fast experiment")
plt.tight_layout()
#plt.savefig("./figures/fed-fast_automated.png")
# %%
fig, axes = plot_experiment_comparision(fed_fast_dataset[2], stat_names)
fig.suptitle("Comparison of distributions for the corrected analysis of the fed-fast experiment")
plt.tight_layout()
plt.savefig("./figures/fed-fast_corrected.png")

# %%
def plot_comparison_QQplot(dataframe):
   
    fig, ax = plt.subplots(1, 1, figsize = (6, 5))

    s1, s2, s2_fitted, r = calculate_QQplot(dataframe.iloc[:, 0], dataframe.iloc[:, 1])
        
    ax.scatter(s2, s1, label = "Q-Q plot")
    ax.plot(s2, s2_fitted, 'r-', label = f"linear fit: $r = {r :.3f}$")
    ax.plot(s2, s2, 'k--', label = "$y = x$")

    x_scale = 100
    x_ticks = ticker.FuncFormatter(lambda x, pos: f"{x/x_scale :0g}")
    ax.xaxis.set_major_formatter(x_ticks)
    ax.yaxis.set_major_formatter(x_ticks)
    #ax.set_xlim((150, 7500))
    #ax.set_ylim((0, 7500))

    labels = list(map(str.lower, dataframe.columns))
    ax.legend()
    ax.set_xlabel(f"Quantiles: {labels[0]} ($\mu m^2 \\times 100$)")
    ax.set_ylabel(f"Quantiles: {labels[1]} ($\mu m^2 \\times 100$)")

    return fig, ax

# %%
fig, ax = plot_comparison_QQplot(fed_fast_dataset[0])
ax.set_title("Q-Q plot for the manual analysis")
plt.tight_layout()
#plt.savefig("./figures/fed_fast - QQplot manual.png")
# %%
fig, ax = plot_comparison_QQplot(fed_fast_dataset[1])
ax.set_title("Q-Q plot for the automated analysis")
plt.tight_layout()
#plt.savefig("./figures/fed_fast - QQplot automated.png")
# %%
fig, ax = plot_comparison_QQplot(fed_fast_dataset[2])
ax.set_title("Q-Q plot for the corrected analysis")
plt.tight_layout()
#plt.savefig("./figures/fed_fast - QQplot corrected.png")
# %%

# %%
fig, axes = plot_experiment_comparision(shLacz_dataset[0], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the manual analysis of the shLacz experiment")
fig.tight_layout()
#plt.savefig("./figures/shLacz_manual.png")
# %%
fig, axes = plot_experiment_comparision(shLacz_dataset[1], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the automated analysis of the shLacz experiment")
fig.tight_layout()
#plt.savefig("./figures/shLacz_automated.png")
# %%
fig, axes = plot_experiment_comparision(shLacz_dataset[2], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the corrected(Aviv) analysis of the shLacz experiment")
fig.tight_layout()
#plt.savefig("./figures/shLacz_correctedAviv.png")
# %%
# %%
fig, axes = plot_experiment_comparision(shLacz_dataset[3], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the corrected(Nadav) analysis of the shLacz experiment")
fig.tight_layout()
plt.savefig("./figures/shLacz_correctedNadav.png")
# %%
fig, ax = plot_comparison_QQplot(shLacz_dataset[0])
ax.set_title("Q-Q plot for the manual analysis")
plt.tight_layout()
#plt.savefig("./figures/shLacz - QQplot manual.png")
# %%
fig, ax = plot_comparison_QQplot(shLacz_dataset[1])
ax.set_title("Q-Q plot for the automated analysis")
plt.tight_layout()
plt.savefig("./figures/shLacz - QQplot automated.png")
# %%
fig, ax = plot_comparison_QQplot(shLacz_dataset[2])
ax.set_title("Q-Q plot for the corrected(Aviv) analysis")
plt.tight_layout()
plt.savefig("./figures/shLacz - QQplot correctedAviv.png")
# %%
fig, ax = plot_comparison_QQplot(shLacz_dataset[3])
ax.set_title("Q-Q plot for the corrected(Nadav) analysis")
plt.tight_layout()
plt.savefig("./figures/shLacz - QQplot correctedNadav.png")
# %%

# %%
fig, axes = plot_experiment_comparision(shCAPN1_dataset[0], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the manual(Aviv) analysis of the shCAPN1 experiment")
fig.tight_layout()
plt.savefig("./figures/shCAPN1_manualAviv.png")
# %%
fig, axes = plot_experiment_comparision(shCAPN1_dataset[1], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the manual(Nadav) analysis of the shCAPN1 experiment")
fig.tight_layout()
plt.savefig("./figures/shCAPN1_manualNadav.png")
# %%
fig, axes = plot_experiment_comparision(shCAPN1_dataset[2], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the automated analysis of the shCAPN1 experiment")
fig.tight_layout()
plt.savefig("./figures/shCAPN1_automated.png")
# %%
fig, axes = plot_experiment_comparision(shCAPN1_dataset[3], stat_names, figsize = (12, 5))
fig.suptitle("Comparison of distributions for the corrected analysis of the shCAPN1 experiment")
fig.tight_layout()
plt.savefig("./figures/shCAPN1_corrected.png")

# %%
fig, ax = plot_comparison_QQplot(shCAPN1_dataset[0])
ax.set_title("Q-Q plot for the manual(Aviv) analysis")
plt.tight_layout()
plt.savefig("./figures/shCAPN1 - QQplot manualAviv.png")
# %%
fig, ax = plot_comparison_QQplot(shCAPN1_dataset[1])
ax.set_title("Q-Q plot for the manual(Nadav) analysis")
plt.tight_layout()
plt.savefig("./figures/shCAPN1 - QQplot manualNadav.png")
# %%
fig, ax = plot_comparison_QQplot(shCAPN1_dataset[2])
ax.set_title("Q-Q plot for the automated analysis")
plt.tight_layout()
plt.savefig("./figures/shCAPN1 - QQplot automated.png")
# %%
fig, ax = plot_comparison_QQplot(shCAPN1_dataset[3])
ax.set_title("Q-Q plot for the corrected analysis")
plt.tight_layout()
plt.savefig("./figures/shCAPN1 - QQplot corrected.png")
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
