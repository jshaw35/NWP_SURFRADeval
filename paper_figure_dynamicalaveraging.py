"""
Plot the dynamical averaging results across locations and cloud regimes.
"""
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# %%

if __name__ == "__main__":

    data_path = "data/figure_outputs/"
    save_path = "figures/paper_figures"

    # %%
    # Find results with different parameters:
    Afactor_files = glob.glob(data_path + "error_all_cloud_conditions_HRRRweighted*.csv")
    # Parse the files to get the Afactor values and combine into a single dataframe:
    # Add the unweighted mean as a control:
    df = pd.read_csv(Afactor_files[0])
    control = df.loc[df["nwp_source"] == "all_ensmean"]
    control["Afactor"] = 0.0
    Afactor_dfs = [control]
    for file in Afactor_files:
        # Extract the Afactor value from the filename:
        Afactor = float(file.split("HRRRweighted_Afactor_")[1].split(".csv")[0])
        df = pd.read_csv(file)
        df = df.loc[df["nwp_source"] == "weighted_mean"]
        df["Afactor"] = Afactor
        Afactor_dfs.append(df)
    Afactor_df = pd.concat(Afactor_dfs, ignore_index=True)

    # %%
    error_variables = ["mbe", "stddev", "rmse"]
    ylims_dict = {
        "mbe": [-45, 5],
        "stddev": [-12, 8],
        "rmse": [-30, 10],
    }
    ylabels_dict = {
        "mbe": "Mean Bias Error",
        "stddev": "Error Standard Deviation",
        "rmse": "RMSE",
    }

    cloud_regimes = ["cloudy", "broken", "clear", "all"]
    cloud_regimes_labels = ["Overcast", "Partly Cloudy", "Clear", "All"]
    control_nwp_source = "all_ensmean"
    test_nwp_source = "weighted_mean"
    drop_Afactors = [0.0, 3.0]  # Drop the control Afactor for plotting
    panel_letters = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.']

    locations = Afactor_df["location"].unique()
    Afactors = Afactor_df["Afactor"].unique()
    colors = sns.color_palette("colorblind", len(Afactors))

    # Produce a bar plot comparing the control and test results for each location and cloud regime:
    for plot_variable in error_variables:

        ylims = ylims_dict[plot_variable]

        fig, axes = plt.subplots(len(locations), 1, figsize=(10, 2.5 * len(locations)), sharex=True)
        fig.subplots_adjust(hspace=0.1)

        for ax, location, letter in zip(axes, locations, panel_letters):
            location_data = Afactor_df.loc[(Afactor_df["location"] == location) & (Afactor_df["variable"] == plot_variable)]

            x0 = 0
            xticks = []
            for i, cloud_regime in enumerate(cloud_regimes):
                regime_data = location_data[[cloud_regime, "Afactor"]].sort_values(by="Afactor")
                regime_data_norm = regime_data.loc[regime_data["Afactor"] == 0.0][cloud_regime]  # Remove the control data for plotting
                # drop rows with Afactor in drop_Afactors
                regime_data = regime_data.loc[~regime_data["Afactor"].isin(drop_Afactors)]
                x = x0 + np.arange(len(regime_data))
                out = ax.bar(
                    x=x,
                    height=regime_data[cloud_regime] - regime_data_norm.values,
                    label=f"{cloud_regime} (Afactor)",
                    color=colors,
                )
                x0 = x[-1] + 2  # Update x0 for the next cloud regime
                xticks += [x.mean()]  # Use the mean of the x values for the tick position
            ax.axhline(
                y=0,
                color="black",
                linewidth=1,
            )
            # Label the panel with the letter and location:
            ax.annotate(f"{letter} {location.upper()}", xy=(0.01, 0.9), xycoords="axes fraction", fontsize=12)
            ax.set_ylim(ylims)

        axes[0].xaxis.set_ticks_position("top")
        axes[0].set_xticks(xticks)
        axes[0].set_xticklabels(cloud_regimes_labels, fontsize=12)
        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels(cloud_regimes_labels, fontsize=12)

        axes[0].legend(
            handles=out,
            labels=[f"A={afactor}" for afactor in regime_data["Afactor"]],
            loc=[0.4, 0.02],
            fontsize=10,
            ncol=2,
        )
        # Add a y-axis label for all panels:
        fig.text(0.06, 0.5, f"Change in GHI {ylabels_dict[plot_variable]} (Wm$^{-2}$)", va="center", rotation="vertical", fontsize=16)
        fig.savefig(f"{save_path}/dynamicalaveraging_{plot_variable}_comparison", dpi=300, bbox_inches="tight")

# %%
