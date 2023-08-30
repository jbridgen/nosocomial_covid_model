import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import tensorflow_probability as tfp
import os
import scipy.stats as stats

import general_funcs as gf

tfd = tfp.distributions


def plot_post_pred_infections(
    fixed_ei_event_list, simulated_infection_events, fig_version, chain
):
    fixed_ei_event_list["time"] = np.floor(fixed_ei_event_list["time"]).astype(
        int
    )
    hosp_data_time_series = pd.DataFrame(
        {"day": fixed_ei_event_list["time"], "freq": 1}
    )

    hosp_data_time_series = (
        hosp_data_time_series.groupby(by="day").sum().reset_index()
    )
    days = pd.DataFrame(
        {"day": range(0, np.max(fixed_ei_event_list["time"]).item())}
    )
    hosp_data_time_series_changed = hosp_data_time_series.merge(
        days, on="day", how="right"
    ).fillna(0)

    # Plot mean simulation values with confidence interval and observed events with simulation traces
    sim_infections_df = pd.concat(simulated_infection_events)
    mean_val = (
        sim_infections_df.groupby("day")
        .mean("freq")
        .rename(columns={"freq": "mean_val"})
    )
    lower_quant = (
        sim_infections_df.groupby("day")
        .quantile("0.025")
        .rename(columns={"freq": "lower_quant"})
    )
    upper_quant = (
        sim_infections_df.groupby("day")
        .quantile("0.975")
        .rename(columns={"freq": "upper_quant"})
    )
    sim_agg = pd.concat([mean_val, lower_quant, upper_quant], axis=1)
    rng = np.random.default_rng(seed=168)

    fig = plt.figure(figsize=(15, 4))
    for i in rng.integers(
        low=0, high=len(simulated_infection_events) - 1, size=5
    ):
        sns.lineplot(
            data=simulated_infection_events[i],
            x="day",
            y="freq",
            color="springgreen",
            alpha=0.25,
        )
    mean = sns.lineplot(
        x=sim_agg.index,
        y=sim_agg.mean_val,
        label="simulation mean",
        color="green",
    )
    lower_quant = sns.lineplot(
        x=sim_agg.index, y=sim_agg.lower_quant, color="lightgreen", alpha=0.3
    )
    upper_quant = sns.lineplot(
        x=sim_agg.index, y=sim_agg.upper_quant, color="lightgreen", alpha=0.3
    )
    sns.lineplot(
        data=hosp_data_time_series,
        x="day",
        y="freq",
        label="observed",
        color="black",
        linestyle="dashed",
    )

    upper_quant.set_xlabel("Day")
    upper_quant.set_ylabel("Number of EI transitions")
    upper_quant.set_title("posterior predictive")
    mean.set_xlim(0, np.max(fixed_ei_event_list["time"]))
    line = upper_quant.get_lines()

    plt.fill_between(
        line[5].get_xdata(),
        line[6].get_ydata(),
        line[7].get_ydata(),
        color="lightgreen",
        alpha=0.3,
    )
    plt.savefig("posterior_predictive_" + chain + "_" + fig_version)
    plt.close()


def save_trace_plots(post, fig_version, chain):
    # Parameter grid plot
    fig = plt.figure(figsize=(10, 7))

    plt.subplot(5, 1, 1)
    plt.plot(post.beta1)
    plt.title("beta 1")

    plt.subplot(5, 1, 2)
    plt.plot(post.beta2)
    plt.title("beta 2")

    plt.subplot(5, 1, 3)
    plt.plot(post.beta3)
    plt.title("beta 3")

    plt.subplot(5, 1, 4)
    plt.plot(post.beta4)
    plt.title("beta 4")

    plt.subplot(5, 1, 5)
    plt.plot(post.beta5)
    plt.title("beta 5")

    plt.tight_layout()
    plt.savefig("trace_plots_" + chain + fig_version)
    plt.close()


def plot_att_fracts(
    att_fracts_all,
    exp_by_ward,
    ward_cols_nosoc,
    ward_foi_all,
    ward_foi_within_ward_all,
    ward_foi_between_ward_all,
    ward_foi_background_all,
    hospital_event_times,
    path,
    fig_version,
    chain="chain_1",
):
    """Create plots and tables for exposure attributable fractions"""
    att_fract_merged = pd.concat(att_fracts_all)
    mean_att_fract = att_fract_merged.groupby("event_unit").mean()
    lower_ci_att_fract = att_fract_merged.groupby("event_unit").quantile(
        "0.025"
    )
    upper_ci_att_fract = att_fract_merged.groupby("event_unit").quantile(
        "0.975"
    )

    nosoc_att_fract_indices = np.where(mean_att_fract["pr_community"] < 0.5)
    mean_nosoc_att_fract = mean_att_fract.iloc[nosoc_att_fract_indices]

    # Figure without pid for publication
    mean_nosoc_att_fract_anon = mean_nosoc_att_fract[
        ["pr_within_ward", "pr_between_ward", "pr_background", "pr_community"]
    ]
    mean_nosoc_att_fract_anon.index = np.arange(
        1, len(mean_nosoc_att_fract_anon) + 1
    )
    att_fracts_fig = mean_nosoc_att_fract_anon.plot.barh(
        stacked=True,
        width=0.9,
        color=["#0E8E52", "#165895", "#D9781D", "#ABA8B2"],
        figsize=(10, 6),
    )
    plt.ylabel("SARS-CoV-2 exposure event")
    plt.xlabel("Attributable fraction")
    plt.legend(
        labels=[
            "within ward transmission",
            "between ward transmission",
            "background hospital transmission",
            "community transmission",
        ],
        loc=(0.25, -0.2),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.margins(x=0)
    plt.savefig("attributable_fractions_" + chain + "_" + fig_version)
    plt.close()

    # Nosocomial exposures by ward
    exp_by_ward_merged = pd.concat(exp_by_ward)
    mean_exp_by_ward = exp_by_ward_merged.groupby("wards").mean()
    mean_exp_by_ward["perc"] = (
        mean_exp_by_ward["exposures"] / np.sum(mean_exp_by_ward["exposures"])
    ) * 100
    lower_ci_exp_by_ward = (
        exp_by_ward_merged.groupby("wards")
        .quantile("0.025")
        .rename(columns={"exposures": "lower_ci_exp", "perc": "lower_ci_perc"})
    )
    upper_ci_exp_by_ward = (
        exp_by_ward_merged.groupby("wards")
        .quantile("0.975")
        .rename(columns={"exposures": "upper_ci_exp", "perc": "upper_ci_perc"})
    )
    mean_exp_by_ward = pd.concat(
        [mean_exp_by_ward, lower_ci_exp_by_ward, upper_ci_exp_by_ward], axis=1
    )
    nosoc_exp_by_ward = mean_exp_by_ward[mean_exp_by_ward["exposures"] > 0]
    nosoc_exp_by_ward.sort_values(by=["perc"], ascending=False, inplace=True)

    nosoc_exp_by_ward["perc"] = round(nosoc_exp_by_ward["perc"], 2)
    nosoc_exp_by_ward["Percentage of hospital exposures (95% CI)"] = (
        nosoc_exp_by_ward["perc"].astype(str)
        + " ("
        + round(nosoc_exp_by_ward["lower_ci_perc"], 2).astype(str)
        + "-"
        + round(nosoc_exp_by_ward["upper_ci_perc"], 2).astype(str)
        + ")"
    )

    perc_nosoc_exp_by_ward = nosoc_exp_by_ward[
        ["exposures", "Percentage of hospital exposures (95% CI)"]
    ]

    # Mean infectious pressure by ward
    all_wards = gf.load_pickled_file(path, "wards_saved")
    cov_pointers_times = dict(
        zip(range(len(hospital_event_times)), np.array(hospital_event_times))
    )
    foi_by_ward = pd.concat(ward_foi_all)
    mean_foi = foi_by_ward.groupby("cov_points").mean()
    mean_foi.index = mean_foi.index.to_series().map(cov_pointers_times)

    ward_foi_within_ward = pd.concat(ward_foi_within_ward_all)
    mean_within_ward_foi = ward_foi_within_ward.groupby("cov_points").mean()
    mean_within_ward_foi["day"] = mean_within_ward_foi.index.to_series().map(
        cov_pointers_times
    )

    ward_foi_between_ward = pd.concat(ward_foi_between_ward_all)
    mean_between_ward_foi = ward_foi_between_ward.groupby("cov_points").mean()
    mean_between_ward_foi["day"] = mean_between_ward_foi.index.to_series().map(
        cov_pointers_times
    )

    ward_foi_background_ward = pd.concat(ward_foi_background_all)
    mean_foi_background_ward = ward_foi_background_ward.groupby(
        "cov_points"
    ).mean()
    mean_foi_background_ward[
        "day"
    ] = mean_foi_background_ward.index.to_series().map(cov_pointers_times)
    mean_foi_background_ward = mean_foi_background_ward.rename(
        columns={0: "foi"}
    )

    # Identify the four wards with the most nosocomial infections
    ward_ind = perc_nosoc_exp_by_ward[0:4].index
    ward_a = np.where(all_wards == ward_ind[0])[0]
    ward_b = np.where(all_wards == ward_ind[1])[0]
    ward_c = np.where(all_wards == ward_ind[2])[0]
    ward_d = np.where(all_wards == ward_ind[3])[0]

    fig = plt.figure(figsize=(14, 7))
    plt.subplot(2, 2, 1)
    plt.title("Ward " + str(ward_a.item() + 1))
    plt.plot(mean_foi[ward_a], color="gray", alpha=0.2)
    plt.scatter(
        x=mean_within_ward_foi["day"],
        y=mean_within_ward_foi[ward_a],
        color="#0E8E52",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_between_ward_foi["day"],
        y=mean_between_ward_foi[ward_a],
        color="#165895",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_foi_background_ward["day"],
        y=mean_foi_background_ward["foi"],
        color="#D9781D",
        alpha=0.75,
        s=16,
    )

    plt.subplot(2, 2, 2)
    plt.title("Ward " + str(ward_b.item() + 1))
    plt.plot(mean_foi[ward_b], color="gray", alpha=0.2)
    plt.scatter(
        x=mean_within_ward_foi["day"],
        y=mean_within_ward_foi[ward_b],
        color="#0E8E52",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_between_ward_foi["day"],
        y=mean_between_ward_foi[ward_b],
        color="#165895",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_foi_background_ward["day"],
        y=mean_foi_background_ward["foi"],
        color="#D9781D",
        alpha=0.75,
        s=16,
    )

    plt.subplot(2, 2, 3)
    plt.title("Ward " + str(ward_c.item() + 1))
    plt.plot(mean_foi[ward_c], color="gray", alpha=0.2)

    plt.scatter(
        x=mean_within_ward_foi["day"],
        y=mean_within_ward_foi[ward_c],
        color="#0E8E52",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_between_ward_foi["day"],
        y=mean_between_ward_foi[ward_c],
        color="#165895",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_foi_background_ward["day"],
        y=mean_foi_background_ward["foi"],
        color="#D9781D",
        alpha=0.75,
        s=16,
    )

    plt.subplot(2, 2, 4)
    plt.title("Ward " + str(ward_d.item() + 1))
    plt.plot(mean_foi[ward_d], color="gray", alpha=0.2)

    plt.scatter(
        x=mean_within_ward_foi["day"],
        y=mean_within_ward_foi[ward_d],
        color="#0E8E52",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_between_ward_foi["day"],
        y=mean_between_ward_foi[ward_d],
        color="#165895",
        alpha=0.75,
        s=16,
    )
    plt.scatter(
        x=mean_foi_background_ward["day"],
        y=mean_foi_background_ward["foi"],
        color="#D9781D",
        alpha=0.75,
        s=16,
    )

    plt.tight_layout()
    fig.legend(
        [
            "mean infectious pressure",
            "within-ward",
            "between-ward",
            "background",
        ],
        loc="lower right",
        frameon=False,
        borderpad=0,
        ncol=2,
    )
    fig.supylabel("Infectious pressure")
    fig.supxlabel("Day")
    plt.tight_layout()
    plt.savefig("infectious_pressure_by_ward_" + chain + fig_version)
    plt.close()

    # Ward colours
    ward_cols_nosoc_merged = pd.concat(ward_cols_nosoc).reset_index()
    mean_ward_cols_nosoc = ward_cols_nosoc_merged.groupby("colour").mean()
    mean_ward_cols_nosoc["perc"] = (
        mean_ward_cols_nosoc["freq"] / np.sum(mean_ward_cols_nosoc["freq"])
    ) * 100
    lower_ci_ward_cols_nosoc = (
        ward_cols_nosoc_merged.groupby("colour")
        .quantile("0.025")
        .rename(columns={"freq": "lower_ci_exp", "perc": "lower_ci_perc"})
    )
    upper_ci_ward_cols_nosoc = (
        ward_cols_nosoc_merged.groupby("colour")
        .quantile("0.975")
        .rename(columns={"freq": "upper_ci_exp", "perc": "upper_ci_perc"})
    )
    mean_ward_cols_nosoc = pd.concat(
        [
            mean_ward_cols_nosoc,
            lower_ci_ward_cols_nosoc,
            upper_ci_ward_cols_nosoc,
        ],
        axis=1,
    )
    mean_ward_cols_nosoc.sort_values(
        by=["perc"], ascending=False, inplace=True
    )

    mean_ward_cols_nosoc["perc"] = round(mean_ward_cols_nosoc["perc"], 2)
    mean_ward_cols_nosoc["Percentage of hospital exposures (95% CI)"] = (
        mean_ward_cols_nosoc["perc"].astype(str)
        + " ("
        + round(mean_ward_cols_nosoc["lower_ci_perc"], 2).astype(str)
        + "-"
        + round(mean_ward_cols_nosoc["upper_ci_perc"], 2).astype(str)
        + ")"
    )
    perc_ward_cols_nosoc = mean_ward_cols_nosoc[
        ["freq", "Percentage of hospital exposures (95% CI)"]
    ]
    perc_ward_cols_nosoc.to_csv(
        os.path.join(path, chain + "perc_ward_cols_nosoc.csv"), index=True
    )

    return mean_nosoc_att_fract, perc_nosoc_exp_by_ward, perc_ward_cols_nosoc


def create_density_plot(post, fixed_event_list, fig_version):
    # Create kernel density plots
    post_df = pd.DataFrame(
        {
            "beta1": post.beta1,
            "beta2": post.beta2,
            "beta3": post.beta3,
            "beta4": post.beta4,
            "beta5": post.beta5,
        }
    )

    # Function to round to 1 significant figure
    def round_to_1sf(x):
        return round(x, -int(np.floor(np.log10(abs(x)))))

    x = np.linspace(0, 0.005, 10000)
    y = stats.gamma.pdf(x, 1.1, scale=1 / 1000)

    # Density plots with priors grid
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(5, 2, 1)
    sns.kdeplot(
        data=post_df["beta1"], ax=ax1, color="green", fill="green", cut=0
    )
    sns.lineplot(x=x, y=y, color="gray", linestyle="dashed", ax=ax1)
    ax1.xaxis.set_major_locator(
        ticker.MultipleLocator(round_to_1sf(np.max(post_df["beta1"]) / 3))
    )
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set(
        xlim=(
            -round_to_1sf(np.max(post_df["beta1"]) / 3) / 50,
            np.max(post_df["beta1"]) * 1.2,
        ),
        xlabel=r"$\beta_1$",
    )
    ax1.yaxis.label.set_visible(False)

    ax2 = fig.add_subplot(5, 2, 3)
    sns.kdeplot(
        post_df["beta2"],
        ax=ax2,
        bw_adjust=1.5,
        color="green",
        fill="green",
        cut=0,
    )
    sns.lineplot(
        x=x / 100, y=y * 100, color="gray", linestyle="dashed", ax=ax2
    )
    ax2.set(
        xlim=(
            -round_to_1sf(np.max(post_df["beta2"]) / 3) / 50,
            np.max(post_df["beta2"]) * 1.2,
        ),
        xlabel=r"$\beta_2$",
    )
    ax2.xaxis.set_major_locator(
        ticker.MultipleLocator(round_to_1sf(np.max(post_df["beta2"]) / 3))
    )
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.label.set_visible(False)

    ax3 = fig.add_subplot(5, 2, 5)
    sns.kdeplot(
        post_df["beta3"],
        ax=ax3,
        bw_adjust=1.5,
        color="green",
        fill="green",
        cut=0,
    )
    sns.lineplot(x=x / 10, y=y * 10, color="gray", linestyle="dashed", ax=ax3)
    ax3.set(
        xlim=(
            -round_to_1sf(np.max(post_df["beta3"]) / 3) / 50,
            np.max(post_df["beta3"]) * 1.2,
        ),
        xlabel=r"$\beta_3$",
    )
    ax3.xaxis.set_major_locator(
        ticker.MultipleLocator(round_to_1sf(np.max(post_df["beta3"]) / 3))
    )
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.yaxis.label.set_visible(False)

    ax4 = fig.add_subplot(5, 2, 7)
    sns.kdeplot(
        post_df["beta4"],
        ax=ax4,
        bw_adjust=1.5,
        color="green",
        fill="green",
        cut=0,
    )
    sns.lineplot(x=x, y=y, color="gray", linestyle="dashed", ax=ax4)
    ax4.set(
        xlim=(
            -round_to_1sf(np.max(post_df["beta4"]) / 3) / 50,
            np.max(post_df["beta4"]) * 1.2,
        ),
        xlabel=r"$\beta_4$",
    )
    ax4.xaxis.set_major_locator(
        ticker.MultipleLocator(round_to_1sf(np.max(post_df["beta4"]) / 3))
    )
    ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax4.yaxis.label.set_visible(False)

    ax5 = fig.add_subplot(5, 2, 9)
    sns.kdeplot(
        post_df["beta5"],
        ax=ax5,
        bw_adjust=1.5,
        color="green",
        fill="green",
        cut=0,
    )
    sns.lineplot(x=x, y=y, color="gray", linestyle="dashed", ax=ax5)

    ax5.set(
        xlim=(
            -round_to_1sf(np.max(post_df["beta5"]) / 3) / 50,
            np.max(post_df["beta5"]) * 1.2,
        ),
        xlabel=r"$\beta_5$",
    )
    ax5.xaxis.set_major_locator(
        ticker.MultipleLocator(round_to_1sf(np.max(post_df["beta5"]) / 3))
    )
    ax5.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax5.yaxis.label.set_visible(False)

    fig.supylabel("Density", fontsize=11)

    # Create violin plot
    post_infs = tf.reshape(tf.where(post.event[0] == 0), [-1])
    rng = np.random.default_rng(seed=124)
    post_inf_inds = []

    for index, i in enumerate(rng.choice(post_infs, size=8, replace=False)):
        post_ind = pd.DataFrame(
            {
                "time": post.time[:, i],
                "unit": post.unit[:, i],
                "event": post.event[:, i],
                "ind": index + 1,
            }
        )
        post_inf_inds.append(post_ind)

    post_inf_inds = pd.concat(post_inf_inds)
    post_inf_inds["ind"] = post_inf_inds["ind"].astype(str)
    units_violin = post_inf_inds.unit.unique()

    post_obs_inds = []
    post_rec_inds = []
    for index, i in enumerate(units_violin):
        ind_rec = tf.reshape(
            tf.where(((post.unit[0] == i)) & (post.event[0] == 2)), [-1]
        )
        post_rec = pd.DataFrame(
            {
                "time": post.time[:, int(ind_rec)],
                "unit": post.unit[:, int(ind_rec)],
                "event": post.event[:, int(ind_rec)],
                "ind": index + 1,
            }
        )
        post_rec_inds.append(post_rec)

        ind_obs = tf.reshape(tf.where(fixed_event_list.unit == i), [-1])
        ind_obs = pd.DataFrame(
            {
                "time": fixed_event_list.time[int(ind_obs)].values,
                "unit": fixed_event_list.unit[int(ind_obs)].values,
                "ind": index + 1,
            },
            index=[0],
        )
        post_obs_inds.append(ind_obs)

    post_rec_inds = pd.concat(post_rec_inds)
    post_rec_inds["ind"] = post_rec_inds["ind"].astype(str)

    post_obs_inds = pd.concat(post_obs_inds)
    post_obs_inds["ind"] = post_obs_inds["ind"].astype(str)
    post_obs_inds["event"] = 1
    violin_df = pd.concat([post_inf_inds, post_rec_inds])

    ax6 = fig.add_subplot(1, 2, 2)

    sns.violinplot(
        data=post_inf_inds, x="time", y="ind", color="lightgrey", ax=ax6, cut=0
    )
    sns.violinplot(
        data=post_rec_inds,
        x="time",
        y="ind",
        color="dodgerblue",
        ax=ax6,
        cut=0,
    )
    sns.pointplot(
        data=post_obs_inds,
        x="time",
        y="ind",
        join=False,
        color="green",
        capsize=0.2,
        ax=ax6,
    )

    plt.setp(ax6.collections, alpha=0.7)
    ax6.set_xlabel("Day", fontdict={"size": 11})
    ax6.set_ylabel("Sample", fontdict={"size": 11})

    ax1.annotate(
        "A",
        xy=(-0.15, 1.01),
        xycoords="axes fraction",
        color="black",
        fontsize=20,
    )
    ax6.annotate(
        "B",
        xy=(-0.08, 1.01),
        xycoords="axes fraction",
        color="black",
        fontsize=20,
    )

    plt.tight_layout()
    plt.savefig("density_plots_" + fig_version)
    plt.close()


def evaluate_chain_convergence(path):
    """Evaluates chain convergence for: 'chain_1', 'chain_2' and 'chain_3' by
    calculating the potential scale reduction statistic. Saves trace plot of chains.
    """
    post_1, fixed_event_list = gf.load_mcmc_output(path, chain="chain_1")
    post_2, _ = gf.load_mcmc_output(path, chain="chain_2")
    post_3, _ = gf.load_mcmc_output(path, chain="chain_3")

    beta1 = tf.stack(
        [
            tf.convert_to_tensor(post_1.beta1),
            tf.convert_to_tensor(post_2.beta1),
            tf.convert_to_tensor(post_3.beta1),
        ]
    )
    rhat_beta1 = tfp.mcmc.diagnostic.potential_scale_reduction(
        tf.transpose(beta1), independent_chain_ndims=1
    )

    beta2 = tf.stack(
        [
            tf.convert_to_tensor(post_1.beta2),
            tf.convert_to_tensor(post_2.beta2),
            tf.convert_to_tensor(post_3.beta2),
        ]
    )
    rhat_beta2 = tfp.mcmc.diagnostic.potential_scale_reduction(
        tf.transpose(beta2), independent_chain_ndims=1
    )

    beta3 = tf.stack(
        [
            tf.convert_to_tensor(post_1.beta3),
            tf.convert_to_tensor(post_2.beta3),
            tf.convert_to_tensor(post_3.beta3),
        ]
    )
    rhat_beta3 = tfp.mcmc.diagnostic.potential_scale_reduction(
        tf.transpose(beta3), independent_chain_ndims=1
    )

    beta4 = tf.stack(
        [
            tf.convert_to_tensor(post_1.beta4),
            tf.convert_to_tensor(post_2.beta4),
            tf.convert_to_tensor(post_3.beta4),
        ]
    )
    rhat_beta4 = tfp.mcmc.diagnostic.potential_scale_reduction(
        tf.transpose(beta4), independent_chain_ndims=1
    )

    beta5 = tf.stack(
        [
            tf.convert_to_tensor(post_1.beta5),
            tf.convert_to_tensor(post_2.beta5),
            tf.convert_to_tensor(post_3.beta5),
        ]
    )
    rhat_beta5 = tfp.mcmc.diagnostic.potential_scale_reduction(
        tf.transpose(beta5), independent_chain_ndims=1
    )

    reduction_stat = {
        "beta1": rhat_beta1,
        "beta2": rhat_beta2,
        "beta3": rhat_beta3,
        "beta4": rhat_beta4,
        "beta5": rhat_beta5,
    }
    for i in reduction_stat:
        print(
            "chain covergence - reduction statistic:\n"
            "{}\t{}".format(i, reduction_stat[i])
        )

    fig = plt.figure(figsize=(15, 9))

    plt.subplot(5, 1, 1)

    plt.plot(post_2.beta1, alpha=0.75, color="orange")
    plt.plot(post_3.beta1, color="gray", alpha=0.5)
    plt.plot(post_1.beta1, color="C0")
    plt.title(r"$\beta_1$")

    plt.subplot(5, 1, 2)
    plt.plot(post_2.beta2, color="orange", alpha=0.75)
    plt.plot(post_3.beta2, color="gray", alpha=0.5)
    plt.plot(post_1.beta2, color="C0")

    plt.title(r"$\beta_2$")

    plt.subplot(5, 1, 3)
    plt.plot(post_2.beta3, color="orange", alpha=0.75)
    plt.plot(post_3.beta3, color="gray", alpha=0.5)
    plt.plot(post_1.beta3, color="C0")

    plt.title(r"$\beta_3$")

    plt.subplot(5, 1, 4)
    plt.plot(post_2.beta4, color="orange", alpha=0.75)
    plt.plot(post_3.beta4, color="gray", alpha=0.5)
    plt.plot(post_1.beta4, color="C0")
    plt.title(r"$\beta_4$")

    plt.subplot(5, 1, 5)
    plt.plot(post_2.beta5, color="orange", alpha=0.75)
    plt.plot(post_3.beta5, color="gray", alpha=0.5)
    plt.plot(post_1.beta5, color="C0")
    plt.title(r"$\beta_5$")

    fig.supxlabel("Iteration")
    fig.supylabel("Parameter estimate")
    plt.tight_layout()
    plt.savefig("three_chain_trace")
    plt.close()


def plot_ward_connectivity(mixing_matrices):
    mean_connectivity_adj = tf.math.reduce_mean(
        mixing_matrices["adj_mats"], axis=0
    )
    mean_connectivity_adj_spatial = tf.math.reduce_mean(
        mixing_matrices["spatial_conn_mats"], axis=0
    )

    # Anonymised version
    cmap = sns.color_palette("mako", as_cmap=True)
    vmin = min(
        tf.math.reduce_min(mean_connectivity_adj),
        tf.math.reduce_min(mean_connectivity_adj_spatial),
    )
    vmax = max(
        tf.math.reduce_max(mean_connectivity_adj),
        tf.math.reduce_max(mean_connectivity_adj_spatial),
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig1 = sns.heatmap(
        mean_connectivity_adj,
        ax=axes[0],
        square=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
    )
    axes[0].invert_yaxis()
    fig2 = sns.heatmap(
        mean_connectivity_adj_spatial,
        ax=axes[1],
        square=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
    )
    axes[1].invert_yaxis()

    fig1.annotate(
        "A", xy=(-0.1, 1), xycoords="axes fraction", color="black", fontsize=20
    )
    fig2.annotate(
        "B", xy=(-0.1, 1), xycoords="axes fraction", color="black", fontsize=20
    )

    fig1.set_xticks(range(1, 56, 3))
    fig1.set_xticklabels(np.arange(1, 56, 3), fontsize=12)
    fig1.set_yticks(range(1, 56, 3))
    fig1.set_yticklabels(np.arange(1, 56, 3), rotation=0, fontsize=12)

    fig2.set_xticks(range(1, 56, 3))
    fig2.set_xticklabels(np.arange(1, 56, 3), fontsize=12)
    fig2.set_yticks(range(1, 56, 3))
    fig2.set_yticklabels(np.arange(1, 56, 3), rotation=0, fontsize=12)

    fig.supxlabel("Ward number")
    fig.supylabel("Ward number")
    plt.tight_layout()
    mappable = fig1.get_children()[0]

    plt.colorbar(
        mappable,
        ax=[axes[0], axes[1]],
        location="right",
        label="ward connectivity",
        pad=0.02,
        shrink=0.5,
    )
    plt.savefig("ward_connectivity")
    plt.close()
