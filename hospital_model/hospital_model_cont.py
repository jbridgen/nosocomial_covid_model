import sys
from collections import namedtuple
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from numpy.random import SeedSequence

import general_funcs as gf
import model_funcs as mf
import tf_lik
from tf_lik_funcs import ContinuousTimeModel
import tf_mcmc as mcmc
import simulation as sim
import att_fracts_funcs as af
import plotting_funcs as pf


def run_inference(
    load_from_previous_sim=True,
    fit_to_hospital_data=True,
    its=11000,
    burn_in=1000,
    chain="chain_1",
    save=True,
):
    """
    Function to run the inference.

    arguments:
    fit_to_hospital_data default value is True. MCMC will fit the event times from the hospital data rather than simulation.
    its is the number of iterations for the MCMC.
    save_mats default value False. Writes mixing_matrices and hospital_event_times if a new simulation is created.
    create_mats default value False. Computes mixing_matrices and hospital_event_times if a new simulation is created.
    burn_in is the number of iterations to disregard from the MCMC as burn in.
    chain default value of "chain_1". Specify to use starting values for the second and third chain.
    """

    current_dateTime = datetime.now()

    model_vars = mf.initial_set_up(
        load_from_previous_sim=load_from_previous_sim,
    )
    mixing_matrices = model_vars["mixing_matrices"]

    hospital_event_times = tf.reshape(
        np.array(
            [model_vars["hospital_events"].days_to_event], dtype=np.float32
        ),
        [-1],
    )

    # Create event table
    if fit_to_hospital_data:
        transmission_events = gf.transform_hosp_data(
            model_vars["pat_first_pos"],
            model_vars["global_ids"],
            model_vars["study_start_time"],
            model_vars["t_end"],
        )
        mixing_matrices = mf.adjust_pre_study_mixing(mixing_matrices)
    else:
        mixing_matrices["study_status_mats"] = tf.ones(
            mixing_matrices["study_status_mats"].shape
        )  # Set is_in_study_mats to 1 - obsolete for simulation
        transmission_events = gf.transform_sim_data(model_vars["all_events"])

    # Transform event list to event table
    transmission_table_pids = tf.convert_to_tensor(
        transmission_events.reset_index().pid_index, np.int32
    )
    event_times_table = transmission_events.to_numpy().astype(np.float32).T

    chain_states = mcmc.get_chain_states()
    seeds = {"chain_1": [0, 0], "chain_2": [1, 1], "chain_3": [2, 2]}

    # Create event list and fixed event list for transmission events
    fixed_event_list, event_list = ContinuousTimeModel.event_table_to_list(
        event_times_table, transmission_table_pids
    )

    fixed_parms = {
        k: v
        for k, v in model_vars["parms"].items()
        if k not in ["beta1", "beta2", "beta3", "beta4", "beta5"]
    }

    target_log_prob_fn = tf_lik.build_logposterior_function(
        hospital_event_times,
        model_vars["study_start_time"],
        model_vars["global_ids"],
        transmission_table_pids,
        model_vars["pat_first_pos"],
        mixing_matrices,
        fixed_parms,
        fixed_event_list,
        tf.constant(fit_to_hospital_data),
    )
    tlp = target_log_prob_fn

    tlp(chain_states[chain], event_list)

    # Run log MH algo
    samples, events, results = mcmc.run_mcmc(
        num_samples=its,
        current_state=chain_states[chain],
        target_log_prob_fn=tlp,
        transmission_events=event_list,
        seed=seeds[chain],
    )

    # Calculate acceptance ratios for each parameter
    acceptance_ratios = {
        "beta1": mcmc.accepted_ratio(results, "beta1", burn_in),
        "beta2": mcmc.accepted_ratio(results, "beta2", burn_in),
        "beta3": mcmc.accepted_ratio(results, "beta3", burn_in),
        "beta4": mcmc.accepted_ratio(results, "beta4", burn_in),
        "beta5": mcmc.accepted_ratio(results, "beta5", burn_in),
        "events": tf.reduce_mean(results["events_acceptance"][burn_in:]),
    }

    print(acceptance_ratios)
    end_dateTime = datetime.now()
    print("runtime: ", end_dateTime - current_dateTime)

    # Save plots
    if save:
        gf.save_mcmc_output(
            samples,
            events,
            fixed_event_list,
            burn_in,
            model_vars["path"],
            chain,
        )

    return samples, results, acceptance_ratios


def simulate_from_posterior_draws(
    load_from_previous_sim=True, chain="chain_1", figure_version="v1"
):
    """Function to forward simulate an epidemic using parameter values and initial population states from posterior draws.

    Arguments:
    load_from_previous_sim default value is True. Used to load in previous computed mixing matrices and simulation.
    chain default value is "chain_1". Specify to draw from an alternative posterior distribution.

    Returns:
    post is a xarray of event times and parameter value for each posterior draw. Created and saved via run_inference function.
    simulated_infection_events is a list of NamedTuples containing the event times for a range of simulations using
    parameter values and starting population states from posterior draws.
    fixed_ei_event_list contains the EI events from the hospital swab data
    """

    # Load in initial model variables and posterior draws
    model_vars = mf.initial_set_up(
        load_from_previous_sim=load_from_previous_sim,
    )
    post, fixed_ei_event_list = gf.load_mcmc_output(
        path=model_vars["path"], chain=chain
    )

    parms = model_vars["parms"]
    gamma = parms["gamma"]
    pids = model_vars["global_ids"]

    # Set seeds
    seed_seq = SeedSequence(4356)
    seeds = seed_seq.spawn(post.beta1.shape[0])
    simulated_infection_events = []
    for i in range(0, post.beta1.shape[0] - 1):
        parms = mf.set_params(
            (post.beta1[i] / gamma).item(),
            (post.beta2[i] / gamma).item(),
            (post.beta3[i] / gamma).item(),
            (post.beta4[i]).item(),
            (post.beta5[i]).item(),
        )

        # Identify any SE events that are recorded before study start time - time < 0
        pre_study_se_events_ind = list(
            list(np.where((post.time[i] < 0.0) & (post.event[i] == 0))).pop(0)
        )
        pre_study_se_events_pid = np.array(
            [post.unit[i][j] for j in pre_study_se_events_ind]
        )

        # Run simulation
        output = sim.run_stoch_model(
            parms,
            model_vars["pat_first_pos"],
            model_vars["mixing_matrices"],
            model_vars["study_start_time"],
            model_vars["t_end"],
            pids,
            model_vars["hospital_event_times"],
            seeds[i],
            pre_study_se_events_pid,
        )

        save = False
        all_events, hospital_events = gf.clean_sim_output(
            output,
            model_vars["t_end"],
            pids,
            save,
            model_vars["path"],
            model_vars["hospital_event_times"],
        )

        ei_events = all_events.loc[all_events.event == "e_to_i"]
        ei_event_times = ei_events.days_to_event
        ei_event_times = np.floor(ei_event_times).astype(int)
        sim_time_series = pd.DataFrame({"day": ei_event_times, "freq": 1})
        sim_time_series = sim_time_series.groupby(by="day").sum().reset_index()
        days = pd.DataFrame({"day": range(0, model_vars["t_end"])})
        sim_time_series = sim_time_series.merge(
            days, on="day", how="right"
        ).fillna(0)
        simulated_infection_events.append(sim_time_series)

    pf.plot_post_pred_infections(
        fixed_ei_event_list,
        simulated_infection_events,
        figure_version,
        chain,
    )

    return post, simulated_infection_events, fixed_ei_event_list


def analyse_post_output(
    load_from_previous_sim=True,
    chain="chain_1",
    fig_version="v1",
):
    # Load initial model variables and posterior draws
    model_vars = mf.initial_set_up(
        load_from_previous_sim=load_from_previous_sim,
    )
    post, fixed_event_list = gf.load_mcmc_output(
        path=model_vars["path"], chain=chain
    )

    # Create density plots - Manuscript Figure 3
    pf.create_density_plot(post, fixed_event_list, fig_version)

    EventList = namedtuple("EventList", ["time", "unit", "event"])
    fixed_event_list = EventList(
        tf.convert_to_tensor(fixed_event_list.time),
        tf.convert_to_tensor(fixed_event_list.unit),
        tf.convert_to_tensor(fixed_event_list.event),
    )

    transmission_events = gf.transform_hosp_data(
        model_vars["pat_first_pos"],
        model_vars["global_ids"],
        model_vars["study_start_time"],
        model_vars["t_end"],
    )
    transmission_table_pids = tf.convert_to_tensor(
        transmission_events.reset_index().pid_index, np.int32
    )

    # Load ward list and ward colours
    all_wards = gf.load_pickled_file(model_vars["path"], "wards_saved")
    ward_cols = gf.load_ward_colours(
        model_vars["path"], model_vars["study_start_time"], model_vars["t_end"]
    )

    hospital_event_times = tf.reshape(
        np.array(
            [model_vars["hospital_events"].days_to_event], dtype=np.float32
        ),
        [-1],
    )

    # Calculate attributable fractions for each set of posterior values
    (
        att_fracts_all,
        exp_by_ward,
        ward_foi_all,
        ward_foi_within_ward_all,
        ward_foi_between_ward_all,
        ward_foi_background_all,
        ward_cols_nosoc,
    ) = af.determine_att_fracts(
        post,
        model_vars,
        transmission_table_pids,
        fixed_event_list,
        all_wards,
        hospital_event_times,
        ward_cols,
    )
    # Create Manuscript Figure 5 and 6
    (
        mean_nosoc_att_fract,
        perc_nosoc_exp_by_ward,
        perc_ward_cols_nosoc,
    ) = pf.plot_att_fracts(
        att_fracts_all,
        exp_by_ward,
        ward_cols_nosoc,
        ward_foi_all,
        ward_foi_within_ward_all,
        ward_foi_between_ward_all,
        ward_foi_background_all,
        hospital_event_times,
        model_vars["path"],
        fig_version,
        chain,
    )

    # Create network figure - Manuscript figure S1
    pf.plot_ward_connectivity(model_vars["mixing_matrices"])

    # Evaluate chain convergence - Manuscript figure S2
    pf.evaluate_chain_convergence(model_vars["path"])

    return (
        mean_nosoc_att_fract,
        perc_nosoc_exp_by_ward,
        perc_ward_cols_nosoc,
    )
