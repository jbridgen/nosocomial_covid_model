import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
import general_funcs as gf


def initial_set_up(
    t_end=28,
    study_start_time=np.datetime64("2020-04-12T20:00:00.000000000"),
    load_from_previous_sim=True,
    save_sim=False,
):
    """Function to set initial variables for inference and posterior analysis. Returns dictionary of variables."""

    # Set data path
    path = "/data"

    # Set model parameters
    R_w = 0.11  # within ward
    R_h = 0.0008  # between ward
    R_s = 0.0004  # ward proximity
    beta4 = 1.5e-14  # background hospital transmission rate
    beta5 = 1e-10  # community infection rate

    parms = set_params(R_w, R_h, R_s, beta4, beta5)
    end_date = study_start_time + np.timedelta64(t_end, "D")
    # Load patient infection data
    pat_first_pos = gf.load_pat_pos(path, study_start_time, end_date)

    # Set precision
    DTYPE = tf.float32

    # Load previous simulation or run a new simulation
    if load_from_previous_sim:
        (
            all_events,
            mixing_matrices,
            global_ids,
            hospital_events,
            hospital_event_times,
        ) = gf.load_previous_sim(path, DTYPE)
    else:
        (
            all_events,
            mixing_matrices,
            global_ids,
            hospital_events,
            hospital_event_times,
        ) = gf.run_sim(
            pat_first_pos,
            path,
            parms,
            t_end,
            study_start_time,
            DTYPE,
            save_sim,
        )

    model_vars = {
        "parms": parms,
        "study_start_time": study_start_time,
        "t_end": t_end,
        "pat_first_pos": pat_first_pos,
        "all_events": all_events,
        "mixing_matrices": mixing_matrices,
        "global_ids": global_ids,
        "hospital_events": hospital_events,
        "hospital_event_times": hospital_event_times,
        "path": path,
    }

    return model_vars


def get_initial_infected(pat_first_pos, pids, study_start_time):
    inf_pids = pat_first_pos[
        (
            pat_first_pos.Date_Time_Collected
            <= study_start_time + pd.Timedelta(2, unit="D")
        )
        & (
            pat_first_pos.Date_Time_Collected
            >= study_start_time - pd.Timedelta(3, unit="D")
        )
    ][["pid"]]
    initial_inf_index = inf_pids.merge(pids, on="pid", how="left")
    initial_inf_index = (
        initial_inf_index[~pd.isnull(initial_inf_index.pid_index)]
        .pid_index.astype(int)
        .values
    )
    return initial_inf_index


def set_initial_pop(
    pids, pat_first_pos, study_start_time, pre_study_se_events
):
    status = pd.DataFrame(pids, columns=["pid"])
    status = status.assign(S=1, E=0, I=0, R=0)
    status = xr.DataArray(status.set_index("pid"))

    # Set initial exposed
    if pre_study_se_events is not None:
        for i in pre_study_se_events:
            status[i, :] = 0
            status[i, 1] = 1

    # Set initial infected
    initial_infected_index = get_initial_infected(
        pat_first_pos, pids, study_start_time
    )
    for j in initial_infected_index:
        status[j, :] = 0
        status[j, 2] = 1
    return status


def set_params(R_w, R_h, R_s, beta4, beta5):
    parms = {
        "alpha": 1 / 4,
        "gamma": 1 / 5,
        "beta1": None,
        "beta2": None,
        "beta3": None,
        "beta4": beta4,
        "beta5": beta5,
    }

    parms["beta1"] = R_w * parms["gamma"]
    parms["beta2"] = R_h * parms["gamma"]
    parms["beta3"] = R_s * parms["gamma"]

    return parms


def adjust_pre_study_mixing(mixing_matrices):
    """set initial mixing matrices to 0 to allow for community exposures and ensure that
    initially infected are not contributing to the event rates"""

    def update_initial_mixing(mixing_matrices, var):
        mixing_matrices[var] = mixing_matrices[var] * tf.concat(
            [
                tf.zeros_like(mixing_matrices[var][0:1]),
                (tf.ones_like(mixing_matrices[var][0:-1])),
            ],
            axis=0,
        )
        return mixing_matrices

    mixing_matrices = update_initial_mixing(mixing_matrices, "memb_mats")
    mixing_matrices = update_initial_mixing(mixing_matrices, "adj_mats")
    mixing_matrices = update_initial_mixing(
        mixing_matrices, "hospital_status_mats"
    )
    mixing_matrices = update_initial_mixing(
        mixing_matrices, "study_status_mats"
    )
    mixing_matrices = update_initial_mixing(
        mixing_matrices, "spatial_conn_mats"
    )

    return mixing_matrices
