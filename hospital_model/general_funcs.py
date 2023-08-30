import bz2
import datetime as dt
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp

import simulation as sim

tfd = tfp.distributions


def string_to_time(x):
    """Change data type of pandas DataFrame column to datetime tz = Europe/London"""
    try:
        return np.datetime64(datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return pd.NaT


def load_pat_pos(path, study_start_time, end_date):
    """Loads positive patient swab data"""
    # Load clean data
    pat_first_pos_by_spell = pd.read_csv(
        os.path.join(path, "pat_pos_swabs.csv"), index_col=0
    )

    pat_first_pos_by_spell["Date_Time_Collected"] = pat_first_pos_by_spell[
        "Date_Time_Collected"
    ].apply(string_to_time)

    # Keep swab data for 3 days prior to study start date so currently infected patients at day 0 are known
    pat_first_pos_by_spell = pat_first_pos_by_spell[
        (
            pat_first_pos_by_spell["Date_Time_Collected"]
            >= study_start_time - np.timedelta64(3, "D")
        )
        & (
            pat_first_pos_by_spell["Date_Time_Collected"]
            <= end_date + np.timedelta64(2, "D")
        )
    ]

    return pat_first_pos_by_spell


def load_sim(path):
    """Loads clean simulation data"""
    output = pd.read_csv(os.path.join(path, "sim_output.csv"))
    return output


def load_pickled_file(path, file_name):
    """Loads in a compressed pickled file"""
    infile = bz2.BZ2File(os.path.join(path, file_name), "rb")
    file_loaded = pickle.load(infile)
    infile.close()
    return file_loaded


def load_ward_colours(path, study_start_time, t_end):
    """Loads in ward colours"""
    ward_cols = pd.read_csv(os.path.join(path, "ward_colours.csv"))

    for i in range(0, len(ward_cols)):
        ward_cols.loc[i, "date"] = np.datetime64(
            datetime.strptime(ward_cols.loc[i, "date"], "%d/%m/%Y")
        )
    ward_cols = ward_cols[
        (ward_cols.date >= study_start_time)
        & (ward_cols.date <= study_start_time + pd.Timedelta(t_end, unit="D"))
    ]
    ward_cols.reset_index(drop=True, inplace=True)
    ward_cols = ward_cols.rename_axis("days_from_study_start").reset_index()
    return ward_cols


def save_output(output, path):
    """Saves simulation output"""
    filename = "output_vec"
    outfile = open(os.path.join(path, filename), "wb")
    pickle.dump(output, outfile)
    outfile.close()


def clean_sim_output(output, t_end, pids, save, path, hospital_event_times):
    """Cleans simulation output"""

    seir_events = (
        pd.DataFrame.from_dict(output, orient="index")
        .sort_values("days_to_event")
        .reset_index()
    )
    seir_events["days_to_event"] = seir_events.days_to_event.astype(float)
    seir_events["type"] = "T"
    seir_events.pid = seir_events.pid.astype("string")
    seir_events = pd.merge(seir_events, pids, how="left", on=["pid"])

    hospital_events = pd.DataFrame(
        {
            "type": "H",
            "days_to_event": hospital_event_times[
                hospital_event_times < (t_end)
            ],
        }
    )

    # Combine hospital and transmission events
    all_events = (
        pd.concat([seir_events, hospital_events], ignore_index=True)
        .sort_values("days_to_event")
        .reset_index()
    )
    all_events = all_events[
        ["days_to_event", "type", "event", "pid", "pid_index"]
    ]
    all_events = all_events.astype({"pid_index": "Int32"})

    if save == True:
        filename = "all_events_sim_new"
        outfile = bz2.BZ2File(os.path.join(path, filename), "wb")
        pickle.dump(all_events, outfile)
        outfile.close()

    return all_events, hospital_events


def load_all_events_sim(path):
    """Loads in event list from previous simulation"""
    infile = bz2.BZ2File(os.path.join(path, "all_events_sim_new"), "rb")
    all_events = pickle.load(infile)
    infile.close()
    return all_events


def load_previous_sim(path, DTYPE):
    """Loads pids, connectivity matrices, and event list from previous sim"""

    conn_mats = load_connectivity_matrices(path)

    mixing_matrices = dict(
        memb_mats=tf.convert_to_tensor(conn_mats.memb_mats, DTYPE),
        adj_mats=tf.convert_to_tensor(conn_mats.adj_mats, DTYPE),
        hospital_status_mats=tf.convert_to_tensor(
            conn_mats.hospital_status_mats, DTYPE
        ),
        study_status_mats=tf.convert_to_tensor(
            conn_mats.study_status_mats, DTYPE
        ),
        spatial_conn_mats=tf.convert_to_tensor(
            conn_mats.spatial_conn_mats, DTYPE
        ),
    )

    pids = load_pickled_file(path, "pids_saved")
    hospital_event_times = load_pickled_file(
        path, "hospital_event_times_saved"
    )

    all_events = load_pickled_file(path, "all_events_sim_new")
    hospital_events = all_events[all_events.type == "H"]

    return (
        all_events,
        mixing_matrices,
        pids,
        hospital_events,
        hospital_event_times,
    )


def run_sim(
    pat_first_pos, path, parms, t_end, study_start_time, DTYPE, save_sim
):
    """Runs a new simulation"""

    # Loads connectivity matrices
    conn_mats = load_connectivity_matrices(path)

    # Convert connectivity matrices to tensors
    mixing_matrices = dict(
        memb_mats=tf.convert_to_tensor(conn_mats.memb_mats, DTYPE),
        adj_mats=tf.convert_to_tensor(conn_mats.adj_mats, DTYPE),
        hospital_status_mats=tf.convert_to_tensor(
            conn_mats.hospital_status_mats, DTYPE
        ),
        study_status_mats=tf.convert_to_tensor(
            conn_mats.study_status_mats, DTYPE
        ),
        spatial_conn_mats=tf.convert_to_tensor(
            conn_mats.spatial_conn_mats, DTYPE
        ),
    )

    # Load pids and hospital event times (time points when the hospital contact network is updated)
    pids = load_pickled_file(path, "pids_saved")
    hospital_event_times = load_pickled_file(
        path, "hospital_event_times_saved"
    )

    # Run simulation and format output
    output = sim.run_stoch_model(
        parms,
        pat_first_pos,
        mixing_matrices,
        study_start_time,
        t_end,
        pids,
        hospital_event_times,
        sim_seed=153,
        pre_study_se_events=None,
    )
    all_events, hospital_events = clean_sim_output(
        output, t_end, pids, save_sim, path, hospital_event_times
    )

    return (
        all_events,
        mixing_matrices,
        pids,
        hospital_events,
        hospital_event_times,
    )


def event_times_to_list(transmission_events, hospital_events):
    """Reformats event times to a list format"""
    transmission_events = pd.melt(
        transmission_events,
        id_vars=["pid_index"],
        var_name="event",
        value_name="days_to_event",
    )
    transmission_events["type"] = "T"

    hospital_events["type"] = "H"

    event_list = (
        pd.concat([transmission_events, hospital_events], ignore_index=True)
        .sort_values("days_to_event")
        .reset_index()
    )
    event_list = event_list[["days_to_event", "type", "event", "pid_index"]]
    event_list = event_list.astype({"pid_index": "Int32"})

    return event_list


def transform_sim_data(all_events):
    seir_events = all_events[all_events.type == "T"]
    transmission_events = seir_events.pivot(
        index="pid_index", columns="event", values="days_to_event"
    )[["s_to_e", "e_to_i", "i_to_r"]]

    return transmission_events


def transform_hosp_data(pat_first_pos, pids, study_start_time, t_end):
    """set initial values for SE, EI and IR transition events"""

    infection_event = pat_first_pos[["pid", "Date_Time_Collected"]].copy()
    infection_event = infection_event[
        (
            infection_event.Date_Time_Collected
            >= study_start_time - pd.Timedelta(3, unit="D")
        )
        & (
            infection_event.Date_Time_Collected
            <= study_start_time + pd.Timedelta(t_end + 2, unit="D")
        )
    ]
    infection_event = pd.merge(infection_event, pids, on="pid", how="left")

    for i in range(0, len(infection_event)):
        infection_event.loc[i, "days_to_event"] = (
            infection_event.loc[i, "Date_Time_Collected"] - study_start_time
        ) / dt.timedelta(days=1)

    infection_event = infection_event.dropna()
    infection_event.pid_index = infection_event.pid_index.astype(int)
    infection_event.set_index("pid_index", inplace=True)
    transmission_events = infection_event[["days_to_event"]].copy()
    transmission_events = transmission_events.rename(
        columns={"days_to_event": "e_to_i"}
    )
    transmission_events["e_to_i"] = transmission_events.e_to_i - 2
    transmission_events["s_to_e"] = transmission_events.e_to_i - 4
    transmission_events["i_to_r"] = transmission_events.e_to_i + 5

    transmission_events.e_to_i[transmission_events.e_to_i < 0] = np.nan
    transmission_events.s_to_e[pd.isnull(transmission_events.e_to_i)] = np.nan
    transmission_events = transmission_events[["s_to_e", "e_to_i", "i_to_r"]]

    return transmission_events


def load_connectivity_matrices(path):
    """Loads in netcdf file containing connectivity matrices"""
    conn_mats = xr.open_dataset(os.path.join(path, "connectivity_matrices.nc"))

    return conn_mats


def save_mcmc_output(samples, events, fixed_event_list, burn_in, path, chain):
    """saves posterior draws of transmission rate parameters and event times"""
    samples_reduced = {
        "beta1": samples.beta1[burn_in:],
        "beta2": samples.beta2[burn_in:],
        "beta3": samples.beta3[burn_in:],
        "beta4": samples.beta4[burn_in:],
        "beta5": samples.beta5[burn_in:],
    }

    events_reduced = {
        "time": events.time[burn_in:],
        "unit": events.unit[burn_in:],
        "event": events.event[burn_in:],
    }

    def to_data_array(tensor):
        return xr.DataArray(tensor, dims=["iteraton"])

    post_param_draws = xr.Dataset(
        {k: to_data_array(v) for k, v in samples_reduced.items()}
    )

    # Saves netcdf of EventList
    post_draws = xr.merge(
        [
            xr.DataArray(events_reduced["time"]).to_dataset(name="time"),
            xr.DataArray(events_reduced["unit"]).to_dataset(name="unit"),
            xr.DataArray(events_reduced["event"]).to_dataset(name="event"),
            post_param_draws,
        ]
    )

    post_draws.to_netcdf(
        os.path.join(path, chain + "post_draws_saved.nc"),
        encoding={
            "time": {"zlib": True, "complevel": 9},
            "unit": {"zlib": True, "complevel": 9},
            "event": {"zlib": True, "complevel": 9},
            "beta1": {"zlib": True, "complevel": 9},
            "beta2": {"zlib": True, "complevel": 9},
            "beta3": {"zlib": True, "complevel": 9},
            "beta4": {"zlib": True, "complevel": 9},
            "beta5": {"zlib": True, "complevel": 9},
        },
    )

    fixed_ei_events = xr.merge(
        [
            xr.DataArray(fixed_event_list.time).to_dataset(name="time"),
            xr.DataArray(fixed_event_list.unit).to_dataset(name="unit"),
            xr.DataArray(fixed_event_list.event).to_dataset(name="event"),
        ]
    )

    fixed_ei_events.to_netcdf(
        os.path.join(path, "fixed_ei_events_saved.nc"),
        encoding={
            "time": {"zlib": True, "complevel": 9},
            "unit": {"zlib": True, "complevel": 9},
            "event": {"zlib": True, "complevel": 9},
        },
    )


def load_mcmc_output(path, chain="chain_1"):
    post_draws_loaded = xr.open_dataset(
        os.path.join(path, chain + "post_draws_saved.nc")
    )
    fixed_ei_events_loaded = xr.open_dataset(
        os.path.join(path, "fixed_ei_events_saved.nc")
    )

    return post_draws_loaded, fixed_ei_events_loaded


def load_all_chains(model_vars):
    post_1, fixed_event_list = load_mcmc_output(
        path=model_vars["path"], chain="chain_1"
    )
    post_2, _ = load_mcmc_output(path=model_vars["path"], chain="chain_2")
    post_3, _ = load_mcmc_output(path=model_vars["path"], chain="chain_3")

    return post_1, fixed_event_list, post_2, post_3
