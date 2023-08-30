import xarray as xr
import numpy as np
import pandas as pd
import random
from datetime import datetime
import math
from math import exp
from pandas.core.common import flatten
import tensorflow as tf
import model_funcs as mf


def run_stoch_model(
    parms,
    pat_first_pos,
    mixing_matrices,
    study_start_time,
    t_end,
    pids,
    hospital_event_times,
    sim_seed,
    pre_study_se_events=None,
):
    # Create initial population
    state = mf.set_initial_pop(
        pids, pat_first_pos, study_start_time, pre_study_se_events
    ).T

    t_0 = 0
    t = t_0

    transmission_events = {}
    i = 0
    hospital_num = 0

    rng = np.random.default_rng(seed=sim_seed)

    while t < t_end:
        total_event_rate, transition_rates = sim_transition_rates(
            mixing_matrices, hospital_num, parms, state
        )

        exp_rate_sum = tf.reduce_sum(transition_rates[0])
        inf_rate_sum = tf.reduce_sum(transition_rates[1])

        # Check if sum of event rates = 0
        if total_event_rate == 0:
            dt = math.inf
        else:
            dt = (
                -np.log(rng.uniform(0, 1)) / total_event_rate
            )  # Time until next event

        # Determine if a hospital event or transmission event happens next
        if hospital_event_times[hospital_num] - t < dt:
            t = hospital_event_times[hospital_num]  # Ipdate time
            hospital_num += 1
            continue

        # If time to next event exceeds the time limit then break
        if t + dt > t_end:
            break
        t = t + dt

        # Determine which transmission event happens next
        uni_rate_sum = rng.uniform(0, 1) * total_event_rate

        if uni_rate_sum < exp_rate_sum:
            # S->E chosen as next event, work out which individuals change status
            foi_prob = np.array(transition_rates[0] / exp_rate_sum)
            rand_pid = rng.choice(
                range(0, foi_prob.shape[0]), p=foi_prob.flatten()
            )

            state[0, rand_pid] = state[0, rand_pid] - 1
            state[1, rand_pid] = state[1, rand_pid] + 1
            transmission_events[i] = {
                "pid": state[:, rand_pid].coords["pid"].values,
                "days_to_event": t,
                "event": "s_to_e",
            }
            i += 1

        elif uni_rate_sum <= (exp_rate_sum + inf_rate_sum):
            # E->I chosen as next event, randomly select individual to change state
            rand_pid = rng.choice(
                state[:, state[1, :].values > 0].coords["pid"]
            )
            state.loc["E", rand_pid] = state.loc["E", rand_pid] - 1
            state.loc["I", rand_pid] = state.loc["I", rand_pid] + 1
            transmission_events[i] = {
                "pid": rand_pid,
                "days_to_event": t,
                "event": "e_to_i",
            }
            i += 1

        else:
            # I->R chosen as next event, randomly select individual to change state
            rand_pid = rng.choice(
                state[:, state[2, :].values > 0].coords["pid"]
            )
            state.loc["I", rand_pid] = state.loc["I", rand_pid] - 1
            state.loc["R", rand_pid] = state.loc["R", rand_pid] + 1
            transmission_events[i] = {
                "pid": rand_pid,
                "days_to_event": t,
                "event": "i_to_r",
            }
            i += 1

    return transmission_events


# Adapted make_compute_transition_rate function for simulation
def sim_transition_rates(mixing_matrices, hospital_num, parms, state):
    state = np.array(state, dtype=np.float32)
    memb_mat_t = tf.gather(
        mixing_matrices["memb_mats"],
        hospital_num,
    )
    adj_mat_t = tf.gather(
        mixing_matrices["adj_mats"],
        hospital_num,
    )
    is_in_hospital = tf.gather(
        mixing_matrices["hospital_status_mats"],
        hospital_num,
    )

    spatial_conn_t = tf.gather(
        mixing_matrices["spatial_conn_mats"],
        hospital_num,
    )

    # Number of infectious individuals in each group
    inf_group = tf.linalg.matvec(
        memb_mat_t,
        tf.gather(state, 2) * is_in_hospital,
        transpose_a=True,
    )

    foi_group = (
        parms["beta1"] * inf_group
        + parms["beta2"] * tf.linalg.matvec(adj_mat_t, inf_group)
        + parms["beta3"] * tf.linalg.matvec(spatial_conn_t, inf_group)
        + parms["beta4"]
    )

    # Force of infection for each individual present in the hospital
    susceptibles = tf.gather(state, 0)
    foi_cov_pos = tf.math.reduce_sum(
        (
            (memb_mat_t * susceptibles[:, tf.newaxis])
            * is_in_hospital[:, tf.newaxis]
        )
        * foi_group,
        axis=1,
    )

    # Force of infection for each individual not in the hospital
    foi_cov_neg = (
        susceptibles[:, tf.newaxis]
        * (1 - is_in_hospital[:, tf.newaxis])
        * parms["beta5"]
    )

    foi = foi_cov_pos[:, tf.newaxis] + foi_cov_neg

    # Calculate transition rates
    se_rate = foi
    ei_rate = tf.gather(state, 1)[:, tf.newaxis] * parms["alpha"]
    ir_rate = tf.gather(state, 2)[:, tf.newaxis] * parms["gamma"]
    cov_rate = tf.ones_like(ir_rate)

    transition_rates = tf.stack([se_rate, ei_rate, ir_rate, cov_rate])

    total_event_rate = tf.reduce_sum(tf.stack([se_rate, ei_rate, ir_rate]))
    return total_event_rate, transition_rates
