import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tf_lik_funcs as lf
import model_funcs as mf

tfd = tfp.distributions


def build_logposterior_function(
    hospital_event_times,
    study_start_time,
    pids,
    transmission_table_pids,
    pat_first_pos,
    mixing_matrices,
    fixed_parms,
    fixed_event_list,
    fit_to_hospital_data,
):
    # Set initial conditions
    initial_conditions = np.array(
        mf.set_initial_pop(
            pids, pat_first_pos, study_start_time, pre_study_se_events=None
        ).T,
        np.float32,
    )

    num_individuals = mixing_matrices["memb_mats"].shape[1]
    num_wards = mixing_matrices["memb_mats"].shape[2]

    incidence_matrix = np.array(
        [[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]],
        dtype=np.float32,
    )

    num_states, num_events = incidence_matrix.shape

    fixed_event_list = tf.nest.map_structure(
        lambda x: tf.convert_to_tensor(x), fixed_event_list
    )

    def log_posterior_function(current_state, transmission_events):
        transmission_events = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x), transmission_events
        )

        parms = {
            "beta1": tf.convert_to_tensor(current_state.beta1),
            "beta2": tf.convert_to_tensor(current_state.beta2),
            "beta3": tf.convert_to_tensor(current_state.beta3),
            "beta4": tf.convert_to_tensor(current_state.beta4),
            "beta5": tf.convert_to_tensor(current_state.beta5),
            **fixed_parms,
        }

        priors = {
            "beta1": tfd.Gamma(
                concentration=1.1,
                rate=1000,
            ),
            "beta2": tfd.Gamma(
                concentration=1.1,
                rate=1000,
            ),
            "beta3": tfd.Gamma(
                concentration=1.1,
                rate=1000,
            ),
            "beta4": tfd.Gamma(
                concentration=1.1,
                rate=1000,
            ),
            "beta5": tfd.Gamma(
                concentration=1.1,
                rate=1000,
            ),
        }

        hospital_model = lf.ContinuousTimeModel(
            initial_conditions=initial_conditions,
            num_individuals=num_individuals,
            num_wards=num_wards,
            ids=transmission_table_pids,
            covariate_change_times=hospital_event_times,
            transition_rate_fn=lf.make_compute_transition_rates(
                mixing_matrices, parms
            ),
            incidence_matrix=incidence_matrix,
            fixed_event_list=fixed_event_list,
            fit_to_hospital_data=fit_to_hospital_data,
        )

        return tf.reduce_sum(
            [rv.log_prob(parms[p]) for p, rv in priors.items()]
        ) + hospital_model.log_prob(transmission_events)

    return log_posterior_function
