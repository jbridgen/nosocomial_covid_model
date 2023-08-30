import tensorflow as tf
import numpy as np
import tf_lik_funcs as lf
import model_funcs as mf
import att_fracts_funcs as af


def build_att_fracts_fn(
    hospital_events,
    study_start_time,
    pids,
    transmission_table_pids,
    pat_first_pos,
    mixing_matrices,
    fixed_parms,
    fixed_event_list,
    fit_to_hospital_data,
):
    hospital_event_times = tf.reshape(
        np.array([hospital_events.days_to_event], dtype=np.float32), [-1]
    )

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

    def att_fracts_function(post, transmission_events):
        transmission_events = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x), transmission_events
        )

        parms = {
            "beta1": tf.convert_to_tensor(post.beta1),
            "beta2": tf.convert_to_tensor(post.beta2),
            "beta3": tf.convert_to_tensor(post.beta3),
            "beta4": tf.convert_to_tensor(post.beta4),
            "beta5": tf.convert_to_tensor(post.beta5),
            **fixed_parms,
        }

        hospital_model = lf.ContinuousTimeModel(
            initial_conditions=initial_conditions,
            num_individuals=num_individuals,
            num_wards=num_wards,
            ids=transmission_table_pids,
            covariate_change_times=hospital_event_times,
            transition_rate_fn=af.att_fract_fn(mixing_matrices, parms),
            incidence_matrix=incidence_matrix,
            fixed_event_list=fixed_event_list,
            fit_to_hospital_data=fit_to_hospital_data,
        )
        (
            att_fracts_check,
            between_ward_fracts,
            ward_exp,
            between_ward_transmission,
            covariate_pointers_exp,
            all_wards_pr,
        ) = hospital_model.attributable_fractions(transmission_events)
        return (
            att_fracts_check,
            between_ward_fracts,
            ward_exp,
            between_ward_transmission,
            covariate_pointers_exp,
            all_wards_pr,
        )

    return att_fracts_function
