import tensorflow as tf
import numpy as np
import pandas as pd
import plotting_funcs as pf
from collections import namedtuple
import tf_att_fracts


def att_fract_fn(mixing_matrices, parms):
    def compute_att_fracts(args):
        state, covariate_pointers, event_id, event_pid = args

        memb_mat_t = tf.gather(
            mixing_matrices["memb_mats"],
            covariate_pointers,
            name="gather_memb_mats",
        )

        adj_mat_t = tf.gather(
            mixing_matrices["adj_mats"],
            covariate_pointers,
            name="gather_adj_mats",
        )

        is_in_hospital_t = tf.gather(
            mixing_matrices["hospital_status_mats"],
            covariate_pointers,
            name="gather_hosp_status",
        )

        spatial_conn_t = tf.gather(
            mixing_matrices["spatial_conn_mats"],
            covariate_pointers,
            name="gather_spatial_status",
        )

        # Number of infectious individuals in each group
        inf_group = tf.linalg.matvec(
            memb_mat_t,
            tf.gather(state, 2) * is_in_hospital_t,
            transpose_a=True,
            name="inf_group_matmul",
        )

        foi_group = (
            parms["beta1"] * inf_group
            + parms["beta2"]
            * tf.linalg.matvec(
                adj_mat_t, inf_group, name="adj_mat_t_inf_group_matvec"
            )
            + parms["beta3"]
            * tf.linalg.matvec(
                spatial_conn_t,
                inf_group,
                name="spatial_mat_t_inf_group_matvec",
            )
            + parms["beta4"]
        )

        # FOI components
        foi_within_ward = parms["beta1"] * inf_group
        foi_between_ward = parms["beta2"] * tf.linalg.matvec(
            adj_mat_t, inf_group, name="adj_mat_t_inf_group_matvec"
        ) + parms["beta3"] * tf.linalg.matvec(
            spatial_conn_t,
            inf_group,
            name="spatial_mat_t_inf_group_matvec",
        )
        foi_background = parms["beta4"]

        # Attributable fractions
        pr_within_ward = foi_within_ward / foi_group

        pr_between_ward = foi_between_ward / foi_group

        pr_beta2 = (
            parms["beta2"]
            * tf.linalg.matvec(
                adj_mat_t, inf_group, name="adj_mat_t_inf_group_matvec"
            )
        ) / foi_group

        pr_beta3 = (
            parms["beta3"]
            * tf.linalg.matvec(
                spatial_conn_t,
                inf_group,
                name="spatial_mat_t_inf_group_matvec",
            )
        ) / foi_group

        pr_background = parms["beta4"] / foi_group

        # Identify which ward the exposure event occured in
        ward_exp = tf.cond(
            tf.gather(is_in_hospital_t, event_pid) == 1,
            true_fn=lambda: tf.gather(memb_mat_t, event_pid),
            false_fn=lambda: tf.zeros_like(inf_group),
        )

        # Create dictionary of attributatble fractions - return np.nan if exposure happened in the community
        att_fracts = {
            "event_unit": event_pid,
            "pr_within_ward": tf.reduce_sum(pr_within_ward * ward_exp),
            "pr_between_ward": tf.reduce_sum(pr_between_ward * ward_exp),
            "pr_background": tf.reduce_sum(pr_background * ward_exp),
            "pr_community": tf.cond(
                tf.gather(is_in_hospital_t, event_pid) == 1,
                true_fn=lambda: 0.0,
                false_fn=lambda: 1.0,
            ),
        }

        between_ward_fracts = {
            "event_unit": event_pid,
            "pr_beta2": tf.reduce_sum(pr_beta2 * ward_exp),
            "pr_beta3": tf.reduce_sum(pr_beta3 * ward_exp),
        }

        all_wards_pr = {
            "foi_by_ward": foi_group,
            "foi_within_ward": foi_within_ward * 1,
            "foi_between_ward": foi_between_ward * 1,
            "foi_background": foi_background * 1,
        }

        between_ward_transmission = pr_between_ward * ward_exp

        return (
            att_fracts,
            between_ward_fracts,
            ward_exp,
            between_ward_transmission,
            all_wards_pr,
        )

    return compute_att_fracts


def determine_att_fracts(
    post,
    model_vars,
    transmission_table_pids,
    fixed_event_list,
    all_wards,
    hospital_event_times,
    ward_cols,
):
    between_ward_att_fract = []
    att_fracts_all = []
    exp_by_ward = []
    ward_cols_nosoc = []
    ward_foi_within_ward_all = []
    ward_foi_between_ward_all = []
    ward_foi_background_all = []
    ward_foi_all = []

    EventList = namedtuple("EventList", ["time", "unit", "event"])

    for i in range(0, post.beta1.shape[0] - 1):
        event_list = EventList(
            tf.convert_to_tensor(post.time[i]),
            tf.convert_to_tensor(post.unit[i]),
            tf.convert_to_tensor(post.event[i]),
        )

        ChainState = namedtuple(
            "ChainState", ["beta1", "beta2", "beta3", "beta4", "beta5"]
        )

        fixed_parms = {
            k: v
            for k, v in model_vars["parms"].items()
            if k not in ["beta1", "beta2", "beta3", "beta4", "beta5"]
        }

        compute_att_fracts = tf_att_fracts.build_att_fracts_fn(
            model_vars["hospital_events"],
            model_vars["study_start_time"],
            model_vars["global_ids"],
            transmission_table_pids,
            model_vars["pat_first_pos"],
            model_vars["mixing_matrices"],
            fixed_parms,
            fixed_event_list,
            tf.constant(True),
        )

        post_state = ChainState(
            beta1=post.beta1[i],
            beta2=post.beta2[i],
            beta3=post.beta3[i],
            beta4=post.beta4[i],
            beta5=post.beta5[i],
        )
        (
            att_fracts,
            between_ward_fracts,
            ward_exp,
            between_ward_transmission,
            covariate_pointers_exp,
            all_wards_pr,
        ) = compute_att_fracts(post_state, event_list)

        att_fracts = pd.DataFrame(
            {
                "pr_within_ward": att_fracts["pr_within_ward"],
                "pr_between_ward": att_fracts["pr_between_ward"],
                "pr_background": att_fracts["pr_background"],
                "pr_community": att_fracts["pr_community"],
                "event_unit": att_fracts["event_unit"],
            }
        )

        ward_foi_within_ward = pd.DataFrame(all_wards_pr["foi_within_ward"])
        ward_foi_between_ward = pd.DataFrame(all_wards_pr["foi_between_ward"])
        ward_foi_background = pd.DataFrame(all_wards_pr["foi_background"])
        ward_foi = pd.DataFrame(all_wards_pr["foi_by_ward"])
        ward_foi_cov = pd.DataFrame({"cov_points": covariate_pointers_exp})
        ward_foi = pd.concat([ward_foi, ward_foi_cov], axis=1)
        ward_foi_within_ward = pd.concat(
            [ward_foi_within_ward, ward_foi_cov], axis=1
        )
        ward_foi_between_ward = pd.concat(
            [ward_foi_between_ward, ward_foi_cov], axis=1
        )
        ward_foi_background = pd.concat(
            [ward_foi_background, ward_foi_cov], axis=1
        )

        att_fracts_all.append(att_fracts)
        ward_foi_within_ward_all.append(ward_foi_within_ward)
        ward_foi_between_ward_all.append(ward_foi_between_ward)
        ward_foi_background_all.append(ward_foi_background)
        ward_foi_all.append(ward_foi)

        hosp_exp_between_ward = pd.DataFrame(
            {
                "pr_beta2": between_ward_fracts["pr_beta2"],
                "pr_beta3": between_ward_fracts["pr_beta3"],
                "event_unit": between_ward_fracts["event_unit"],
            }
        )
        between_ward_att_fract.append(hosp_exp_between_ward)

        # Nosocomial exposures by ward
        exposures_on_each_ward = tf.cast(tf.reduce_sum(ward_exp, 0), tf.int32)
        ward_exposures = pd.DataFrame(
            {"wards": all_wards, "exposures": exposures_on_each_ward}
        )
        ward_exposures["perc"] = np.round(
            (ward_exposures["exposures"] / sum(ward_exposures["exposures"]))
            * 100,
            2,
        )
        ward_exposures.sort_values(by=["perc"], ascending=False, inplace=True)
        exp_by_ward.append(ward_exposures)

        # Nosocomial exposure by ward colour
        hosp_exp_inds = tf.where(att_fracts["pr_community"] != 1.0)
        covariate_times_hosp_exp = tf.gather(
            covariate_pointers_exp, hosp_exp_inds
        )
        hosp_exp_time_from_study = tf.gather(
            hospital_event_times, covariate_times_hosp_exp
        )

        ward_exp_hosp = tf.squeeze(tf.gather(ward_exp, hosp_exp_inds))
        ward_exp_hosp = tf.gather(
            tf.where(ward_exp_hosp > 0), indices=1, axis=1
        )
        wards_exp_hosp = [all_wards[x] for x in ward_exp_hosp]

        wards_exp_hosp_col_day = tf.cast(
            tf.math.floor(hosp_exp_time_from_study), tf.int64
        )
        ward_exp_hosp_col = [
            ward_cols[x][y]
            for x, y in zip(
                wards_exp_hosp, np.array(tf.squeeze(wards_exp_hosp_col_day))
            )
        ]
        ward_cols_hosp_exp = pd.DataFrame(
            {"colour": ward_exp_hosp_col, "freq": 1}
        )
        ward_cols_hosp_exp = ward_cols_hosp_exp.groupby(["colour"]).agg(
            {"freq": sum}
        )
        ward_cols_hosp_exp["perc"] = ward_cols_hosp_exp.apply(
            lambda x: 100 * (x / x.sum())
        )

        ward_cols_nosoc.append(ward_cols_hosp_exp)

    return (
        att_fracts_all,
        exp_by_ward,
        ward_foi_all,
        ward_foi_within_ward_all,
        ward_foi_between_ward_all,
        ward_foi_background_all,
        ward_cols_nosoc,
    )
