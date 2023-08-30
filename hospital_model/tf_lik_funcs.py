from collections import namedtuple
import tensorflow as tf
import numpy as np


def expand_event_list(times, pid, event_id, num_individuals, num_events):
    """Expands coordinates [pid, time, event_id] into a dense tensor of
    0s and 1s."""
    with tf.name_scope("expand_event_list"):
        pid = tf.cast(pid, tf.int32)
        num_times = tf.shape(times)[0]  # use rather than .shape[0]

        # Create an array of [pid, time, event] coordinates
        i = tf.range(num_times)
        indices = tf.stack([i, event_id, pid], axis=-1)

        # Scatter a vector of 1s into an array of 0s
        dense_events = tf.scatter_nd(
            indices,
            updates=tf.ones(num_times),
            shape=[num_times, num_events, num_individuals],
        )

        return dense_events


def compute_state(initial_conditions, dense_events, incidence_matrix):
    """Computes a [num_times, num_states, num_individuals] state
    tensor.

    :param initial_conditions: a [num_states, num_individuals] tensor denoting
                               the initial conditions.
    :param dense_events: a [num_times, num_events, num_individuals] tensor
                         denoting events.
    :param incidence_matrix: a [num_states, num_events] tensor denoting the
                             state transition model.
    :returns: a [num_times, num_states, num_individuals] tensor denoting the
              state.
    """

    # First compute the increments we need to add to state
    with tf.name_scope("compute_state"):
        increments = tf.einsum("trk,sr->tsk", dense_events, incidence_matrix)

        # Reconstructs the state by taking a cumsum along the time axis and
        # adding the initial conditions
        state = initial_conditions + tf.cumsum(
            increments, axis=-3, exclusive=True
        )

        return state


def make_compute_transition_rates(mixing_matrices, parms):
    """Make a transition rate computation function

    :param mixing_matrices: a `dict` of mixing_matrices
    :param parms: a `dict` of parameters
    :returns: a function
    """

    parms = {k: tf.convert_to_tensor(v) for k, v in parms.items()}

    def compute_transition_rates(args):
        with tf.name_scope("compute_transition_rates"):
            state, covariate_pointers, event_id, pid = args

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
            is_in_hospital = tf.gather(
                mixing_matrices["hospital_status_mats"],
                covariate_pointers,
                name="gather_hosp_status",
            )

            is_in_study = tf.gather(
                mixing_matrices["study_status_mats"],
                covariate_pointers,
                name="gather_study_status",
            )

            spatial_conn_t = tf.gather(
                mixing_matrices["spatial_conn_mats"],
                covariate_pointers,
                name="gather_spatial_status",
            )

            # Number of infectious individuals in each group
            inf_group = tf.linalg.matvec(
                memb_mat_t,
                tf.gather(state, 2) * is_in_hospital,
                transpose_a=True,
                name="inf_group_matmul",
            )  # Shape [M]

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
            # Force of infection for each individual present in the hospital
            susceptibles = tf.gather(state, 0)

            with tf.name_scope("foi_cov_pos"):
                suscep_and_hosp = tf.math.multiply(
                    susceptibles,
                    is_in_hospital,
                    name="suscep_and_hosp",
                )
                foi = tf.linalg.matvec(
                    memb_mat_t,
                    foi_group,
                    name="memb_and_foi",
                )
                foi_cov_pos = foi * suscep_and_hosp

            # Force of infection for each individual not in the hospital
            with tf.name_scope("foi_cov_neg"):
                foi_cov_neg = (
                    tf.math.multiply(
                        susceptibles,
                        1 - is_in_hospital,
                        name="suscep_not_hosp",
                    )
                    * parms["beta5"]
                )

            foi = foi_cov_pos + foi_cov_neg

            # Calculate transition rates
            se_rate = foi
            ei_rate = tf.gather(state, 1) * parms["alpha"]
            ir_rate = tf.gather(state, 2) * parms["gamma"] * is_in_study
            cov_rate = tf.ones_like(ir_rate)
            transition_rates = tf.stack([se_rate, ei_rate, ir_rate, cov_rate])

            event_rate = tf.gather_nd(
                transition_rates, indices=[[event_id, tf.cast(pid, tf.int32)]]
            )
            total_event_rate = tf.reduce_sum(
                tf.stack([se_rate, ei_rate, ir_rate])
            )

            return total_event_rate, event_rate

    return compute_transition_rates


def compute_loglik(args):
    with tf.name_scope("compute_loglik"):
        total_event_rate, event_rate, time_delta = args
        loglik_t = -total_event_rate * time_delta + tf.math.log(event_rate)

        return loglik_t


class ContinuousTimeModel:
    EventList = namedtuple("EventList", ["time", "unit", "event"])

    def __init__(
        self,
        initial_conditions,
        num_individuals,
        num_wards,
        ids,
        covariate_change_times,
        transition_rate_fn,
        incidence_matrix,
        fixed_event_list,
        fit_to_hospital_data,
    ):
        self._parameters = locals()

        # Pad right-most column of incidence matrix
        # with 0s as a `null` event that does not change
        # the epidemic state.
        self._incidence_matrix = tf.concat(
            [
                self._parameters["incidence_matrix"],
                tf.zeros_like(self._parameters["incidence_matrix"][:, 0:1]),
            ],
            axis=-1,
        )

    @classmethod
    def event_table_to_list(
        cls,
        event_times_table,
        event_table_pid,
    ):
        event_times_table = tf.convert_to_tensor(event_times_table)
        event_table_pid = tf.convert_to_tensor(event_table_pid)
        num_transitions, num_individuals = tf.shape(event_times_table)

        # Construct the event list for fixed event EI
        fixed_event_list = cls.EventList(
            time=tf.reshape(event_times_table[1, :], -1),
            unit=event_table_pid,
            event=tf.repeat(1, [num_individuals]),
        )

        # Mask out any NaN event times so they don't appear in the event list
        is_not_nan = ~tf.math.is_nan(fixed_event_list.time)
        fixed_event_list = tf.nest.map_structure(
            lambda x: tf.boolean_mask(x, is_not_nan), fixed_event_list
        )

        sort_idx = tf.argsort(fixed_event_list.time)
        fixed_event_list = tf.nest.map_structure(
            lambda x: tf.gather(x, sort_idx), fixed_event_list
        )

        # Construct the event list (SE and IR events)
        event_list = cls.EventList(
            time=tf.reshape(tf.gather(event_times_table, [0, 2], axis=0), -1),
            unit=tf.tile(event_table_pid, [num_transitions - 1]),
            event=tf.repeat([0, 2], [num_individuals]),
        )

        # Mask out any NaN event times so they don't appear in the event list
        is_not_nan = ~tf.math.is_nan(event_list.time)
        event_list = tf.nest.map_structure(
            lambda x: tf.boolean_mask(x, is_not_nan), event_list
        )

        # Sort by time
        sort_idx = tf.argsort(event_list.time)
        event_list = tf.nest.map_structure(
            lambda x: tf.gather(x, sort_idx), event_list
        )
        return fixed_event_list, event_list

    @property
    def transition_rate_fn(self):
        return self._parameters["transition_rate_fn"]

    @property
    def initial_conditions(self):
        return self._parameters["initial_conditions"]

    @property
    def num_individuals(self):
        return self._parameters["num_individuals"]

    @property
    def num_wards(self):
        return self._parameters["num_wards"]

    @property
    def num_states(self):
        return self.incidence_matrix.shape[0]

    @property
    def num_events(self):
        return self.incidence_matrix.shape[1]

    @property
    def covariate_change_times(self):
        return self._parameters["covariate_change_times"]

    @property
    def ids(self):
        return self._parameters["ids"]

    @property
    def incidence_matrix(self):
        return self._incidence_matrix

    @property
    def fixed_event_list(self):
        return self._parameters["fixed_event_list"]

    @property
    def fit_to_hospital_data(self):
        return self._parameters["fit_to_hospital_data"]

    def _concatonate_event_lists(self, event_list):
        # Concatenate the fixed and changed event list with covariate times
        num_covariate_changes = tf.shape(self.covariate_change_times)[0]

        new_event_list = self.EventList(
            time=tf.concat(
                [
                    event_list.time,
                    self.fixed_event_list.time,
                    self.covariate_change_times,
                ],
                axis=0,
            ),
            unit=tf.concat(
                [
                    event_list.unit,
                    self.fixed_event_list.unit,
                    tf.fill(
                        (num_covariate_changes,),
                        tf.constant(-1, event_list.unit.dtype),
                    ),
                ],
                axis=0,
            ),
            event=tf.concat(
                [
                    event_list.event,
                    self.fixed_event_list.event,
                    tf.fill((num_covariate_changes,), self.num_events - 1),
                ],
                axis=0,
            ),
        )

        # Sort by time
        sorted_idx = tf.argsort(new_event_list.time, name="sort_event_list")
        event_list = tf.nest.map_structure(
            lambda x: tf.gather(x, sorted_idx), new_event_list
        )

        # Generate a corresponding sequence of pointers to indices in the
        # first dimension of the covariate structure
        covariate_pointers = tf.cumsum(
            tf.cast(event_list.unit == -1, event_list.unit.dtype)
        )
        # Reset -1s to 0s
        event_list = event_list._replace(
            unit=tf.clip_by_value(
                event_list.unit,
                clip_value_min=0,
                clip_value_max=event_list.unit.dtype.max,
            )
        )

        return event_list, covariate_pointers

    def _log_prob_chunk(
        self, times, event_pids, event_id, covariate_pointers, initial_state
    ):
        with tf.name_scope("log_prob_chunk"):
            # Create dense event tensor
            dense_events = expand_event_list(
                times,
                event_pids,
                event_id,
                self.num_individuals,
                self.num_events,
            )

            # Compute the state tensor
            state = compute_state(
                initial_state, dense_events, self.incidence_matrix
            )

            # Compute time between each event
            times = tf.concat([[times[0]], times], 0)
            time_delta = times[1:] - times[:-1]

            # Compute likelihood
            def llik_t_fn(args):
                with tf.name_scope("llik_t_fn"):
                    (
                        state,
                        covariate_pointers,
                        event_id,
                        event_pids,
                        time_delta,
                    ) = args
                    total_event_rate, event_rate = self.transition_rate_fn(
                        (state, covariate_pointers, event_id, event_pids)
                    )

                    return compute_loglik(
                        (total_event_rate, event_rate, time_delta)
                    )

            loglik_t = tf.vectorized_map(
                llik_t_fn,
                elems=(
                    state,
                    covariate_pointers,
                    event_id,
                    event_pids,
                    time_delta,
                ),
            )
            return state[-1], tf.reduce_sum(loglik_t)

    def log_prob(self, event_list):
        """Return the log probability density of observing `event_list`

        :param event_list: an EventList object
        :returns: the log probability density of observing `event_list`
        """
        with tf.name_scope("log_prob"):
            (
                event_list,
                covariate_pointers,
            ) = self._concatonate_event_lists(event_list)

            # Calculate chunks of likelihood
            initial_state = self.initial_conditions

            def chunk_fn(accum, elems):
                initial_state, log_prob = accum
                times, event_pids, event_id, covariate_pointers = elems
                next_state, log_lik_chunk = self._log_prob_chunk(
                    times,
                    event_pids,
                    event_id,
                    covariate_pointers,
                    initial_state,
                )

                return next_state, log_lik_chunk + log_prob

            _, log_prob = tf.scan(
                chunk_fn,
                elems=[
                    tf.expand_dims(x, 0)  # Use tf.split to chunk up, maybe.
                    for x in (
                        event_list.time,
                        event_list.unit,
                        event_list.event,
                        covariate_pointers,
                    )
                ],
                initializer=(initial_state, 0.0),
            )

            conditions = tf.cond(
                self.fit_to_hospital_data,
                true_fn=lambda: tf.reduce_any(
                    (event_list.time < 0.0) & (event_list.event == 2)
                ),
                false_fn=lambda: tf.reduce_any((event_list.time < 0.0)),
            )
            return tf.cond(
                conditions,
                true_fn=lambda: -np.inf,
                false_fn=lambda: log_prob,
            )  # Reduce to a scalar rather than shape (1,)

    def attributable_fractions(self, event_list):
        with tf.name_scope("compute_attributable_fractions"):
            (
                event_list,
                covariate_pointers,
            ) = self._concatonate_event_lists(event_list)

            # Calculate chunks of likelihood
            initial_state = self.initial_conditions

            # Create dense event tensor
            dense_events = expand_event_list(
                event_list.time,
                event_list.unit,
                event_list.event,
                self.num_individuals,
                self.num_events,
            )

            # Compute the state tensor
            state = compute_state(
                initial_state, dense_events, self.incidence_matrix
            )

            # Compute attributable fractions
            exp_inds = tf.where(
                (event_list.event == 0) & (event_list.time >= 0.0)
            )
            exp_events = tf.gather(event_list.event, exp_inds)
            exp_ids = tf.squeeze(tf.gather(event_list.unit, exp_inds))
            covariate_pointers_exp = tf.squeeze(
                tf.gather(covariate_pointers, exp_inds)
            )
            state_exp = tf.squeeze(tf.gather(state, exp_inds))

            (
                att_fracts_check,
                between_ward_fracts,
                ward_exp,
                between_ward_transmission,
                ward_pr,
            ) = tf.vectorized_map(
                self.transition_rate_fn,
                elems=(state_exp, covariate_pointers_exp, exp_events, exp_ids),
            )
            return (
                att_fracts_check,
                between_ward_fracts,
                ward_exp,
                between_ward_transmission,
                covariate_pointers_exp,
                ward_pr,
            )
