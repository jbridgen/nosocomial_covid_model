from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def get_chain_states():
    """Returns initial parameter values for each chain"""
    ChainState = namedtuple(
        "ChainState", ["beta1", "beta2", "beta3", "beta4", "beta5"]
    )
    chain_1 = ChainState(
        beta1=0.03,
        beta2=1e-5,
        beta3=0.00009,
        beta4=1e-13,
        beta5=1e-10,
    )

    tf.random.set_seed(12)
    prior_initial_vals = (
        tfd.Gamma(
            concentration=1.1,
            rate=1000,
        )
        .sample(10)
        .numpy()
    )
    chain_2 = ChainState(
        beta1=prior_initial_vals[0],
        beta2=prior_initial_vals[1],
        beta3=prior_initial_vals[2],
        beta4=prior_initial_vals[3],
        beta5=prior_initial_vals[4],
    )
    chain_3 = ChainState(
        beta1=prior_initial_vals[5],
        beta2=prior_initial_vals[6],
        beta3=prior_initial_vals[7],
        beta4=prior_initial_vals[8],
        beta5=prior_initial_vals[9],
    )

    return {"chain_1": chain_1, "chain_2": chain_2, "chain_3": chain_3}


def log_mh_accept(proposed_log_prob, current_log_prob, log_q_ratio, seed):
    """Decides whether to accept or reject a MH move"""

    acceptance_prob = proposed_log_prob - current_log_prob + log_q_ratio
    u = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0)

    return tf.math.log(u) < acceptance_prob


# Metropolis hastings algorithm for transmission parameters
MHTransmissionParmsResults = namedtuple(
    "MHTransmissionParmsResults",
    [
        "is_accepted",
        "target_log_prob",
        "proposed_state",
        "proposed_target_log_prob",
    ],
)

MHTransmissionEventResults = namedtuple(
    "MHTransmissionEventResults",
    [
        "is_accepted",
        "target_log_prob",
        "proposed_transmission_events",
        "pos_to_update",
        "proposed_target_log_prob",
    ],
)


def mh_transmission_parms(
    target_log_prob_fn,
    proposal_distribution_scale,
    current_state,
    previous_results,
    transmission_events,
    param_name,
    seed,
):
    """Metropolis Hastings algorithm for transmission parameters"""
    current_state_param = getattr(current_state, param_name)

    proposal_seed, accept_seed, next_seed = tfp.random.split_seed(
        seed, n=3, salt="mh_transmission_parms"
    )

    proposed_state = current_state._replace(
        **{
            param_name: tfp.distributions.LogNormal(
                loc=tf.math.log(current_state_param),
                scale=proposal_distribution_scale[param_name],
            ).sample(seed=proposal_seed)
        }
    )

    # Compute proposed likelihood
    proposed_log_prob = target_log_prob_fn(proposed_state, transmission_events)

    # Calculate acceptance probabiliy and update parameter and posterior value
    log_q_ratio = tf.math.log(
        getattr(proposed_state, param_name)
    ) - tf.math.log(current_state_param)
    is_accepted = log_mh_accept(
        proposed_log_prob,
        previous_results,
        log_q_ratio,
        accept_seed,
    )

    new_state, posterior, results = tf.cond(
        is_accepted,
        true_fn=lambda: (
            proposed_state,
            proposed_log_prob,
            MHTransmissionParmsResults(
                is_accepted=True,
                target_log_prob=proposed_log_prob,
                proposed_state=proposed_state,
                proposed_target_log_prob=proposed_log_prob,
            ),
        ),
        false_fn=lambda: (
            current_state,
            previous_results,
            MHTransmissionParmsResults(
                is_accepted=False,
                target_log_prob=previous_results,
                proposed_state=proposed_state,
                proposed_target_log_prob=proposed_log_prob,
            ),
        ),
    )
    return new_state, posterior, results, next_seed


def mh_event_times(
    target_log_prob_fn,
    proposal_distribution_scale,
    current_state,
    previous_results,
    transmission_events,
    pos_to_update,
    seed,
):
    """Metropolis Hastings algorithm for an event time update"""
    proposal_seed, accept_seed, next_seed = tfp.random.split_seed(
        seed, n=3, salt="mh_event_times"
    )

    current_time = tf.gather(transmission_events.time, pos_to_update)

    # Draw a new event time
    proposed_time = tfp.distributions.Normal(
        loc=current_time,
        scale=proposal_distribution_scale["events"],
    ).sample(seed=proposal_seed)

    proposed_transmission_events = transmission_events._replace(
        **{
            "time": tf.tensor_scatter_nd_update(
                tensor=transmission_events.time,
                indices=[pos_to_update],
                updates=proposed_time,
            )
        }
    )

    # Compute proposed likelihood
    proposed_log_prob = target_log_prob_fn(
        current_state, proposed_transmission_events
    )

    # Calculate acceptance probabiliy and update event times and posterior value
    log_q_ratio = 0.0

    is_accepted = log_mh_accept(
        proposed_log_prob,
        previous_results,
        log_q_ratio,
        accept_seed,
    )

    new_transmission_events, posterior = tf.cond(
        is_accepted,
        true_fn=lambda: (proposed_transmission_events, proposed_log_prob),
        false_fn=lambda: (
            transmission_events,
            previous_results,
        ),
    )

    return (
        new_transmission_events,
        posterior,
        next_seed,
        tf.cast(is_accepted, tf.float32),
    )


def event_time_to_update(transmission_events):
    """Returns index position of event time in event_list to change"""
    return tf.random.uniform(
        shape=(1,),
        minval=0,
        maxval=tf.shape(transmission_events.time)[0] - 1,
        dtype=tf.int32,
    )


def event_updates(
    transmission_events,
    target_log_prob_fn,
    proposal_distribution_scale,
    new_state,
    previous_results,
    new_seed,
):
    """Runs mh_event_times function to update an event time"""
    pos_to_update = event_time_to_update(
        transmission_events=transmission_events,
    )

    (
        new_transmission_events,
        new_posterior,
        new_seed,
        is_accepted,
    ) = mh_event_times(
        target_log_prob_fn=target_log_prob_fn,
        proposal_distribution_scale=proposal_distribution_scale,
        current_state=new_state,
        previous_results=previous_results["posterior"],
        transmission_events=transmission_events,
        pos_to_update=pos_to_update,
        seed=new_seed,
    )

    return (new_transmission_events, new_posterior, new_seed, is_accepted)


@tf.function()
def run_mcmc(
    num_samples,
    current_state,
    target_log_prob_fn,
    transmission_events,
    seed=None,
):
    """Updates each transmission parameter and 10% of transmission event times per MCMC iteration"""
    seed = tfp.random.sanitize_seed(seed, salt="run_mcmc")

    proposal_distribution_scale = {
        "beta1": tf.constant(2.95),
        "beta2": tf.constant(2.5),
        "beta3": tf.constant(4.5),
        "beta4": tf.constant(3.22),
        "beta5": tf.constant(0.45),
        "events": tf.constant(12.6),
    }

    def mcmc_one_step(state_results_seed, iteration):
        (
            current_state,
            transmission_events,
            previous_results,
            seed,
        ) = state_results_seed
        mh_results = {
            "beta1": [],
            "beta2": [],
            "beta3": [],
            "beta4": [],
            "beta5": [],
            "posterior": [],
            "events": [],
            "events_acceptance": [],
        }

        seed = tfp.random.sanitize_seed(seed, salt="mcmc_one_step")

        # Run MH for transmission parameters
        param_name = "beta1"  # within ward
        new_state, posterior, results_update, new_seed = mh_transmission_parms(
            target_log_prob_fn=target_log_prob_fn,
            proposal_distribution_scale=proposal_distribution_scale,
            current_state=current_state,
            previous_results=previous_results["posterior"],
            transmission_events=transmission_events,
            param_name=param_name,
            seed=seed,
        )
        mh_results[param_name] = results_update
        mh_results["posterior"] = posterior

        param_name = "beta2"
        new_state, posterior, results_update, new_seed = mh_transmission_parms(
            target_log_prob_fn=target_log_prob_fn,
            proposal_distribution_scale=proposal_distribution_scale,
            current_state=new_state,
            previous_results=mh_results["posterior"],
            transmission_events=transmission_events,
            param_name=param_name,
            seed=new_seed,
        )
        mh_results[param_name] = results_update
        mh_results["posterior"] = posterior

        param_name = "beta3"
        new_state, posterior, results_update, new_seed = mh_transmission_parms(
            target_log_prob_fn=target_log_prob_fn,
            proposal_distribution_scale=proposal_distribution_scale,
            current_state=new_state,
            previous_results=mh_results["posterior"],
            transmission_events=transmission_events,
            param_name=param_name,
            seed=new_seed,
        )
        mh_results[param_name] = results_update
        mh_results["posterior"] = posterior

        param_name = "beta4"
        new_state, posterior, results_update, new_seed = mh_transmission_parms(
            target_log_prob_fn=target_log_prob_fn,
            proposal_distribution_scale=proposal_distribution_scale,
            current_state=new_state,
            previous_results=mh_results["posterior"],
            transmission_events=transmission_events,
            param_name=param_name,
            seed=new_seed,
        )
        mh_results[param_name] = results_update
        mh_results["posterior"] = posterior

        param_name = "beta5"
        new_state, posterior, results_update, new_seed = mh_transmission_parms(
            target_log_prob_fn=target_log_prob_fn,
            proposal_distribution_scale=proposal_distribution_scale,
            current_state=new_state,
            previous_results=mh_results["posterior"],
            transmission_events=transmission_events,
            param_name=param_name,
            seed=new_seed,
        )
        mh_results[param_name] = results_update
        mh_results["posterior"] = posterior

        # Update 10% of event times
        num_changes = tf.cast(
            tf.shape(transmission_events.time)[0] / 10 + 1, tf.int32
        )
        i = tf.constant(0)
        accepted_values = tf.TensorArray(size=num_changes, dtype=tf.float32)

        # While loop condition function
        def n_its(
            i, transmission_events, mh_results, new_seed, accepted_values
        ):
            return tf.math.less(i, num_changes)

        # While loop body function
        def change_event_times(
            i, transmission_events, mh_results, new_seed, accepted_values
        ):
            (
                transmission_events,
                posterior,
                new_seed,
                is_accepted,
            ) = event_updates(
                transmission_events=transmission_events,
                target_log_prob_fn=target_log_prob_fn,
                proposal_distribution_scale=proposal_distribution_scale,
                new_state=new_state,
                previous_results=mh_results,
                new_seed=new_seed,
            )
            mh_results["posterior"] = posterior
            accepted_values = accepted_values.write(i, is_accepted)
            i = i + 1
            return (
                i,
                transmission_events,
                mh_results,
                new_seed,
                accepted_values,
            )

        # While loop to run MH for 10% of event times
        (
            i,
            transmission_events,
            mh_results,
            new_seed,
            accepted_values,
        ) = tf.while_loop(
            cond=n_its,
            body=change_event_times,
            loop_vars=[
                i,
                transmission_events,
                mh_results,
                new_seed,
                accepted_values,
            ],
        )
        mh_results["events"] = transmission_events
        mh_results["events_acceptance"] = tf.reduce_mean(
            accepted_values.stack()
        )
        return new_state, transmission_events, mh_results, new_seed

    # Initial result structure for transmission events
    example_transmission_structure = MHTransmissionParmsResults(
        is_accepted=False,
        target_log_prob=target_log_prob_fn(current_state, transmission_events),
        proposed_state=current_state,
        proposed_target_log_prob=target_log_prob_fn(
            current_state, transmission_events
        ),
    )

    initial_kernel_results = {
        "beta1": example_transmission_structure,
        "beta2": example_transmission_structure,
        "beta3": example_transmission_structure,
        "beta4": example_transmission_structure,
        "beta5": example_transmission_structure,
        "posterior": target_log_prob_fn(current_state, transmission_events),
        "events": transmission_events,
        "events_acceptance": tf.constant(0.0),
    }

    initializer = (
        current_state,
        transmission_events,
        initial_kernel_results,
        seed,
    )

    samples, events, results, _ = tf.scan(
        mcmc_one_step, elems=(tf.range(num_samples)), initializer=initializer
    )

    return samples, events, results


def accepted_ratio(results, param_name, burn_in):
    """Calculate accepted ratio for a parameter"""
    return (
        tf.shape(tf.where(results[param_name].is_accepted[burn_in:] == True))[
            0
        ]
        / tf.shape(results[param_name].is_accepted)[0]
    )
