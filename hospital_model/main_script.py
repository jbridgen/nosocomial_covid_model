import hospital_model_cont as hf
import numpy as np
import pandas as pd
from importlib import reload
from datetime import datetime

start_time = datetime.now()

run_inference = True
simulate_from_post = False
analyse_post_output = False

if run_inference:
    samples, results, acceptance_ratios = hf.run_inference(
        load_from_previous_sim=True,
        fit_to_hospital_data=True,
        its=11000,
        burn_in=1000,
        chain="chain_1",
    )

if simulate_from_post:
    (
        post,
        simulated_infection_events,
        fixed_ei_event_list,
    ) = hf.simulate_from_posterior_draws(
        load_from_previous_sim=True,
        chain="chain_1",
    )

if analyse_post_output:
    (
        mean_nosoc_att_fract,
        perc_nosoc_exp_by_ward,
        perc_ward_cols_nosoc,
    ) = hf.analyse_post_output(
        load_from_previous_sim=True,
        chain="chain_1",
    )

print("runtime: ", datetime.now() - start_time)
