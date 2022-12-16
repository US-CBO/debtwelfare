from datetime import timedelta
import os.path
import time
import pandas as pd
from olg.model import ModelLinear
from olg.utils import read_parameters


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

start_time = time.time()

parameters_linear = read_parameters(CURRENT_PATH + '/../data/inputs/parameters_linear.yml')
model_linear = ModelLinear(parameters_linear)
calibration_targets = pd.read_csv(CURRENT_PATH + '/../data/inputs/calibration_targets_linear.csv')

for _, target in calibration_targets.iterrows():

    model_linear.with_endowment = target['with_endowment']
    model_linear.growth_rate = target['growth_rate']
    model_linear.risky_rate = target['risky_rate'] * (1 + model_linear.growth_rate)
    model_linear.riskfree_rate = target['riskfree_rate'] * (1 + model_linear.growth_rate)

    print('\nSolving model with the following calibration targets:')
    print(f'  With Endowment: {model_linear.with_endowment}')
    print(f'  Growth Rate: {model_linear.growth_rate}')
    print(f'  Risky Rate: {model_linear.risky_rate}')
    print(f'  Risk-free rate: {model_linear.riskfree_rate}')

    model_linear.calibrate()
    model_linear.simulate()
    model_linear.store_results()

    transfer_to_wage_ratio = model_linear.calc_transfer_to_wage_ratio(parameters_linear.transfer_to_capital_ratio)

    model_linear.simulate(transfer_to_wage_ratio, 'fixed')
    model_linear.store_results(transfer_to_wage_ratio, 'fixed')

    model_linear.simulate(transfer_to_wage_ratio, 'variable')
    model_linear.store_results(transfer_to_wage_ratio, 'variable')

summary_with_welfare_effects = model_linear.calc_welfare_effects(model_linear.simulation_results)
output_file = CURRENT_PATH + '/../data/outputs/welfare_effect_summary_linear.csv'
summary_with_welfare_effects.to_csv(output_file, index=False, float_format=f'%.{parameters_linear.output_precision}f')

print('\nModel Complete')
elapsed_seconds = time.time() - start_time
print(f'Model took: {str(timedelta(seconds=elapsed_seconds))}')

