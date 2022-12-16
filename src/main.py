from datetime import timedelta
import os.path
import time
import pandas as pd
from olg.model import Model
from olg.utils import read_parameters


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

start_time = time.time()

parameters = read_parameters(CURRENT_PATH + '/../data/inputs/parameters.yml')
model = Model(parameters)
calibration_targets = pd.read_csv(CURRENT_PATH + '/../data/inputs/calibration_targets.csv')

for _, target in calibration_targets.iterrows():

    model.rho = target['rho']
    model.with_endowment = target['with_endowment']
    model.growth_rate = target['growth_rate']
    model.risky_rate = target['risky_rate'] * (1 + model.growth_rate)
    model.riskfree_rate = target['riskfree_rate'] * (1 + model.growth_rate)

    print('\nSolving model with the following calibration targets:')
    print(f'  Rho: {model.rho}')
    print(f'  With Endowment: {model.with_endowment}')
    print(f'  Growth Rate: {model.growth_rate}')
    print(f'  Risky Rate: {model.risky_rate}')
    print(f'  Risk-free Rate: {model.riskfree_rate}')

    model.calibrate()
    model.store_results()

    transfer_to_wage_ratio = model.calc_transfer_to_wage_ratio(parameters.transfer_to_capital_ratio)

    model.simulate(transfer_to_wage_ratio, 'fixed')
    model.store_results(transfer_to_wage_ratio, 'fixed')

    model.simulate(transfer_to_wage_ratio, 'variable')
    model.store_results(transfer_to_wage_ratio, 'variable')

summary_with_welfare_effects = model.calc_welfare_effects(model.simulation_results)
output_file = CURRENT_PATH + '/../data/outputs/welfare_effect_summary.csv'
summary_with_welfare_effects.to_csv(output_file, index=False, float_format=f'%.{parameters.output_precision}f')

print('\nModel Complete')
elapsed_seconds = time.time() - start_time
print(f'Model took: {str(timedelta(seconds=elapsed_seconds))}')
