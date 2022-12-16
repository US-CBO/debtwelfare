import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

# Load data
cobb_douglas = pd.read_csv(CURRENT_PATH + '/../data/outputs/welfare_effect_summary.csv')
linear = pd.read_csv(CURRENT_PATH + '/../data/outputs/welfare_effect_summary_linear.csv')
blanchard_comparison = pd.read_csv(CURRENT_PATH + '/../data/for_figures/blanchard_comparison.csv')
r_and_g = pd.read_csv(CURRENT_PATH + '/../data/for_figures/r_and_g.csv')

# Put r and R in percent with one decimal place
cobb_douglas['r'] = round(cobb_douglas['riskfree_rate'] / (1 + cobb_douglas['growth_rate']) * 100 - 100, 1)
cobb_douglas['R'] = round(cobb_douglas['risky_rate'] / (1 + cobb_douglas['growth_rate']) * 100 - 100, 1)

linear['r'] = round(linear['riskfree_rate'] / (1 + linear['growth_rate']) * 100 - 100, 1)
linear['R'] = round(linear['risky_rate'] / (1 + linear['growth_rate']) * 100 - 100, 1)

# Set R values for later use
R_values = [1.0, 2.5, 4.0]

# Define functions for repeated figures
def decomposition_panel(figure_name, data_file, parameters):
    if data_file is cobb_douglas:
        data_index = (
            (data_file['rho'] == parameters['rho']) &
            (data_file['chi'] == parameters['chi']) &
            (data_file['growth_rate'] == parameters['growth_rate']) &
            (data_file['with_endowment'] == parameters['with_endowment'])
        )
    elif data_file is linear:
        data_index = (
            (data_file['chi'] == parameters['chi']) &
            (data_file['growth_rate'] == parameters['growth_rate']) &
            (data_file['with_endowment'] == parameters['with_endowment'])
        )
    else:
        raise ValueError('Inappropriate argument value for data_file.')

    fig, axes = plt.subplots(1, len(R_values), sharey=True, figsize=(10, 7))
    axes[0].set_xlabel("Risk Free Rate (percent)")

    for index, R_value in enumerate(R_values):
        axes[index].set_title("R = %.1f Percent" % R_value)
        data_subset = data_file[data_index & (data_file['R'] == R_value)]
        axes[index].plot(data_subset['r'], data_subset['crowd_out_effect'], color='g', linestyle='-.')
        axes[index].plot(data_subset['r'], data_subset['risk_shifting_effect'], color='b', linestyle='--')
        axes[index].plot(data_subset['r'], data_subset['total_welfare_effect'], color='r')

    axes[0].lines[0].set_label('crowd out effect')
    axes[0].lines[1].set_label('risk shifting effect')
    axes[0].lines[2].set_label('total welfare effect')

    fig.legend(loc='upper center', ncol=3)
    fig.savefig(f'{CURRENT_PATH}/../figures/{figure_name}.png')

    return None


def comparison_panel(figure_name, data_file, parameter_sets):
    fig, axes = plt.subplots(1, len(R_values), sharey = True, figsize = (10, 7))
    axes[0].set_xlabel("Risk Free Rate (percent)")
    for index, R_value in enumerate(R_values):
        axes[index].set_title("R = %.1f Percent" % R_value)
        for parameters in parameter_sets:
            data_index = (
                (data_file['rho'] == parameters['rho']) &
                (data_file['chi'] == parameters['chi']) &
                (data_file['growth_rate'] == parameters['growth_rate']) &
                (data_file['with_endowment'] == parameters['with_endowment'])
            )
            data_subset = data_file[data_index & (data_file['R'] == R_value)]
            axes[index].plot(
                data_subset['r'],
                data_subset['total_welfare_effect'],
                color = parameters['color'],
                linestyle = parameters['style']
            )
            if index == 0:
                axes[0].lines[-1].set_label(parameters['label'])

    fig.legend(loc='upper center', ncol=len(parameter_sets))
    fig.savefig(f'{CURRENT_PATH}/../figures/{figure_name}.png')


def three_d_figure(figure_name, column_name, figure_title):
    plt.clf()
    ax = plt.axes(projection="3d")
    ax.invert_xaxis()
    ax.view_init(30, 240)
    ax.set_ylabel("Marginal Product")
    ax.set_xlabel("Risk Free Rate")
    ax.set_xticks([-2.0, -0.5, 1.0])
    ax.set_yticks([1.0, 2.5, 4])
    ax.set_zlim(-3, 3)
    ax.set_title(figure_title)
    ax.bar3d(
        blanchard_comparison['r'],
        blanchard_comparison['R'],
        np.zeros(9),
        0.5,
        0.5,
        blanchard_comparison[column_name]
    )

    plt.savefig(f'{CURRENT_PATH}/../figures/{figure_name}.png')


""" Figure 1: Comparison of r and g """
ax = plt.axes()
ax.set_xlabel("Year")
ax.plot(r_and_g['Year'], r_and_g['r']*100, color = 'r')
ax.plot(r_and_g['Year'], r_and_g['g']*100, color = 'b')
ax.plot(r_and_g['Year'], r_and_g['projected r']*100, color = 'r', linestyle ="--")
ax.plot(r_and_g['Year'], r_and_g['projected g']*100, color = 'b', linestyle ="--")
plt.text(1952, 15, '"g" - nominal \n growth rate of GDP', fontsize = 6)
plt.text(1964, 1.7, '"r" - Average interest rate \n on the U.S Federal Debt', fontsize = 6)
plt.text(2025, 4.5, '"g" - projected', fontsize = 6)
plt.text(2025, 1.5, '"r" - projected', fontsize = 6)
plt.savefig(CURRENT_PATH + '/../figures/Figure1.png')

""" Figures 2a-f: 3-D comparisons with Blanchard """
three_d_figure('Figure2a', "study_cd", "Current Study Estimates With Cobb Douglas Production Function")
three_d_figure('Figure2b', "study_linear", "Current Study Estimates With Linear Production Function")
three_d_figure('Figure2c', "cobb_douglas_effect", "Blanchard Estimates With Cobb Douglas Production Function")
three_d_figure('Figure2d', "linear_effect", "Blanchard Estimates With Linear Production Function")
three_d_figure('Figure2e', "diff_cb", "Difference in Estimates With Cobb Douglas Production Function")
three_d_figure('Figure2f', "diff_linear", "Difference in Estimates With Linear Production Function")


""" Figure 3: Decomposition of welfare effects, CB production with endowment """
decomposition_panel('Figure3', cobb_douglas,
    {"with_endowment": True, "rho": 0.0, "chi": 1, "growth_rate": 0.0}
)

""" Figure 4: Decomposition of welfare effects, linear production with endowment """
decomposition_panel('Figure4', linear,
    {"with_endowment": True, "rho": 0.0, "chi": 1, "growth_rate": 0.0}
)

""" Figure 5: Effects of transfer with and without endowment """
comparison_panel('Figure5', cobb_douglas,
    [{"with_endowment": True, "rho": 0.0, "chi": 1, "growth_rate": 0.0, "color": "g", "style": "-.", "label": "with endowment"},
    {"with_endowment": False, "rho": 0.0, "chi": 1, "growth_rate": 0.0, "color": "b", "style": "-", "label": "without endowment"}]
)

""" Figure 6: Effects of transfer with different levels of aversion to intergenerational risk """
comparison_panel('Figure6', cobb_douglas,
    [{"with_endowment": False, "rho": 0.0, "chi": 1, "growth_rate": 0.0, "color": "b", "style": "-", "label": "without endowment, chi = 1"},
    {"with_endowment": False, "rho": 0.0, "chi": 10, "growth_rate": 0.0, "color": "r", "style": "-.", "label": "without endowment, chi = 10"}]
)

""" Figure 7: Effects of transfer with and without persistence in technology shocks """
comparison_panel('Figure7', cobb_douglas,
    [{"with_endowment": False, "rho": 0.0, "chi": 1, "growth_rate": 0.0, "color": "b", "style": "-", "label": "without endowment, rho = 0.0"},
    {"with_endowment": False, "rho": 0.9, "chi": 1, "growth_rate": 0.0, "color": "r", "style": "-.", "label": "without endowment, rho = 0.9"}]
)

""" Figure 8: Effects of transfer with high intergenerational risk aversion and different levels of persistence in technology shocks """
comparison_panel('Figure8', cobb_douglas,
    [{"with_endowment": False, "rho": 0.0, "chi": 10, "growth_rate": 0.0, "color": "b", "style": "-", "label": "without endowment, chi = 10, rho = 0.0"},
    {"with_endowment": False, "rho": 0.9, "chi": 10, "growth_rate": 0.0, "color": "r", "style": "-.", "label": "without endowment, chi = 10, rho = 0.9"}]
)

""" Figure 9: Effects of transfer with high intergenerational risk aversion and different levels of persistence in technology shocks """
comparison_panel('Figure9', cobb_douglas,
    [{"with_endowment": True, "rho": 0.0, "chi": 1, "growth_rate": 0.0, "color": "g", "style": "-", "label": "with endowment"},
    {"with_endowment": False, "rho": 0.0, "chi": 1, "growth_rate": 0.0, "color": "b", "style": "-.", "label": "without endowment"},
    {"with_endowment": False, "rho": 0.0, "chi": 10, "growth_rate": 0.0, "color": "r", "style": "--", "label": "without endowment, chi = 10, rho = 0.0"},
    {"with_endowment": False, "rho": 0.9, "chi": 10, "growth_rate": 0.0, "color": "y", "style": ":", "label": "without endowment, chi = 10, rho = 0.9"}]
)

""" Figure 10: Decomposition of welfare effects, without endowment, high persistence and intergenerational risk """
decomposition_panel('Figure10', cobb_douglas,
    {"with_endowment": False, "rho": 0.9, "chi": 10, "growth_rate": 0.0}
)

""" Figure 11: Effects of transfer with high intergenerational risk aversion and different levels of persistence in technology shocks """
comparison_panel('Figure11', cobb_douglas,
    [{"with_endowment": False, "rho": 0.0, "chi": 1, "growth_rate": 0.0, "color": "b", "style": "-", "label": "growth rate = 0 percent"},
    {"with_endowment": False, "rho": 0.0, "chi": 1, "growth_rate": 0.01, "color": "r", "style": "--", "label": "growth rate = 1 percent"},
    {"with_endowment": False, "rho": 0.0, "chi": 1, "growth_rate": 0.02, "color": "y", "style": ":", "label": "growth rate = 2 percent"}]
)
