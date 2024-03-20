from solver_parameters import SolverParameters
from helper_functions import *


# ======================================================================================================================
#  Energy Storage Parameters
# ======================================================================================================================
class SharedEnergyStorageParameters:

    def __init__(self):
        self.budget = 1e6                               # 1 M m.u.
        self.max_capacity = 2.50                        # Max energy capacity (related to space constraints)
        self.min_pe_factor = 0.10                       # Minimum S/E factor (related to the ESS technology)
        self.max_pe_factor = 4.00                       # Maximum S/E factor (related to the ESS technology)
        self.ess_relax_comp = True                      # Charging/Discharging complementarity (relax/use slack variables or not)
        self.ess_relax_soc = False                      # SoC (relax/use slack variables or not)
        self.ess_relax_apparent_power = False           # Charging/Discharging apparent power (relax/use slack variables or not)
        self.ess_relax_day_balance = False              # Daily energy balance (relax/use slack variables or not)
        self.ess_relax_installed_capacity = False       # Daily energy balance (relax/use slack variables or not)
        self.ess_relax_capacity_available = False       # Capacity available (relax/use slack variables or not)
        self.ess_relax_capacity_degradation = False     # Capacity degradation (relax/use slack variables or not)
        self.ess_relax_capacity_relative = False        # Relative capacity (relax/use slack variables or not)
        self.ess_relax_secondary_reserve = False        # Secondary reserve (relax/use slack variables or not)
        self.ess_interface_relax = False                # Expected interface values (relax/use slack variables or not)
        self.slacks_used = False
        self.plot_results = False                       # Plot results
        self.print_results_to_file = False              # Write results to file
        self.verbose = False                            # Verbose -- Bool
        self.solver_params = SolverParameters()         # Solver Parameters

    def read_parameters_from_file(self, filename):
        _read_parameters_from_file(self, filename)


def _read_parameters_from_file(planning_parameters, filename):

    params_data = convert_json_to_dict(read_json_file(filename))

    planning_parameters.budget = float(params_data['budget'])
    planning_parameters.max_capacity = float(params_data['max_capacity'])
    planning_parameters.min_pe_factor = float(params_data['min_pe_factor'])
    planning_parameters.max_pe_factor = float(params_data['max_pe_factor'])
    planning_parameters.ess_relax_comp = bool(params_data['ess_relax_comp'])
    planning_parameters.ess_relax_apparent_power = bool(params_data['ess_relax_apparent_power'])
    planning_parameters.ess_relax_soc = bool(params_data['ess_relax_soc'])
    planning_parameters.ess_relax_day_balance = bool(params_data['ess_relax_day_balance'])
    planning_parameters.ess_relax_installed_capacity = bool(params_data['ess_relax_installed_capacity'])
    planning_parameters.ess_relax_capacity_available = bool(params_data['ess_relax_capacity_available'])
    planning_parameters.ess_relax_capacity_degradation = bool(params_data['ess_relax_capacity_degradation'])
    planning_parameters.ess_relax_capacity_relative = bool(params_data['ess_relax_capacity_relative'])
    planning_parameters.ess_relax_secondary_reserve = bool(params_data['ess_relax_secondary_reserve'])
    planning_parameters.ess_interface_relax = bool(params_data['ess_interface_relax'])
    planning_parameters.print_results_to_file = bool(params_data['print_results_to_file'])
    planning_parameters.solver_params.read_solver_parameters(params_data['solver'])

    if planning_parameters.ess_relax_comp or planning_parameters.ess_relax_apparent_power or\
            planning_parameters.ess_relax_soc or planning_parameters.ess_relax_day_balance or\
            planning_parameters.ess_relax_installed_capacity or planning_parameters.ess_relax_capacity_available or\
            planning_parameters.ess_relax_capacity_degradation or planning_parameters.ess_relax_secondary_reserve or\
            planning_parameters.ess_interface_relax:
        planning_parameters.slacks_used = True
