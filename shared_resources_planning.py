import os
import time
from copy import copy
import pandas as pd
from math import sqrt, isclose
import networkx as nx
import matplotlib.pyplot as plt
import pyomo.opt as po
import pyomo.environ as pe
from network_data import NetworkData
from shared_energy_storage import SharedEnergyStorage
from shared_energy_storage_data import SharedEnergyStorageData
from planning_parameters import PlanningParameters
from helper_functions import *


# ======================================================================================================================
#   Class SHARED RESOURCES PLANNING
# ======================================================================================================================
class SharedResourcesPlanning:

    def __init__(self, data_dir, filename):
        self.name = filename.replace('.json', '')
        self.data_dir = data_dir
        self.filename = filename
        self.market_data_file = str()
        self.results_dir = os.path.join(data_dir, 'Results')
        self.diagrams_dir = os.path.join(data_dir, 'Diagrams')
        self.params_file = str()
        self.years = dict()
        self.days = dict()
        self.num_instants = int()
        self.discount_factor = float()
        self.cost_energy_p = dict()
        self.cost_secondary_reserve = dict()
        self.cost_tertiary_reserve_up = dict()
        self.cost_tertiary_reserve_down = dict()
        self.prob_market_scenarios = list()
        self.distribution_networks = dict()
        self.transmission_network = NetworkData()
        self.shared_ess_data = SharedEnergyStorageData()
        self.active_distribution_network_nodes = list()
        self.params = PlanningParameters()

    def read_planning_problem(self):
        _read_planning_problem(self)

    def read_market_data_from_file(self):
        _read_market_data_from_file(self)

    def read_planning_parameters_from_file(self):
        print(f'[INFO] Reading PLANNING PARAMETERS from file...')
        filename = os.path.join(self.data_dir, self.params_file)
        self.params.read_parameters_from_file(filename)

    def run_operational_planning(self, candidate_solution=dict()):
        print('[INFO] Running OPERATIONAL PLANNING...')
        if not candidate_solution:
            candidate_solution = self.get_initial_candidate_solution()
        return _run_operational_planning(self, candidate_solution)

    def update_admm_consensus_variables(self, tso_model, dso_models, esso_model, consensus_vars, dual_vars, consensus_vars_prev_iter, params):
        _update_admm_consensus_variables(self, tso_model, dso_models, esso_model, consensus_vars, dual_vars, consensus_vars_prev_iter, params)

    def get_initial_candidate_solution(self):
        return _get_initial_candidate_solution(self)

    def plot_diagram(self):
        _plot_networkx_diagram(self)


# ======================================================================================================================
#  OPERATIONAL PLANNING functions
# ======================================================================================================================
def _run_operational_planning(planning_problem, candidate_solution):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    shared_ess_data = planning_problem.shared_ess_data
    admm_parameters = planning_problem.params.admm
    results = {'tso': dict(), 'dso': dict(), 'esso': dict()}

    # ------------------------------------------------------------------------------------------------------------------
    # 0. Initialization

    print('[INFO]\t\t - Initializing...')

    start = time.time()
    from_warm_start = False
    primal_evolution = list()

    # Create ADMM variables
    consensus_vars, dual_vars, consensus_vars_prev_iter = create_admm_variables(planning_problem)

    # Create Operational Planning models
    dso_models = create_distribution_networks_models(distribution_networks, consensus_vars['interface']['pf']['dso'], consensus_vars['ess']['dso'], candidate_solution['total_capacity'])
    update_distribution_models_to_admm(distribution_networks, dso_models, consensus_vars['interface']['pf']['dso'], admm_parameters)

    tso_model = create_transmission_network_model(transmission_network, consensus_vars['interface']['v'], consensus_vars['interface']['pf'], consensus_vars['ess']['tso'], candidate_solution['total_capacity'])
    update_transmission_model_to_admm(transmission_network, tso_model, consensus_vars['interface']['pf'], admm_parameters)

    esso_model = create_shared_energy_storage_model(shared_ess_data, candidate_solution['investment'])
    update_shared_energy_storage_model_to_admm(shared_ess_data, esso_model, admm_parameters)

    # ------------------------------------------------------------------------------------------------------------------
    # ADMM -- Main cycle
    # ------------------------------------------------------------------------------------------------------------------
    convergence, num_iter = False, 1
    for iter in range(admm_parameters.num_max_iters):

        print(f'[INFO]\t - ADMM. Iter {num_iter}...')

        iter_start = time.time()

        # --------------------------------------------------------------------------------------------------------------
        # 2. Solve TSO problem
        results['tso'] = update_transmission_coordination_model_and_solve(transmission_network, tso_model,
                                                                          consensus_vars['interface']['pf']['dso'], dual_vars['pf']['tso'],
                                                                          consensus_vars['ess']['esso'], dual_vars['ess']['tso'],
                                                                          admm_parameters, from_warm_start=from_warm_start)

        # 2.1 Update ADMM CONSENSUS variables
        planning_problem.update_admm_consensus_variables(tso_model, dso_models, esso_model,
                                                         consensus_vars, dual_vars, consensus_vars_prev_iter,
                                                         admm_parameters)


    if not convergence:
        print(f'[WARNING] ADMM did NOT converge in {admm_parameters.num_max_iters} iterations!')
    else:
        print(f'[INFO] \t - ADMM converged in {iter + 1} iterations.')

    end = time.time()
    total_execution_time = end - start
    print('[INFO] \t - Execution time: {:.2f} s'.format(total_execution_time))

    optim_models = {'tso': tso_model, 'dso': dso_models}
    sensitivities = shared_ess_data.get_sensitivities(esso_model)

    return results, optim_models, sensitivities, primal_evolution


def create_admm_variables(planning_problem):

    num_instants = planning_problem.num_instants

    consensus_variables = {
        'interface': {
            'v': dict(),
            'pf': {'tso': dict(), 'dso': dict()}
        },
        'ess': {'tso': dict(), 'dso': dict(), 'esso': dict(), 'capacity': {'s': dict(), 'e': dict()}}
    }

    dual_variables = {
        'pf': {'tso': dict(), 'dso': dict()},
        'ess': {'tso': dict(), 'dso': dict()}
    }

    consensus_variables_prev_iter = {
        'interface': {'pf': {'tso': dict(), 'dso': dict()}},
        'ess': {'tso': dict(), 'dso': dict(), 'esso': dict()}
    }

    for dn in range(len(planning_problem.active_distribution_network_nodes)):

        node_id = planning_problem.active_distribution_network_nodes[dn]

        consensus_variables['interface']['v'][node_id] = dict()
        consensus_variables['interface']['pf']['tso'][node_id] = dict()
        consensus_variables['interface']['pf']['dso'][node_id] = dict()
        consensus_variables['ess']['tso'][node_id] = dict()
        consensus_variables['ess']['dso'][node_id] = dict()
        consensus_variables['ess']['esso'][node_id] = dict()

        dual_variables['pf']['tso'][node_id] = dict()
        dual_variables['pf']['dso'][node_id] = dict()
        dual_variables['ess']['tso'][node_id] = dict()
        dual_variables['ess']['dso'][node_id] = dict()

        consensus_variables_prev_iter['interface']['pf']['tso'][node_id] = dict()
        consensus_variables_prev_iter['interface']['pf']['dso'][node_id] = dict()
        consensus_variables_prev_iter['ess']['tso'][node_id] = dict()
        consensus_variables_prev_iter['ess']['dso'][node_id] = dict()
        consensus_variables_prev_iter['ess']['esso'][node_id] = dict()

        for year in planning_problem.years:

            consensus_variables['interface']['v'][node_id][year] = dict()
            consensus_variables['interface']['pf']['tso'][node_id][year] = dict()
            consensus_variables['interface']['pf']['dso'][node_id][year] = dict()
            consensus_variables['ess']['tso'][node_id][year] = dict()
            consensus_variables['ess']['dso'][node_id][year] = dict()
            consensus_variables['ess']['esso'][node_id][year] = dict()

            dual_variables['pf']['tso'][node_id][year] = dict()
            dual_variables['pf']['dso'][node_id][year] = dict()
            dual_variables['ess']['tso'][node_id][year] = dict()
            dual_variables['ess']['dso'][node_id][year] = dict()

            consensus_variables_prev_iter['interface']['pf']['tso'][node_id][year] = dict()
            consensus_variables_prev_iter['interface']['pf']['dso'][node_id][year] = dict()
            consensus_variables_prev_iter['ess']['tso'][node_id][year] = dict()
            consensus_variables_prev_iter['ess']['dso'][node_id][year] = dict()
            consensus_variables_prev_iter['ess']['esso'][node_id][year] = dict()

            for day in planning_problem.days:

                consensus_variables['interface']['v'][node_id][year][day] = [1.0] * num_instants
                consensus_variables['interface']['pf']['tso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['interface']['pf']['dso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['tso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['dso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables['ess']['esso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}

                dual_variables['pf']['tso'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['pf']['dso'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['tso'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}
                dual_variables['ess']['dso'][node_id][year][day] = {'p': [0.0] * planning_problem.num_instants, 'q': [0.0] * num_instants}

                consensus_variables_prev_iter['interface']['pf']['tso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables_prev_iter['interface']['pf']['dso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables_prev_iter['ess']['tso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables_prev_iter['ess']['dso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}
                consensus_variables_prev_iter['ess']['esso'][node_id][year][day] = {'p': [0.0] * num_instants, 'q': [0.0] * num_instants}

    return consensus_variables, dual_variables, consensus_variables_prev_iter


def create_transmission_network_model(transmission_network, interface_v_vars, interface_pf_vars, sess_vars, candidate_solution):

    # Build model, fix candidate solution, and Run S-MPOPF model
    tso_model = transmission_network.build_model()
    transmission_network.update_model_with_candidate_solution(tso_model, candidate_solution)
    for node_id in transmission_network.active_distribution_network_nodes:
        for year in transmission_network.years:
            for day in transmission_network.days:
                node_idx = transmission_network.network[year][day].get_node_idx(node_id)
                s_base = transmission_network.network[year][day].baseMVA
                for s_m in tso_model[year][day].scenarios_market:
                    for s_o in tso_model[year][day].scenarios_operation:
                        for p in tso_model[year][day].periods:
                            pc = interface_pf_vars['dso'][node_id][year][day]['p'][p] / s_base
                            qc = interface_pf_vars['dso'][node_id][year][day]['q'][p] / s_base
                            tso_model[year][day].pc[node_idx, s_m, s_o, p].fix(pc)
                            tso_model[year][day].qc[node_idx, s_m, s_o, p].fix(qc)
                            if transmission_network.params.fl_reg:
                                tso_model[year][day].flex_p_up[node_idx, s_m, s_o, p].fix(0.0)
                                tso_model[year][day].flex_p_down[node_idx, s_m, s_o, p].fix(0.0)
    transmission_network.optimize(tso_model)

    # Get initial interface PF values
    for year in transmission_network.years:
        for day in transmission_network.days:
            s_base = transmission_network.network[year][day].baseMVA
            for dn in tso_model[year][day].active_distribution_networks:
                node_id = transmission_network.active_distribution_network_nodes[dn]
                for p in tso_model[year][day].periods:
                    v_mag = sqrt(pe.value(tso_model[year][day].expected_interface_vmag_sqr[dn, p]))
                    interface_pf_p = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                    interface_pf_q = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base
                    interface_v_vars[node_id][year][day][p] = v_mag
                    interface_pf_vars['tso'][node_id][year][day]['p'][p] = interface_pf_p
                    interface_pf_vars['tso'][node_id][year][day]['q'][p] = interface_pf_q

    # Get initial Shared ESS values
    for year in transmission_network.years:
        for day in transmission_network.days:
            s_base = transmission_network.network[year][day].baseMVA
            for dn in tso_model[year][day].active_distribution_networks:
                node_id = transmission_network.active_distribution_network_nodes[dn]
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                for p in tso_model[year][day].periods:
                    shared_ess_p = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base
                    shared_ess_q = pe.value(tso_model[year][day].expected_shared_ess_q[shared_ess_idx, p]) * s_base
                    sess_vars[node_id][year][day]['p'][p] = shared_ess_p
                    sess_vars[node_id][year][day]['q'][p] = shared_ess_q

    return tso_model


def create_distribution_networks_models(distribution_networks, interface_vars, sess_vars, candidate_solution):

    dso_models = dict()

    for node_id in distribution_networks:

        distribution_network = distribution_networks[node_id]

        # Build model, fix candidate solution, and Run S-MPOPF model
        dso_model = distribution_network.build_model()
        distribution_network.update_model_with_candidate_solution(dso_model, candidate_solution)
        distribution_network.optimize(dso_model)

        # Get initial interface PF values
        for year in distribution_network.years:
            for day in distribution_network.days:
                s_base = distribution_network.network[year][day].baseMVA
                for p in dso_model[year][day].periods:
                    interface_pf_p = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                    interface_pf_q = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base
                    interface_vars[node_id][year][day]['p'][p] = interface_pf_p
                    interface_vars[node_id][year][day]['q'][p] = interface_pf_q

        # Get initial Shared ESS values
        for year in distribution_network.years:
            for day in distribution_network.days:
                s_base = distribution_network.network[year][day].baseMVA
                for p in dso_model[year][day].periods:
                    p_ess = pe.value(dso_model[year][day].expected_shared_ess_p[p]) * s_base
                    q_ess = pe.value(dso_model[year][day].expected_shared_ess_q[p]) * s_base
                    sess_vars[node_id][year][day]['p'][p] = p_ess
                    sess_vars[node_id][year][day]['q'][p] = q_ess

        dso_models[node_id] = dso_model

    return dso_models


def create_shared_energy_storage_model(shared_ess_data, candidate_solution):

    esso_model = shared_ess_data.build_subproblem()
    shared_ess_data.update_model_with_candidate_solution(esso_model, candidate_solution)
    shared_ess_data.optimize(esso_model)

    return esso_model


def update_transmission_model_to_admm(transmission_network, model, initial_interface_pf, params):

    for year in transmission_network.years:
        for day in transmission_network.days:

            init_of_value = 1.00
            if transmission_network.params.obj_type == OBJ_MIN_COST:
                init_of_value = pe.value(model[year][day].objective)

            s_base = transmission_network.network[year][day].baseMVA

            # Free Pc and Qc at the connection point with distribution networks
            for node_id in transmission_network.active_distribution_network_nodes:
                node_idx = transmission_network.network[year][day].get_node_idx(node_id)
                for s_m in model[year][day].scenarios_market:
                    for s_o in model[year][day].scenarios_operation:
                        for p in model[year][day].periods:
                            model[year][day].pc[node_idx, s_m, s_o, p].fixed = False
                            model[year][day].pc[node_idx, s_m, s_o, p].setub(None)
                            model[year][day].pc[node_idx, s_m, s_o, p].setlb(None)
                            model[year][day].qc[node_idx, s_m, s_o, p].fixed = False
                            model[year][day].qc[node_idx, s_m, s_o, p].setub(None)
                            model[year][day].qc[node_idx, s_m, s_o, p].setlb(None)

            # Add ADMM variables
            model[year][day].rho_pf = pe.Var(domain=pe.NonNegativeReals)
            model[year][day].rho_pf.fix(params.rho['pf'][transmission_network.name])

            model[year][day].p_pf_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)  # Active power - requested by distribution networks
            model[year][day].q_pf_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)  # Reactive power - requested by distribution networks
            model[year][day].dual_pf_p_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)  # Dual variable - active power requested
            model[year][day].dual_pf_q_req = pe.Var(model[year][day].active_distribution_networks, model[year][day].periods, domain=pe.Reals)  # Dual variable - reactive power requested

            model[year][day].rho_ess = pe.Var(domain=pe.NonNegativeReals)
            model[year][day].rho_ess.fix(params.rho['ess'][transmission_network.name])

            model[year][day].p_ess_req = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)  # Shared ESS - Active power requested (DSO/ESSO)
            model[year][day].q_ess_req = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)  # Shared ESS - Reactive power requested (DSO/ESSO)
            model[year][day].dual_ess_p = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)  # Dual variable - Shared ESS active power
            model[year][day].dual_ess_q = pe.Var(model[year][day].shared_energy_storages, model[year][day].periods, domain=pe.Reals)  # Dual variable - Shared ESS reactive power

            # Objective function - augmented Lagrangian
            obj = model[year][day].objective.expr / abs(init_of_value)
            for dn in model[year][day].active_distribution_networks:
                node_id = transmission_network.active_distribution_network_nodes[dn]
                for p in model[year][day].periods:
                    init_p = initial_interface_pf['dso'][node_id][year][day]['p'][p] / s_base
                    init_q = initial_interface_pf['dso'][node_id][year][day]['q'][p] / s_base
                    constraint_p_req = (model[year][day].expected_interface_pf_p[dn, p] - model[year][day].p_pf_req[dn, p]) / abs(init_p)
                    constraint_q_req = (model[year][day].expected_interface_pf_q[dn, p] - model[year][day].q_pf_req[dn, p]) / abs(init_q)
                    obj += model[year][day].dual_pf_p_req[dn, p] * constraint_p_req
                    obj += model[year][day].dual_pf_q_req[dn, p] * constraint_q_req
                    obj += (model[year][day].rho_pf / 2) * constraint_p_req ** 2
                    obj += (model[year][day].rho_pf / 2) * constraint_q_req ** 2

            for e in model[year][day].active_distribution_networks:
                rating = transmission_network.network[year][day].shared_energy_storages[e].s
                if isclose(rating, 0.00, abs_tol=1e-3/s_base):
                    rating = 1.00       # Do not balance residuals
                for p in model[year][day].periods:
                    constraint_ess_p = (model[year][day].expected_shared_ess_p[e, p] - model[year][day].p_ess_req[e, p]) / (2 * rating)
                    constraint_ess_q = (model[year][day].expected_shared_ess_q[e, p] - model[year][day].q_ess_req[e, p]) / (2 * rating)
                    obj += model[year][day].dual_ess_p[e, p] * constraint_ess_p
                    obj += model[year][day].dual_ess_q[e, p] * constraint_ess_q
                    obj += (model[year][day].rho_ess / 2) * constraint_ess_p ** 2
                    obj += (model[year][day].rho_ess / 2) * constraint_ess_q ** 2

            model[year][day].objective.expr = obj


def update_distribution_models_to_admm(distribution_networks, models, initial_interface_pf, params):

    for node_id in distribution_networks:

        dso_model = models[node_id]
        distribution_network = distribution_networks[node_id]

        # Free voltage at the connection point with the transmission network
        # Free Pg and Qg at the connection point with the transmission network
        for year in distribution_network.years:
            for day in distribution_network.days:

                s_base = distribution_network.network[year][day].baseMVA

                init_of_value = 1.00
                if distribution_network.params.obj_type == OBJ_MIN_COST:
                    init_of_value = pe.value(dso_model[year][day].objective)

                rating = distribution_network.network[year][day].shared_energy_storages[0].s
                if isclose(rating, 0.00, abs_tol=1e-3 / s_base):  # Do not balance residuals
                    rating = 1.00

                ref_node_id = distribution_network.network[year][day].get_reference_node_id()
                ref_node_idx = distribution_network.network[year][day].get_node_idx(ref_node_id)
                ref_gen_idx = distribution_network.network[year][day].get_reference_gen_idx()
                for s_m in dso_model[year][day].scenarios_market:
                    for s_o in dso_model[year][day].scenarios_operation:
                        for p in dso_model[year][day].periods:
                            dso_model[year][day].e[ref_node_idx, s_m, s_o, p].fixed = False
                            dso_model[year][day].e[ref_node_idx, s_m, s_o, p].setub(None)
                            dso_model[year][day].e[ref_node_idx, s_m, s_o, p].setlb(None)

                            dso_model[year][day].pg[ref_gen_idx, s_m, s_o, p].fixed = False
                            dso_model[year][day].pg[ref_gen_idx, s_m, s_o, p].setub(None)
                            dso_model[year][day].pg[ref_gen_idx, s_m, s_o, p].setlb(None)
                            dso_model[year][day].qg[ref_gen_idx, s_m, s_o, p].fixed = False
                            dso_model[year][day].qg[ref_gen_idx, s_m, s_o, p].setub(None)
                            dso_model[year][day].qg[ref_gen_idx, s_m, s_o, p].setlb(None)

                # Add ADMM variables
                dso_model[year][day].rho_pf = pe.Var(domain=pe.NonNegativeReals)
                dso_model[year][day].rho_pf.fix(params.rho['pf'][distribution_network.network[year][day].name])

                dso_model[year][day].p_pf_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)    # Active power - requested by transmission network
                dso_model[year][day].q_pf_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)    # Reactive power - requested by transmission network
                dso_model[year][day].dual_pf_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals)   # Dual variable - active power
                dso_model[year][day].dual_pf_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals)   # Dual variable - reactive power

                dso_model[year][day].rho_ess = pe.Var(domain=pe.NonNegativeReals)
                dso_model[year][day].rho_ess.fix(params.rho['ess'][distribution_network.network[year][day].name])

                dso_model[year][day].p_ess_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)  # Shared ESS - active power requested (TSO/ESSO)
                dso_model[year][day].q_ess_req = pe.Var(dso_model[year][day].periods, domain=pe.Reals)  # Shared ESS - reactive power requested (TSO/ESSO)
                dso_model[year][day].dual_ess_p = pe.Var(dso_model[year][day].periods, domain=pe.Reals)  # Dual variable - Shared ESS active power
                dso_model[year][day].dual_ess_q = pe.Var(dso_model[year][day].periods, domain=pe.Reals)  # Dual variable - Shared ESS reactive power

                # Objective function - augmented Lagrangian
                obj = dso_model[year][day].objective.expr / max(abs(init_of_value), 1.00)

                # Augmented Lagrangian -- Interface power flow (residual balancing)
                for p in dso_model[year][day].periods:
                    init_p = abs(initial_interface_pf[node_id][year][day]['p'][p]) / s_base
                    init_q = abs(initial_interface_pf[node_id][year][day]['q'][p]) / s_base
                    constraint_p_req = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_req[p]) / abs(init_p)
                    constraint_q_req = (dso_model[year][day].expected_interface_pf_q[p] - dso_model[year][day].q_pf_req[p]) / abs(init_q)
                    obj += (dso_model[year][day].dual_pf_p[p]) * (constraint_p_req)
                    obj += (dso_model[year][day].dual_pf_q[p]) * (constraint_q_req)
                    obj += (dso_model[year][day].rho_pf / 2) * (constraint_p_req) ** 2
                    obj += (dso_model[year][day].rho_pf / 2) * (constraint_q_req) ** 2

                # Augmented Lagrangian -- Shared ESS (residual balancing)
                for p in dso_model[year][day].periods:
                    constraint_ess_p = (dso_model[year][day].expected_shared_ess_p[p] - dso_model[year][day].p_ess_req[p]) / (2 * rating)
                    constraint_ess_q = (dso_model[year][day].expected_shared_ess_q[p] - dso_model[year][day].q_ess_req[p]) / (2 * rating)
                    obj += dso_model[year][day].dual_ess_p[p] * (constraint_ess_p)
                    obj += dso_model[year][day].dual_ess_q[p] * (constraint_ess_q)
                    obj += (dso_model[year][day].rho_ess / 2) * (constraint_ess_p) ** 2
                    obj += (dso_model[year][day].rho_ess / 2) * (constraint_ess_q) ** 2

                dso_model[year][day].objective.expr = obj


def update_shared_energy_storage_model_to_admm(shared_ess_data, model, params):

    repr_years = [year for year in shared_ess_data.years]

    # Add ADMM variables
    model.rho = pe.Var(domain=pe.NonNegativeReals)
    model.rho.fix(params.rho['ess']['esso'])

    # Active and Reactive power requested by TSO and DSOs
    model.p_req_transm = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)   # Active power - transmission network
    model.q_req_transm = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)   # Active power - transmission network
    model.p_req_distr = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)    # Reactive power - distribution networks
    model.q_req_distr = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)    # Reactive power - distribution networks
    model.dual_p_transm = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)  # Dual variable - active power - transmission network
    model.dual_q_transm = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)  # Dual variable - reactive power - transmission network
    model.dual_p_distr = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)   # Dual variable - active power - distribution networks
    model.dual_q_distr = pe.Var(model.energy_storages, model.years, model.days, model.periods, domain=pe.Reals)   # Dual variable - reactive power - distribution networks

    # Objective function - augmented Lagrangian
    init_of_value = pe.value(model.objective)
    obj = model.objective.expr / abs(init_of_value)
    for e in model.energy_storages:
        for y in model.years:
            year = repr_years[y]
            rating_s = shared_ess_data.shared_energy_storages[year][e].s
            if rating_s == 0.0:
                rating_s = 1.00     # Do not balance residuals
            for d in model.days:
                for p in model.periods:
                    p_ess = model.es_expected_p[e, y, d, p]
                    q_ess = model.es_expected_q[e, y, d, p]
                    constraint_p_transm = (p_ess - model.p_req_transm[e, y, d, p]) / (2 * rating_s)
                    constraint_q_transm = (q_ess - model.q_req_transm[e, y, d, p]) / (2 * rating_s)
                    constraint_p_distr = (p_ess - model.p_req_distr[e, y, d, p]) / (2 * rating_s)
                    constraint_q_distr = (q_ess - model.q_req_distr[e, y, d, p]) / (2 * rating_s)
                    obj += model.dual_p_transm[e, y, d, p] * (constraint_p_transm)
                    obj += model.dual_q_transm[e, y, d, p] * (constraint_q_transm)
                    obj += model.dual_p_distr[e, y, d, p] * (constraint_p_distr)
                    obj += model.dual_q_distr[e, y, d, p] * (constraint_q_distr)
                    obj += (model.rho / 2) * (constraint_p_transm) ** 2
                    obj += (model.rho / 2) * (constraint_q_transm) ** 2
                    obj += (model.rho / 2) * (constraint_p_distr) ** 2
                    obj += (model.rho / 2) * (constraint_q_distr) ** 2

    model.objective.expr = obj

    return model


def update_transmission_coordination_model_and_solve(transmission_network, model, pf_req, dual_pf, ess_req, dual_ess, params, from_warm_start=False):

    print('[INFO] \t\t - Updating transmission network...')

    for year in transmission_network.years:
        for day in transmission_network.days:

            s_base = transmission_network.network[year][day].baseMVA

            rho_pf = params.rho['pf'][transmission_network.name]
            rho_ess = params.rho['ess'][transmission_network.name]
            if params.adaptive_penalty:
                rho_pf = pe.value(model[year][day].rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)
                rho_ess = pe.value(model[year][day].rho_pf) * (1 + ADMM_ADAPTIVE_PENALTY_FACTOR)

            # Update Rho parameter
            model[year][day].rho_pf.fix(rho_pf)
            model[year][day].rho_ess.fix(rho_ess)

            for dn in model[year][day].active_distribution_networks:

                node_id = transmission_network.active_distribution_network_nodes[dn]

                # Update interface PF power requests
                for p in model[year][day].periods:
                    model[year][day].dual_pf_p_req[dn, p].fix(dual_pf[node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_pf_q_req[dn, p].fix(dual_pf[node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_pf_req[dn, p].fix(pf_req[node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_pf_req[dn, p].fix(pf_req[node_id][year][day]['q'][p] / s_base)

                # Update shared ESS capacity and power requests
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                for p in model[year][day].periods:
                    model[year][day].dual_ess_p[shared_ess_idx, p].fix(dual_ess[node_id][year][day]['p'][p] / s_base)
                    model[year][day].dual_ess_q[shared_ess_idx, p].fix(dual_ess[node_id][year][day]['q'][p] / s_base)
                    model[year][day].p_ess_req[shared_ess_idx, p].fix(ess_req[node_id][year][day]['p'][p] / s_base)
                    model[year][day].q_ess_req[shared_ess_idx, p].fix(ess_req[node_id][year][day]['q'][p] / s_base)

    # Solve!
    res = transmission_network.optimize(model, from_warm_start=from_warm_start)
    for year in transmission_network.years:
        for day in transmission_network.days:
            if res[year][day].solver.status == po.SolverStatus.error:
                print(f'[ERROR] Network {model[year][day].name} did not converge!')
                # exit(ERROR_NETWORK_OPTIMIZATION)
    return res


def _update_admm_consensus_variables(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, consensus_vars_prev_iter, params):
    _update_previous_consensus_variables(planning_problem, consensus_vars, consensus_vars_prev_iter)
    _update_interface_power_flow_variables(planning_problem, tso_model, dso_models, consensus_vars['interface'], dual_vars['pf'], params)
    _update_shared_energy_storage_variables(planning_problem, tso_model, dso_models, esso_model, consensus_vars['ess'], dual_vars['ess'], params)


def _update_previous_consensus_variables(planning_problem, consensus_vars, consensus_vars_prev_iter):
    for dn in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[dn]
        for year in planning_problem.years:
            for day in planning_problem.days:
                for p in range(planning_problem.num_instants):
                    consensus_vars_prev_iter['interface']['pf']['tso'][node_id][year][day]['p'][p] = copy(consensus_vars['interface']['pf']['tso'][node_id][year][day]['p'][p])
                    consensus_vars_prev_iter['interface']['pf']['tso'][node_id][year][day]['q'][p] = copy(consensus_vars['interface']['pf']['tso'][node_id][year][day]['q'][p])
                    consensus_vars_prev_iter['interface']['pf']['dso'][node_id][year][day]['p'][p] = copy(consensus_vars['interface']['pf']['dso'][node_id][year][day]['p'][p])
                    consensus_vars_prev_iter['interface']['pf']['dso'][node_id][year][day]['q'][p] = copy(consensus_vars['interface']['pf']['dso'][node_id][year][day]['q'][p])
                    consensus_vars_prev_iter['ess']['tso'][node_id][year][day]['p'][p] = copy(consensus_vars['ess']['tso'][node_id][year][day]['p'][p])
                    consensus_vars_prev_iter['ess']['dso'][node_id][year][day]['p'][p] = copy(consensus_vars['ess']['dso'][node_id][year][day]['p'][p])
                    consensus_vars_prev_iter['ess']['esso'][node_id][year][day]['p'][p] = copy(consensus_vars['ess']['esso'][node_id][year][day]['p'][p])


def _update_interface_power_flow_variables(planning_problem, tso_model, dso_models, interface_vars, dual_vars, params):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks

    # Transmission network - Update Vmag and PF at the TN-DN interface
    for dn in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[dn]
        for year in planning_problem.years:
            for day in planning_problem.days:
                s_base = planning_problem.transmission_network.network[year][day].baseMVA
                for p in tso_model[year][day].periods:
                    interface_vars['v'][node_id][year][day][p] = sqrt(pe.value(tso_model[year][day].expected_interface_vmag_sqr[dn, p]))
                    interface_vars['pf']['tso'][node_id][year][day]['p'][p] = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base
                    interface_vars['pf']['tso'][node_id][year][day]['q'][p] = pe.value(tso_model[year][day].expected_interface_pf_q[dn, p]) * s_base

    # Distribution Network - Update PF at the TN-DN interface
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        dso_model = dso_models[node_id]
        for year in planning_problem.years:
            for day in planning_problem.days:
                s_base = distribution_network.network[year][day].baseMVA
                for p in dso_model[year][day].periods:
                    interface_vars['pf']['dso'][node_id][year][day]['p'][p] = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base
                    interface_vars['pf']['dso'][node_id][year][day]['q'][p] = pe.value(dso_model[year][day].expected_interface_pf_q[p]) * s_base

    # Update Lambdas
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        for year in planning_problem.years:
            for day in planning_problem.days:
                for p in range(planning_problem.num_instants):

                    error_p_pf = interface_vars['pf']['tso'][node_id][year][day]['p'][p] - interface_vars['pf']['dso'][node_id][year][day]['p'][p]
                    error_q_pf = interface_vars['pf']['tso'][node_id][year][day]['q'][p] - interface_vars['pf']['dso'][node_id][year][day]['q'][p]

                    dual_vars['tso'][node_id][year][day]['p'][p] += params.rho[transmission_network.name] * (error_p_pf)
                    dual_vars['tso'][node_id][year][day]['q'][p] += params.rho[transmission_network.name] * (error_q_pf)
                    dual_vars['dso'][node_id][year][day]['p'][p] += params.rho[distribution_network.name] * (-error_p_pf)
                    dual_vars['dso'][node_id][year][day]['q'][p] += params.rho[distribution_network.name] * (-error_q_pf)

                '''
                print(f"Ptso[{node_id},{year},{day}] = {interface_vars['pf']['tso'][node_id][year][day]['p']}")
                print(f"Pdso[{node_id},{year},{day}] = {interface_vars['pf']['dso'][node_id][year][day]['p']}")
                print(f"Qtso[{node_id},{year},{day}] = {interface_vars['pf']['tso'][node_id][year][day]['q']}")
                print(f"Qdso[{node_id},{year},{day}] = {interface_vars['pf']['dso'][node_id][year][day]['q']}")
                '''


def _update_shared_energy_storage_variables(planning_problem, tso_model, dso_models, sess_model, shared_ess_vars, dual_vars, params):

    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    shared_ess_data = planning_problem.shared_ess_data
    repr_days = [day for day in planning_problem.days]
    repr_years = [year for year in planning_problem.years]

    for node_id in distribution_networks:

        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]

        # Shared Energy Storage - Power requested by ESSO, and Capacity available
        for y in sess_model.years:
            year = repr_years[y]
            shared_ess_idx = shared_ess_data.get_shared_energy_storage_idx(node_id)
            for d in sess_model.days:
                day = repr_days[d]
                shared_ess_vars['esso'][node_id][year][day]['p'] = [0.0 for _ in range(planning_problem.num_instants)]
                for p in sess_model.periods:
                    shared_ess_vars['esso'][node_id][year][day]['p'][p] = pe.value(sess_model.es_expected_p[shared_ess_idx, y, d, p])

        # Shared Energy Storage - Power requested by TSO
        for y in range(len(planning_problem.years)):
            year = repr_years[y]
            for d in range(len(repr_days)):
                day = repr_days[d]
                s_base = transmission_network.network[year][day].baseMVA
                shared_ess_idx = transmission_network.network[year][day].get_shared_energy_storage_idx(node_id)
                shared_ess_vars['tso'][node_id][year][day]['p'] = [0.0 for _ in range(planning_problem.num_instants)]
                for p in tso_model[year][day].periods:
                    shared_ess_vars['tso'][node_id][year][day]['p'][p] = pe.value(tso_model[year][day].expected_shared_ess_p[shared_ess_idx, p]) * s_base

        # Shared Energy Storage - Power requested by DSO
        for y in range(len(planning_problem.years)):
            year = repr_years[y]
            for d in range(len(repr_days)):
                day = repr_days[d]
                s_base = distribution_network.network[year][day].baseMVA
                shared_ess_vars['dso'][node_id][year][day]['p'] = [0.0 for _ in range(planning_problem.num_instants)]
                for p in dso_model[year][day].periods:
                    shared_ess_vars['dso'][node_id][year][day]['p'][p] = pe.value(dso_model[year][day].expected_shared_ess_p[p]) * s_base

        '''
        for year in planning_problem.years:
            for day in planning_problem.days:
                print(f"Preq, TN, Node {node_id}, {year}, {day} = {shared_ess_vars['tso'][node_id][year][day]['p']}")
                print(f"Preq, DN, Node {node_id}, {year}, {day} = {shared_ess_vars['dso'][node_id][year][day]['p']}")
                print(f"Preq, ESS, Node {node_id}, {year}, {day} = {shared_ess_vars['esso'][node_id][year][day]['p']}")
        '''

        # Update dual variables Shared ESS
        for year in planning_problem.years:
            for day in planning_problem.days:
                for t in range(planning_problem.num_instants):
                    error_p_ess_transm = shared_ess_vars['tso'][node_id][year][day]['p'][t] - shared_ess_vars['esso'][node_id][year][day]['p'][t]
                    error_p_ess_distr = shared_ess_vars['dso'][node_id][year][day]['p'][t] - shared_ess_vars['esso'][node_id][year][day]['p'][t]
                    dual_vars['tso'][node_id][year][day]['p'][t] += params.rho['ESSO'] * (error_p_ess_transm)
                    dual_vars['dso'][node_id][year][day]['p'][t] += params.rho['ESSO'] * (error_p_ess_distr)


# ======================================================================================================================
#  PLANNING PROBLEM read functions
# ======================================================================================================================
def _read_planning_problem(planning_problem):

    # Create results folder
    if not os.path.exists(planning_problem.results_dir):
        os.makedirs(planning_problem.results_dir)

    # Create diagrams folder
    if not os.path.exists(planning_problem.diagrams_dir):
        os.makedirs(planning_problem.diagrams_dir)

    # Read specification file
    filename = os.path.join(planning_problem.data_dir, planning_problem.filename)
    planning_data = convert_json_to_dict(read_json_file(filename))

    # General Parameters
    for year in planning_data['Years']:
        planning_problem.years[int(year)] = planning_data['Years'][year]
    planning_problem.days = planning_data['Days']
    planning_problem.num_instants = planning_data['NumInstants']

    # Market Data
    planning_problem.discount_factor = planning_data['DiscountFactor']
    planning_problem.market_data_file = planning_data['MarketData']
    planning_problem.read_market_data_from_file()

    # Distribution Networks
    for distribution_network in planning_data['DistributionNetworks']:

        print('[INFO] Reading DISTRIBUTION NETWORK DATA from file(s)...')

        network_name = distribution_network['name']                         # Network filename
        params_file = distribution_network['params_file']                   # Params filename
        connection_nodeid = distribution_network['connection_node_id']      # Connection node ID

        distribution_network = NetworkData()
        distribution_network.name = network_name
        distribution_network.is_transmission = False
        distribution_network.data_dir = planning_problem.data_dir
        distribution_network.results_dir = planning_problem.results_dir
        distribution_network.diagrams_dir = planning_problem.diagrams_dir
        distribution_network.years = planning_problem.years
        distribution_network.days = planning_problem.days
        distribution_network.num_instants = planning_problem.num_instants
        distribution_network.discount_factor = planning_problem.discount_factor
        distribution_network.prob_market_scenarios = planning_problem.prob_market_scenarios
        distribution_network.cost_energy_p = planning_problem.cost_energy_p
        distribution_network.params_file = params_file
        distribution_network.read_network_parameters()
        if distribution_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
            distribution_network.prob_market_scenarios = [1.00]
        distribution_network.read_network_data()
        distribution_network.tn_connection_nodeid = connection_nodeid
        planning_problem.distribution_networks[connection_nodeid] = distribution_network
    planning_problem.active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]

    # Transmission Network
    print('[INFO] Reading TRANSMISSION NETWORK DATA from file(s)...')
    transmission_network = NetworkData()
    transmission_network.name = planning_data['TransmissionNetwork']['name']
    transmission_network.is_transmission = True
    transmission_network.data_dir = planning_problem.data_dir
    transmission_network.results_dir = planning_problem.results_dir
    transmission_network.diagrams_dir = planning_problem.diagrams_dir
    transmission_network.years = planning_problem.years
    transmission_network.days = planning_problem.days
    transmission_network.num_instants = planning_problem.num_instants
    transmission_network.discount_factor = planning_problem.discount_factor
    transmission_network.prob_market_scenarios = planning_problem.prob_market_scenarios
    transmission_network.cost_energy_p = planning_problem.cost_energy_p
    transmission_network.params_file = planning_data['TransmissionNetwork']['params_file']
    transmission_network.read_network_parameters()
    if transmission_network.params.obj_type == OBJ_CONGESTION_MANAGEMENT:
        transmission_network.prob_market_scenarios = [1.00]
    transmission_network.read_network_data()
    transmission_network.active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]
    for year in transmission_network.years:
        for day in transmission_network.days:
            transmission_network.network[year][day].active_distribution_network_nodes = transmission_network.active_distribution_network_nodes
    planning_problem.transmission_network = transmission_network

    # Shared ESS
    shared_ess_data = SharedEnergyStorageData()
    shared_ess_data.name = planning_problem.name
    shared_ess_data.data_dir = planning_problem.data_dir
    shared_ess_data.results_dir = planning_problem.results_dir
    shared_ess_data.years = planning_problem.years
    shared_ess_data.days = planning_problem.days
    shared_ess_data.num_instants = planning_problem.num_instants
    shared_ess_data.discount_factor = planning_problem.discount_factor
    shared_ess_data.prob_market_scenarios = planning_problem.prob_market_scenarios
    shared_ess_data.cost_energy_p = planning_problem.cost_energy_p
    shared_ess_data.cost_secondary_reserve = planning_problem.cost_secondary_reserve
    shared_ess_data.cost_tertiary_reserve_up = planning_problem.cost_tertiary_reserve_up
    shared_ess_data.cost_tertiary_reserve_down = planning_problem.cost_tertiary_reserve_down
    shared_ess_data.params_file = planning_data['SharedEnegyStorage']['params_file']
    shared_ess_data.read_parameters_from_file()
    shared_ess_data.create_shared_energy_storages(planning_problem)
    shared_ess_data.data_file = planning_data['SharedEnegyStorage']['data_file']
    shared_ess_data.read_shared_energy_storage_data_from_file()
    shared_ess_data.active_distribution_network_nodes = [node_id for node_id in planning_problem.distribution_networks]
    planning_problem.shared_ess_data = shared_ess_data

    # Planning Parameters
    planning_problem.params_file = planning_data['PlanningParameters']['params_file']
    planning_problem.read_planning_parameters_from_file()

    # Add Shared Energy Storages to Transmission and Distribution Networks
    _add_shared_energy_storage_to_transmission_network(planning_problem)
    _add_shared_energy_storage_to_distribution_network(planning_problem)


# ======================================================================================================================
#  MARKET DATA read functions
# ======================================================================================================================
def _read_market_data_from_file(planning_problem):

    try:
        for year in planning_problem.years:
            filename = os.path.join(planning_problem.data_dir, 'Market Data', f'{planning_problem.market_data_file}_{year}.xlsx')
            num_scenarios, prob_scenarios = _get_market_scenarios_info_from_excel_file(filename, 'Scenarios')
            planning_problem.prob_market_scenarios = prob_scenarios
            planning_problem.cost_energy_p[year] = dict()
            planning_problem.cost_secondary_reserve[year] = dict()
            planning_problem.cost_tertiary_reserve_up[year] = dict()
            planning_problem.cost_tertiary_reserve_down[year] = dict()
            for day in planning_problem.days:
                planning_problem.cost_energy_p[year][day] = _get_market_costs_from_excel_file(filename, f'Cp, {day}', num_scenarios)
                planning_problem.cost_secondary_reserve[year][day] = _get_market_costs_from_excel_file(filename, f'Csr, {day}', num_scenarios)
                planning_problem.cost_tertiary_reserve_up[year][day] = _get_market_costs_from_excel_file(filename, f'Ctr_up, {day}', num_scenarios)
                planning_problem.cost_tertiary_reserve_down[year][day] = _get_market_costs_from_excel_file(filename, f'Ctr_down, {day}', num_scenarios)
    except:
        print(f'[ERROR] Reading market data from file(s). Exiting...')
        exit(ERROR_SPECIFICATION_FILE)


def _get_market_scenarios_info_from_excel_file(filename, sheet_name):

    num_scenarios = 0
    prob_scenarios = list()

    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        if is_int(df.iloc[0, 1]):
            num_scenarios = int(df.iloc[0, 1])
        for i in range(num_scenarios):
            if is_number(df.iloc[0, i + 2]):
                prob_scenarios.append(float(df.iloc[0, i + 2]))
    except:
        print('[ERROR] Workbook {}. Sheet {} does not exist.'.format(filename, sheet_name))
        exit(1)

    if num_scenarios != len(prob_scenarios):
        print('[WARNING] EnergyStorage file. Number of scenarios different from the probability vector!')

    if round(sum(prob_scenarios), 2) != 1.00:
        print('[ERROR] Probability of scenarios does not add up to 100%. Check file {}. Exiting.'.format(filename))
        exit(ERROR_MARKET_DATA_FILE)

    return num_scenarios, prob_scenarios


def _get_market_costs_from_excel_file(filename, sheet_name, num_scenarios):
    data = pd.read_excel(filename, sheet_name=sheet_name)
    _, num_cols = data.shape
    cost_values = dict()
    scn_idx = 0
    for i in range(num_scenarios):
        cost_values_scenario = list()
        for j in range(num_cols - 1):
            cost_values_scenario.append(float(data.iloc[i, j + 1]))
        cost_values[scn_idx] = cost_values_scenario
        scn_idx = scn_idx + 1
    return cost_values


# ======================================================================================================================
#   NETWORK diagram functions (plot)
# ======================================================================================================================
def _plot_networkx_diagram(planning_problem):

    for year in planning_problem.years:
        for day in planning_problem.days:

            transmission_network = planning_problem.transmission_network.network[year][day]

            node_labels = {}
            ref_nodes, pv_nodes, pq_nodes = [], [], []
            res_pv_nodes = [gen.bus for gen in transmission_network.generators if gen.gen_type == GEN_RES_SOLAR]
            res_wind_nodes = [gen.bus for gen in transmission_network.generators if gen.gen_type == GEN_RES_WIND]
            adn_nodes = planning_problem.active_distribution_network_nodes

            branches = []
            line_list, open_line_list = [], []
            transf_list, open_transf_list = [], []
            for branch in transmission_network.branches:
                if branch.is_transformer:
                    branches.append({'type': 'transformer', 'data': branch})
                else:
                    branches.append({'type': 'line', 'data': branch})

            # Build graph
            graph = nx.Graph()
            for i in range(len(transmission_network.nodes)):
                node = transmission_network.nodes[i]
                graph.add_node(node.bus_i)
                node_labels[node.bus_i] = '{}'.format(node.bus_i)
                if node.type == BUS_REF:
                    ref_nodes.append(node.bus_i)
                elif node.type == BUS_PV:
                    pv_nodes.append(node.bus_i)
                elif node.type == BUS_PQ:
                    if node.bus_i not in (res_pv_nodes + res_wind_nodes + adn_nodes):
                        pq_nodes.append(node.bus_i)
            for i in range(len(branches)):
                branch = branches[i]
                if branch['type'] == 'line':
                    graph.add_edge(branch['data'].fbus, branch['data'].tbus)
                    if branch['data'].status == 1:
                        line_list.append((branch['data'].fbus, branch['data'].tbus))
                    else:
                        open_line_list.append((branch['data'].fbus, branch['data'].tbus))
                if branch['type'] == 'transformer':
                    graph.add_edge(branch['data'].fbus, branch['data'].tbus)
                    if branch['data'].status == 1:
                        transf_list.append((branch['data'].fbus, branch['data'].tbus))
                    else:
                        open_transf_list.append((branch['data'].fbus, branch['data'].tbus))

            # Plot diagram
            pos = nx.spring_layout(graph, k=0.50, iterations=1000)
            fig, ax = plt.subplots(figsize=(12, 8))
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=ref_nodes, node_color='red', node_size=250, label='Reference bus')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=pv_nodes, node_color='lightgreen', node_size=250, label='Conventional generator')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=pq_nodes, node_color='lightblue', node_size=250, label='PQ buses')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=res_pv_nodes, node_color='yellow', node_size=250, label='RES, PV')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=res_wind_nodes, node_color='blue', node_size=250, label='RES, Wind')
            nx.draw_networkx_nodes(graph, ax=ax, pos=pos, nodelist=adn_nodes, node_color='orange', node_size=250, label='ADN buses')
            nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=node_labels, font_size=12)
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=line_list, width=1.50, edge_color='black')
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=transf_list, width=2.00, edge_color='blue', label='Transformer')
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_line_list, style='dashed', width=1.50, edge_color='red')
            nx.draw_networkx_edges(graph, ax=ax, pos=pos, edgelist=open_transf_list, style='dashed', width=2.00, edge_color='red')
            plt.legend(scatterpoints=1, frameon=False, prop={'size': 12})
            plt.axis('off')

            filename = os.path.join(planning_problem.diagrams_dir, f'{planning_problem.name}_{year}_{day}')
            plt.savefig(f'{filename}.pdf', bbox_inches='tight')
            plt.savefig(f'{filename}.png', bbox_inches='tight')


# ======================================================================================================================
#   Aux functions
# ======================================================================================================================
def _get_initial_candidate_solution(planning_problem):
    candidate_solution = {'investment': {}, 'total_capacity': {}}
    for e in range(len(planning_problem.active_distribution_network_nodes)):
        node_id = planning_problem.active_distribution_network_nodes[e]
        candidate_solution['investment'][node_id] = dict()
        candidate_solution['total_capacity'][node_id] = dict()
        for year in planning_problem.years:
            candidate_solution['investment'][node_id][year] = dict()
            candidate_solution['investment'][node_id][year]['s'] = 0.00
            candidate_solution['investment'][node_id][year]['e'] = 0.00
            candidate_solution['total_capacity'][node_id][year] = dict()
            candidate_solution['total_capacity'][node_id][year]['s'] = 0.00
            candidate_solution['total_capacity'][node_id][year]['e'] = 0.00
    return candidate_solution


def _add_shared_energy_storage_to_transmission_network(planning_problem):
    for year in planning_problem.years:
        for day in planning_problem.days:
            s_base = planning_problem.transmission_network.network[year][day].baseMVA
            for node_id in planning_problem.distribution_networks:
                shared_energy_storage = SharedEnergyStorage()
                shared_energy_storage.bus = node_id
                shared_energy_storage.dn_name = planning_problem.distribution_networks[node_id].name
                shared_energy_storage.s = shared_energy_storage.s / s_base
                shared_energy_storage.e = shared_energy_storage.e / s_base
                planning_problem.transmission_network.network[year][day].shared_energy_storages.append(shared_energy_storage)


def _add_shared_energy_storage_to_distribution_network(planning_problem):
    for year in planning_problem.years:
        for day in planning_problem.days:
            for node_id in planning_problem.distribution_networks:
                s_base = planning_problem.distribution_networks[node_id].network[year][day].baseMVA
                shared_energy_storage = SharedEnergyStorage()
                shared_energy_storage.bus = planning_problem.distribution_networks[node_id].network[year][day].get_reference_node_id()
                shared_energy_storage.dn_name = planning_problem.distribution_networks[node_id].network[year][day].name
                shared_energy_storage.s = shared_energy_storage.s / s_base
                shared_energy_storage.e = shared_energy_storage.e / s_base
                planning_problem.distribution_networks[node_id].network[year][day].shared_energy_storages.append(shared_energy_storage)
