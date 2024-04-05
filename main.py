import os
import sys
import getopt
from shared_resources_planning import SharedResourcesPlanning


# ======================================================================================================================
#  Read Execution Arguments
# ======================================================================================================================
def print_help_message():
    print('\nShared Resources Planning Tool usage:')
    print('\n Usage: main.py [OPTIONS]')
    print('\n Options:')
    print('   {:25}  {}'.format('-d, --test_case=', 'Directory of the Test Case to be run (located inside "data" directory)'))
    print('   {:25}  {}'.format('-f, --specification_file=', 'Specification file of the Test Case to be run (located inside the "test_case" directory)'))
    print('   {:25}  {}'.format('-h, --help', 'Help. Displays this message'))


def read_execution_arguments(argv):

    test_case_dir = str()
    spec_filename = str()

    try:
        opts, args = getopt.getopt(argv, 'hd:f:', ['help', 'test_case=', 'specification_file='])
    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)

    if not argv or not opts:
        print_help_message()
        sys.exit()

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help_message()
            sys.exit()
        elif opt in ('-d', '--dir'):
            test_case_dir = arg
        elif opt in ('-f', '--file'):
            spec_filename = arg

    if not test_case_dir or not spec_filename:
        print('Shared Resource Planning Tool usage:')
        print('\tmain.py -d <test_case> -f <specification_file>')
        sys.exit()

    return spec_filename, test_case_dir


# ======================================================================================================================
#  Shared Resources Planning
# ======================================================================================================================
def shared_resources_planning(working_directory, specification_filename):

    print('==========================================================================================================')
    print('                                    SHARED RESOURCES PLANNING (w/ ESSO)                                   ')
    print('==========================================================================================================')

    planning_problem = SharedResourcesPlanning(working_directory, specification_filename)
    planning_problem.read_planning_problem()
    #planning_problem.plot_diagram()
    #planning_problem.run_planning_problem()

    candidate_solution = planning_problem.get_initial_candidate_solution()
    #planning_problem.run_operational_planning(candidate_solution=candidate_solution, print_results=True, debug_flag=False)
    #planning_problem.run_without_coordination(print_results=True)

    '''
    transmission_network = planning_problem.transmission_network
    transmission_network.update_data_with_candidate_solution(candidate_solution['total_capacity'])
    tn_model = transmission_network.build_model()
    results = transmission_network.optimize(tn_model)
    processed_results = transmission_network.process_results(tn_model, results)
    transmission_network.write_optimization_results_to_excel(processed_results)
    '''
    distribution_networks = planning_problem.distribution_networks
    for node_id in distribution_networks:
        distribution_network = distribution_networks[node_id]
        distribution_network.update_data_with_candidate_solution(candidate_solution['total_capacity'])
        dn_model = distribution_network.build_model()
        results = distribution_network.optimize(dn_model)
        processed_results = distribution_network.process_results(dn_model, results)
        distribution_network.write_optimization_results_to_excel(processed_results)
    '''
    candidate_solution = planning_problem.get_initial_candidate_solution()
    esso_model = planning_problem.shared_ess_data.build_subproblem()
    planning_problem.shared_ess_data.update_model_with_candidate_solution(esso_model, candidate_solution['investment'])
    results = planning_problem.shared_ess_data.optimize(esso_model)
    planning_problem.shared_ess_data.write_optimization_results_to_excel(esso_model)
    '''

    print('==========================================================================================================')
    print('                                                 END                                                      ')
    print('==========================================================================================================')


# ======================================================================================================================
#  Main
# ======================================================================================================================
if __name__ == '__main__':
    filename, test_case = read_execution_arguments(sys.argv[1:])
    directory = os.path.join(os.getcwd(), 'data', test_case)
    shared_resources_planning(directory, filename)
