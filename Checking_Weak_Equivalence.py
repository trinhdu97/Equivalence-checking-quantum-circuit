# from qiskit_ibm_runtime import EstimatorV2 as EstimatorV2
# from qiskit_ibm_runtime import EstimatorOptions
# from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile
#from qiskit_aer.primitives import Estimator as local_estimator
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Operator
import numpy as np
import json
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Pauli
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, Operator
import numpy as np




import Create_quantum_circuit
# import Compare_closeness_quantum_state 
# import Simulate_local_observable
import Manipulate_layers
# import Computing_with_reduced_state
# import Statistics
# import Deal_with_dictionary
import time
import local_projection_computation



def matrix_distances(A: np.ndarray, B: np.ndarray):
    """
    Compute the distance-1 (Frobenius norm) and distance-2 (spectral norm) between two matrices A and B.

    Parameters:
        A (np.ndarray): First input matrix.
        B (np.ndarray): Second input matrix.

    Returns:
        tuple: (distance_1, distance_2) where
            - distance_1 is the Frobenius norm ||A - B||_F
            - distance_2 is the spectral norm ||A - B||_2
    """
    diff = A - B
    # print(diff)

    distance_1 = np.linalg.norm(diff, 'fro')  # Frobenius norm
    distance_2 = np.linalg.norm(diff, 2)      # Spectral norm (largest singular value)

    return distance_1, distance_2



def local_projection_check_if_two_circuits_are_equal(qc_info_1, qc_info_2,tolerance = 1e-15):
    """
    Check if two quantum circuits are equal by checking their local projections.
    """
    # print("check with local projection")
    # Step 1: Get circuit properties
    n_qubits_1, depth_1 = Manipulate_layers.find_circuit_properties(qc_info_1)
    n_qubits_2, depth_2 = Manipulate_layers.find_circuit_properties(qc_info_2)

    # Step 2: Check if qubit count and depth are the same
    # if n_qubits_1 != n_qubits_2 or depth_1 != depth_2:
    #     print('Two circuits have different properties')
    #     return False

    # Step 3: Compute local projections for qc_info_1
    local_projection_1 = local_projection_computation.compute_local_projections_fullly_general(
        qc_info_1, n_qubits_1, depth_1, yes_print=False
    )

    # Step 4: Generate the inverse of qc_info_2 and compute local projections for it

    ## Remarks: local_projection_2 is computed from an initial local_projection_1, which is computed from qc_info_1, not from the beginning qc_info_2.
    qc_2_inverse, qc_2_inverse_info = Create_quantum_circuit.load_inverse_circuit_from_gate_info(
        qc_info_2, n_qubits_2
    )
    local_projection_2 = local_projection_computation.compute_local_projections_fullly_general(
        qc_2_inverse_info, n_qubits_2, depth_2, local_projection_1, yes_print=False
    )
    # print("local_projection_2",local_projection_2)
    # Step 5: Check each 'local_projection' in local_projection_2
    # tolerance = 1e-50
    count = 0
    # print("Type of local_projection_2:", type(local_projection_2))
    for entry in (local_projection_2):
        local_proj = entry['local_projection']
        # print(local_proj)
        count += 1
        # print(count)

        # Ensure local_proj is a square matrix
        if not isinstance(local_proj, np.ndarray) or local_proj.shape[0] != local_proj.shape[1]:
            print(f"Entry does not contain a valid square matrix.")
            return False

        # Create the target eigenvector
        target_eigenvector = np.zeros(local_proj.shape[0], dtype=np.complex128)
        target_eigenvector[0] = 1  # (1, 0, 0, ..., 0)
        projection_result = local_proj @ target_eigenvector
        # print("projection_result:",projection_result)
        if np.linalg.norm(projection_result-target_eigenvector,ord=1)>local_proj.shape[0]*tolerance:
            return False

    # If all checks pass
    # print("All local_projections in local_projection_2 have (1, 0, 0, ..., 0) as an eigenvector with eigenvalue 1.")
    return True


def state_vector_computation_check_if_two_circuits_are_equal(qc1, qc2, tolerance=1e-15):
    """
    This function checks if two quantum circuits are equivalent by comparing their output states.

    Verifies if two quantum circuits are equivalent by concatenating qc1 with the inverse of qc2
    and checking if the resulting output state is |0> (i.e., the vector [1, 0, 0, ...]).

    Arguments:
    - qc1 (QuantumCircuit): The first quantum circuit.
    - qc2 (QuantumCircuit): The second quantum circuit, which will be inverted.
    - tolerance (float): Precision tolerance for checking if the output state is close to [1, 0, 0, ..., 0].

    Returns:
    - bool: True if the circuits are equivalent (output is close to [1, 0, 0, ..., 0]), False otherwise.
    """

    # Check that both circuits have the same number of qubits
    if qc1.num_qubits != qc2.num_qubits:
        print("Circuits have different numbers of qubits and cannot be compared.")
        return False

    n_qubits = qc1.num_qubits

    # Create a new circuit that applies qc1 followed by the decomposed inverse of qc2
    combined_qc = QuantumCircuit(n_qubits)
    combined_qc.append(qc1.to_instruction(), range(n_qubits))
    
    # Decompose qc2.inverse() and append to combined_qc
    qc2_inverse = qc2.inverse().decompose()
    combined_qc.append(qc2_inverse.to_instruction(), range(n_qubits))

    # Simulate the combined circuit
    simulator = AerSimulator(method='statevector')
    transpiled_circuit = transpile(combined_qc, simulator)

    # Add a save_statevector instruction at the end of the transpiled circuit
    transpiled_circuit.save_statevector()

    # Run the circuit on the simulator and obtain the result (statevector)
    result = simulator.run(transpiled_circuit).result()
    statevector = result.get_statevector()
    
    # print("statevector:",statevector)
    # Define the target state vector: [1, 0, 0, ..., 0]
    target_state = np.zeros(2 ** n_qubits, dtype=np.complex128)
    target_state[0] = 1
    # print("target_state:",target_state)

    # Check if the resulting statevector is close to the target state
    if np.linalg.norm(statevector-target_state,ord=1)<target_state.shape[0]*tolerance:
        
    # if np.allclose(statevector, target_state, atol=tolerance):
        # print("The circuits are equivalent.")
        return True
    else:
        # print("The circuits are not equivalent.")
        return False




def benchmark_methods_comparison_old(file_name="comparison_results_running_time_checking_Inequivalence.txt", min_qubits=12, max_qubits=20, depth=3, n_patterns=3):
    """
    Benchmarks the running time of two equivalence-checking methods: local projection and statevector.

    Arguments:
    - file_name (str): Name of the output file to save results.
    - min_qubits (int): Minimum number of qubits to test (must be even).
    - max_qubits (int): Maximum number of qubits to test (must be even).
    - depth (int): Depth of the random circuits.
    - patterns (int): Number of consistent patterns required for each number of qubits.

    This function runs random circuits until it reaches the required number of consistent patterns for each qubit count,
    records average times and error bars, and notes any inconsistencies.
    """
    if min_qubits % 2 != 0:
        min_qubits += 1
    if max_qubits % 2 != 0:
        max_qubits -= 1
    
    results = []

    # Iterate over each even number of qubits in the specified range
    for n_qubits in range(min_qubits, max_qubits + 1, 2):
        local_projection_times = []
        statevector_times = []
        inconsistencies = 0

        # Run until we have the required number of consistent patterns
        while len(local_projection_times) < n_patterns:
            # Generate a random Haar circuit
            qc_1, qc_info_1 = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)
            qc_2, qc_info_2 = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)

            # Run the local projection method and measure time
            start_time = time.time()
            local_projection_result = local_projection_check_if_two_circuits_are_equal(qc_info_1, qc_info_2, tolerance=1e-15)
            end_time = time.time()
            local_projection_times.append(end_time - start_time)

            # Run the statevector method and measure time
            start_time = time.time()
            statevector_result = state_vector_computation_check_if_two_circuits_are_equal(qc_1, qc_2,tolerance=1e-15)
            end_time = time.time()
            statevector_times.append(end_time - start_time)

            # Check if results are consistent
            if local_projection_result != statevector_result:
                inconsistencies += 1  # Increment inconsistency count
                local_projection_times.pop()  # Remove this entry since it's inconsistent
                statevector_times.pop()  # Remove this entry as well
                print(f"Inconsistent result for {n_qubits} qubits, depth {depth}")
                continue  # Retry with a new pattern

        # Calculate average and standard deviation of times
        avg_lp_time = np.mean(local_projection_times)
        std_lp_time = np.std(local_projection_times)
        avg_sv_time = np.mean(statevector_times)
        std_sv_time = np.std(statevector_times)

        results.append((n_qubits, avg_lp_time, std_lp_time, avg_sv_time, std_sv_time, inconsistencies))

        # Write results to the file
        with open(file_name, 'w') as f:
            for n, avg_lp, std_lp, avg_sv, std_sv, incons in results:
                f.write(f"{n}\t{avg_lp}\t{std_lp}\t{avg_sv}\t{std_sv}\t{incons}\n")
        print(f"{n_qubits} qubits: Local Projection = {avg_lp_time:.6f}s ± {std_lp_time:.6f}s, "
              f"Statevector = {avg_sv_time:.6f}s ± {std_sv_time:.6f}s, Inconsistencies = {inconsistencies}")


def benchmark_methods_comparison(
    file_name="comparison_results_running_time_checking_Equivalence_with_larger_n_qubis_for_local_projection.txt", checking_Inequivalence=False,
    min_qubits_lp=12, max_qubits_lp=16,  # Qubit range for local projection
    min_qubits_sv=12, max_qubits_sv=14,  # Qubit range for statevector
    depth=3, n_patterns=3
):
    """
    Benchmarks the running time of two equivalence-checking methods: local projection and statevector.

    Arguments:
    - file_name (str): Name of the output file to save results.
    - min_qubits_lp (int): Minimum number of qubits for local projection (must be even).
    - max_qubits_lp (int): Maximum number of qubits for local projection (must be even).
    - min_qubits_sv (int): Minimum number of qubits for statevector (must be even).
    - max_qubits_sv (int): Maximum number of qubits for statevector (must be even).
    - depth (int): Depth of the random circuits.
    - patterns (int): Number of consistent patterns required for each number of qubits.

    This function runs random circuits until it reaches the required number of consistent patterns for each qubit count,
    records average times and error bars, and notes any inconsistencies.
    """
    # Adjust qubit ranges to ensure even numbers
    if min_qubits_lp % 2 != 0:
        min_qubits_lp += 1
    if max_qubits_lp % 2 != 0:
        max_qubits_lp -= 1
    if min_qubits_sv % 2 != 0:
        min_qubits_sv += 1
    if max_qubits_sv % 2 != 0:
        max_qubits_sv -= 1
    
    results = []

    # Determine the union of qubit ranges to cover both methods
    max_qubits = max(max_qubits_lp, max_qubits_sv)

    # Iterate over each even number of qubits in the unified range
    for n_qubits in range(min(min_qubits_lp, min_qubits_sv), max_qubits + 1, 2):
        local_projection_times = []
        statevector_times = []
        inconsistencies = 0

        # Skip if n_qubits falls outside the specified range for both methods
        if n_qubits > max_qubits_lp and n_qubits > max_qubits_sv:
            continue

        # Run until we have the required number of consistent patterns
        while len(local_projection_times) < n_patterns:
            # Generate a random Haar circuit
            qc_1, qc_info_1 = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)
            # Generate a second random Haar circuit
            qc_2, qc_info_2 = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)

            # Run the local projection method and measure time if within range
            if min_qubits_lp <= n_qubits <= max_qubits_lp:
                start_time = time.time()
                if checking_Inequivalence==True:
                    local_projection_result = local_projection_check_if_two_circuits_are_equal(qc_info_1, qc_info_2,tolerance=1e-15)
                else:
                    # For equivalence checking, compare qc_1 with itself
                    local_projection_result = local_projection_check_if_two_circuits_are_equal(qc_info_1, qc_info_1,tolerance=1e-15)
                # local_projection_result = local_projection_check_if_two_circuits_are_equal(qc_info_1, qc_info_2,tolerance=1e-15)
                end_time = time.time()
                local_projection_times.append(end_time - start_time)

            # Run the statevector method and measure time if within range
            if min_qubits_sv <= n_qubits <= max_qubits_sv:
                start_time = time.time()
                if checking_Inequivalence==True:
                    statevector_result = state_vector_computation_check_if_two_circuits_are_equal(qc_1, qc_2, tolerance=1e-15)
                else:
                    statevector_result = state_vector_computation_check_if_two_circuits_are_equal(qc_1, qc_1, tolerance=1e-15)
                end_time = time.time()
                statevector_times.append(end_time - start_time)

            # Check if results are consistent if both methods are applicable
            if min_qubits_lp <= n_qubits <= max_qubits_lp and min_qubits_sv <= n_qubits <= max_qubits_sv:
                if local_projection_result != statevector_result:
                    inconsistencies += 1  # Increment inconsistency count
                    local_projection_times.pop()  # Remove this entry since it's inconsistent
                    statevector_times.pop()  # Remove this entry as well
                    print(f"Inconsistent result for {n_qubits} qubits, depth {depth}")
                    continue  # Retry with a new pattern

        # Calculate average and standard deviation of times for each method
        avg_lp_time = np.mean(local_projection_times) if local_projection_times else None
        std_lp_time = np.std(local_projection_times) if local_projection_times else None
        avg_sv_time = np.mean(statevector_times) if statevector_times else None
        std_sv_time = np.std(statevector_times) if statevector_times else None

        results.append((n_qubits, avg_lp_time, std_lp_time, avg_sv_time, std_sv_time, inconsistencies))

        # Write results to the file
        with open(file_name, 'w') as f:
            for n, avg_lp, std_lp, avg_sv, std_sv, incons in results:
                f.write(f"{n}\t{avg_lp}\t{std_lp}\t{avg_sv}\t{std_sv}\t{incons}\n")
        
        # Print results with handling for None values
        lp_time_str = f"{avg_lp_time:.6f}s" if avg_lp_time is not None else "None"
        lp_std_str = f"± {std_lp_time:.6f}s" if std_lp_time is not None else ""
        sv_time_str = f"{avg_sv_time:.6f}s" if avg_sv_time is not None else "None"
        sv_std_str = f"± {std_sv_time:.6f}s" if std_sv_time is not None else ""

        print(f"{n_qubits} qubits: Local Projection = {lp_time_str} {lp_std_str}, "
              f"Statevector = {sv_time_str} {sv_std_str}, Inconsistencies = {inconsistencies}")





def plot_statevector_vs_local_projection(
    file_difference="Evaluation_Artifact_comparison_results_running_time_checking_Inequivalence.txt",
    file_equivalence="Evaluation_Artifact_comparison_results_running_time_checking_Equivalence.txt",
    output_file_name="comparison_results_running_time_checking_Equivalence_and_Inequivalence.png",
    depth=3):
    """
    Reads benchmark results from two files (one for difference and one for equivalence checking)
    and plots running time with error bars for each method. Uses dashed lines for the difference
    file and solid lines for the equivalence file. Colors are set to blue for Local Projection
    and orange for Statevector.
    
    Arguments:
    - file_difference (str): Name of the file containing benchmark results for checking difference.
    - file_equivalence (str): Name of the file containing benchmark results for checking equivalence.
    """
    def read_file(file_name):
        n_qubits, lp_times, lp_errors, sv_times, sv_errors = [], [], [], [], []
        with open(file_name, 'r') as f:
            for line in f:
                data = line.strip().split()
                # print(data)
                n_qubit = int(data[0])
                # print(n_qubit)
                lp_time = float(data[1]) if data[1] != 'None' else None
                lp_error = float(data[2]) if data[2] != 'None' else None
                sv_time = float(data[3]) if data[3] != 'None' else None
                sv_error = float(data[4]) if data[4] != 'None' else None
                
                n_qubits.append(n_qubit)
                lp_times.append(lp_time)
                lp_errors.append(lp_error)
                sv_times.append(sv_time)
                sv_errors.append(sv_error)
        return n_qubits, lp_times, lp_errors, sv_times, sv_errors

    # Read data from both files
    n_qubits_diff, lp_times_diff, lp_errors_diff, sv_times_diff, sv_errors_diff = read_file(file_difference)
    n_qubits_eq, lp_times_eq, lp_errors_eq, sv_times_eq, sv_errors_eq = read_file(file_equivalence)
    print(n_qubits_eq)
    n_qubits_eq
    # Plot the data
    plt.figure(figsize=(10, 10))

    # Plot Local Projection for Equivalence (solid blue line)
    if any(lp_times_eq):
        plt.errorbar(
            [n for n, t in zip(n_qubits_eq, lp_times_eq) if t is not None],
            [t for t in lp_times_eq if t is not None],
            yerr=[e for e in lp_errors_eq if e is not None],
            fmt='o-', color='orange', capsize=3, label="Local Projection (Equiv)"
        )

    # Plot Statevector for Equivalence (solid orange line)
    if any(sv_times_eq):
        plt.errorbar(
            [n for n, t in zip(n_qubits_eq, sv_times_eq) if t is not None],
            [t for t in sv_times_eq if t is not None],
            yerr=[e for e in sv_errors_eq if e is not None],
            fmt='s-', color='red', capsize=3, label="Statevector (Equiv)"
        )

    # Plot Local Projection for Difference (dashed blue line)
    if any(lp_times_diff):
        plt.errorbar(
            [n for n, t in zip(n_qubits_diff, lp_times_diff) if t is not None],
            [t for t in lp_times_diff if t is not None],
            yerr=[e for e in lp_errors_diff if e is not None],
            fmt='o--', color='blue', capsize=3, label="Local Projection (Inequiv)"
        )

    # Plot Statevector for Difference (dashed orange line)
    if any(sv_times_diff):
        plt.errorbar(
            [n for n, t in zip(n_qubits_diff, sv_times_diff) if t is not None],
            [t for t in sv_times_diff if t is not None],
            yerr=[e for e in sv_errors_diff if e is not None],
            fmt='s--', color='green', capsize=3, label="Statevector (Inequiv)"
        )

    

    # Labeling and visual details with specified font sizes
    plt.xlabel("Number of Qubits", fontsize=22)
    plt.ylabel("Average Running Time (seconds)", fontsize=22)
    plt.title(f"Checking Equivalence/Inequivalence at depth {depth}", fontsize=22)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(output_file_name, dpi=400)
    plt.show()
