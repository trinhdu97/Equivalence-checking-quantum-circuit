from qiskit import QuantumCircuit, transpile
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
import itertools
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, Operator
import numpy as np
import time

import Create_quantum_circuit
import Manipulate_layers
import time
import local_projection_computation

def matrix_distances(A: np.ndarray, B: np.ndarray):
    """
    Compute vector-like distances between two matrices A and B.

    Parameters:
        A (np.ndarray): First input matrix.
        B (np.ndarray): Second input matrix.

    Returns:
        tuple: (L1_distance, L2_distance, Linf_distance) where
            - L1_distance is the Manhattan norm ||A - B||_1 (sum of absolute differences)
            - L2_distance is the Euclidean norm ||A - B||_2 (square root of sum of squared differences)
            - Linf_distance is the Chebyshev norm ||A - B||_∞ (max absolute difference)
    """
    diff = A - B
    # print(diff)
    # Flatten the matrices into vectors
    diff_flat = diff.flatten()
    
    # Compute vector norms
    L1_distance = np.linalg.norm(diff_flat, 1)  # Manhattan norm (sum of absolute differences)
    L2_distance = np.linalg.norm(diff_flat, 2)  # Euclidean norm (square root of sum of squares)
    Linf_distance = np.linalg.norm(diff_flat, np.inf)  # Chebyshev norm (max absolute difference)

    # print("L1 Distance =", L1_distance, ', L2 Distance =', L2_distance, ', Linf Distance =', Linf_distance)
    return L1_distance, L2_distance, Linf_distance




def check_error_in_equivalence_checking(qc_info_1, qc_info_2):
    """
    Check equivalence of two quantum circuits via local projection analysis,
    and return the averaged error vector.

    Returns:
        average_error (np.ndarray): Averaged error vector of shape (3,), or None if check fails.
    """
    # Step 1: Extract circuit properties
    n_qubits_1, depth_1 = Manipulate_layers.find_circuit_properties(qc_info_1)
    n_qubits_2, depth_2 = Manipulate_layers.find_circuit_properties(qc_info_2)

    # Step 2: Sanity check
    if n_qubits_1 != n_qubits_2 or depth_1 != depth_2:
        print("Two circuits have different properties")
        return None

    # Step 3: Compute local projections for qc_info_1
    local_projection_1 = local_projection_computation.compute_local_projections_fullly_general(
        qc_info_1, n_qubits_1, depth_1, yes_print=False
    )

    # Step 4: Compute projections for inverse of qc_info_2
    _, qc_2_inverse_info = Create_quantum_circuit.load_inverse_circuit_from_gate_info(
        qc_info_2, n_qubits_2
    )
    local_projection_2 = local_projection_computation.compute_local_projections_fullly_general(
        qc_2_inverse_info, n_qubits_2, depth_2, local_projection_1, yes_print=False
    )

    # Step 5: Analyze local projections
    tolerance = 1e-15
    max_proj_dim = max(entry['local_projection'].shape[0] for entry in local_projection_2)
    error_vector_cumulative = np.zeros(3)
    count = 0

    for i, entry in enumerate(local_projection_2):
        local_proj = entry['local_projection']

        if not isinstance(local_proj, np.ndarray) or local_proj.shape[0] != local_proj.shape[1]:
            print(f"Entry {i} does not contain a valid square matrix.")
            return None
        shape = local_proj.shape[0]
        target_eigenvector = np.zeros(shape, dtype=np.complex128)
        target_eigenvector[0] = 1
        projection_result = local_proj @ target_eigenvector

        # if local_proj.shape[0] == max_proj_dim:
        #     count += 1
        #     error_vector = matrix_distances(projection_result, target_eigenvector)
        #     error_vector_cumulative += np.array([
        #         error_vector[0] / (max_proj_dim),
        #         error_vector[1] / np.sqrt(max_proj_dim),
        #         error_vector[2]
        #     ])

        
        count += 1
        error_vector = matrix_distances(projection_result, target_eigenvector)
        error_vector_cumulative += np.array([
                error_vector[0] / shape,
                error_vector[1] / np.sqrt(shape),
                error_vector[2]
            ])

        if np.linalg.norm(projection_result-target_eigenvector,ord=1)>local_proj.shape[0]*tolerance:
            return None

    if count == 0:
        return None

    average_error = error_vector_cumulative / count
    return average_error



def run_and_record_errors_for_random_circuits(n_qubits = 300, min_depth=1, max_depth=3, output_file_path='Evaluation_Artifact_Error_equivalence_checking.txt', n_patterns=2):
    """
    Run equivalence checking for random Haar circuits over multiple trials per depth,
    and record the average and standard deviation of error vectors into a .txt file.

    Parameters:
        min_depth (int): Minimum circuit depth to evaluate.
        max_depth (int): Maximum circuit depth to evaluate (inclusive).
        output_file_path (str): Path to the output .txt file.
        num_trials (int): Number of repetitions per depth.

    Returns:
        None
    """
    
    results = []

    for depth in range(min_depth, max_depth + 1):
        print(f"\nRunning {n_patterns} trials for depth = {depth}")
        trial_errors = []

        for trial in range(n_patterns):
            # print(trial)
            try:
                qc, qc_info = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)
                avg_error = check_error_in_equivalence_checking(qc_info, qc_info)

                if avg_error is not None:
                    trial_errors.append(avg_error)
                else:
                    print(f"Trial {trial + 1} at depth {depth} failed.")
            except Exception as e:
                print(f"Trial {trial + 1} at depth {depth} raised exception: {e}")

        if len(trial_errors) == 0:
            results.append((depth, [None, None, None], [None, None, None]))
            print(f"No successful trials for depth {depth}.")
        else:
            error_matrix = np.array(trial_errors)  # shape: (num_trials_successful, 3)
            avg = np.mean(error_matrix, axis=0)
            std = np.std(error_matrix, axis=0)
            results.append((depth, avg, std))
            # print(f"Avg error: {avg}, Std: {std}")

    # Write to file
    with open(output_file_path, 'w') as f:
        f.write("depth,avg_error_norm,avg_vector_dist,avg_inf_norm,std_error_norm,std_vector_dist,std_inf_norm\n")
        for depth, avg, std in results:
            if None in avg:
                f.write(f"{depth},NaN,NaN,NaN,NaN,NaN,NaN\n")
            else:
                f.write(f"{depth},{avg[0]},{avg[1]},{avg[2]},{std[0]},{std[1]},{std[2]}\n")

    print(f"\nAll results written to {output_file_path}")



def plot_log_errors_from_file(file_path='Evaluation_Artifact_Error_equivalence_checking.txt', output_file='Evaluation_ArtifactError_from_equivalence_checking.pdf',n_qubits=300,n_patterns=2):
    """
    Reads error data with averages and standard deviations from a file and plots
    log10 of the errors with error bars.

    Parameters:
        file_path (str): Path to the .txt file with columns:
            depth, avg_e1, avg_e2, avg_e3, std_e1, std_e2, std_e3

    Returns:
        None
    """
    depths = []
    log_avg_e1, log_avg_e2, log_avg_e3 = [], [], []
    log_std_e1, log_std_e2, log_std_e3 = [], [], []

    with open(file_path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue
            try:
                depth = int(parts[0])
                avg_e1 = float(parts[1])
                avg_e2 = float(parts[2])
                avg_e3 = float(parts[3])
                std_e1 = float(parts[4])
                std_e2 = float(parts[5])
                std_e3 = float(parts[6])

                if all(e > 0 for e in (avg_e1, avg_e2, avg_e3)):
                    depths.append(depth)
                    log_avg_e1.append(np.log10(avg_e1))
                    log_avg_e2.append(np.log10(avg_e2))
                    log_avg_e3.append(np.log10(avg_e3))

                    # std(log10(x)) ≈ std(x) / (x * ln(10))
                    log_std_e1.append(std_e1 / (avg_e1 * np.log(10)))
                    log_std_e2.append(std_e2 / (avg_e2 * np.log(10)))
                    log_std_e3.append(std_e3 / (avg_e3 * np.log(10)))
            except ValueError:
                continue  # Skip lines with invalid data

    if not depths:
        print("No valid data to plot.")
        return

    # Sort by depth
    sorted_data = sorted(zip(depths, log_avg_e1, log_std_e1,
                                      log_avg_e2, log_std_e2,
                                      log_avg_e3, log_std_e3))
    depths, log_avg_e1, log_std_e1, log_avg_e2, log_std_e2, log_avg_e3, log_std_e3 = zip(*sorted_data)

    # Plotting with error bars and custom fonts
    plt.figure(figsize=(12, 8))
    plt.errorbar(depths, log_avg_e1, yerr=log_std_e1, fmt='o-', capsize=4, label='log₁₀(L1)')
    plt.errorbar(depths, log_avg_e2, yerr=log_std_e2, fmt='o-', capsize=4, label='log₁₀(L2)')
    plt.errorbar(depths, log_avg_e3, yerr=log_std_e3, fmt='o-', capsize=4, label='log₁₀(inf_norm)')

    plt.xlabel('Circuit Depth', fontsize=20)
    plt.ylabel('log₁₀(Error)', fontsize=20)
    plt.title(f'Log-scaled Averaged Errors (over {n_patterns} Haar random {n_qubits}-qubit circuits)', fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(np.arange(min(depths), max(depths) + 1, 1), fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()