from qiskit import QuantumCircuit, transpile
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import itertools
from qiskit_aer import AerSimulator

import Create_quantum_circuit
import Manipulate_layers
import local_projection_computation




def kron_power(vector, times):
    """
    Computes the Kronecker product of a vector with itself multiple times.

    Parameters:
    - vector (np.ndarray): The input vector to be tensorized.
    - times (int): The number of times to take the Kronecker product of the vector with itself.

    Returns:
    - np.ndarray: The resulting vector after applying the Kronecker product `times` times.
    """
    result = vector
    for _ in range(times - 1):
        result = np.kron(result, vector)
    return result

# Example Usage
v = np.array([1, 0])  # Define a simple vector, e.g., |0⟩
n_times = 3  # Number of times to take the Kronecker product

# kron_result = kron_power(v, n_times)
# print(f"Kronecker product of the vector {v} with itself {n_times} times:\n", kron_result)


def find_position_of_initial_qubit_in_entry(entry):
    """
    Finds the position of the initial qubit in the qubit_indices list and calculates how many qubits are before and after it.
    
    Arguments:
    - entry (dict): A dictionary representing a single entry from the local_projection_dict. 
                    It contains 'qubit_indices' (list) and 'initial_qubit' (int).
    
    Returns:
    - before (int): The number of qubits before the initial qubit in the qubit_indices list.
    - after (int): The number of qubits after the initial qubit in the qubit_indices list.
    """
    # Extract the qubit_indices and initial_qubit from the entry
    qubit_indices = entry['qubit_indices']
    initial_qubit = entry['initial_qubit']
    
    try:
        # Find the index of the initial_qubit in qubit_indices
        position = qubit_indices.index(initial_qubit)
        
        # Calculate the number of qubits before and after the initial qubit
        before = position
        after = len(qubit_indices) - position - 1
        
        return before, after
    except ValueError:
        # If the initial_qubit is not in qubit_indices, return a message indicating the error
        print(f"Error: The qubit {initial_qubit} is not in the qubit_indices list.")
        return None, None
    


def split_matrix_into_named_blocks(A):
    """
    Splits a square matrix into 4 blocks and returns them with specific names (B_00, B_01, B_10, B_11).
    
    Arguments:
    - A (ndarray): A square matrix of size 2^n x 2^n.

    Returns:
    - blocks (tuple): A tuple containing the four blocks:
      (B_00, B_01, B_10, B_11).
    """
    # Get the size of the matrix
    size = A.shape[0]
    
    # Ensure the matrix size is a power of 2
    if size % 2 != 0:
        raise ValueError("Matrix size must be a power of 2.")
    
    # Compute the middle index (for splitting the matrix)
    mid = size // 2
    
    # Split the matrix into four blocks
    B_00 = A[:mid, :mid]  # Top-left block
    B_01 = A[:mid, mid:]  # Top-right block
    B_10 = A[mid:, :mid]  # Bottom-left block
    B_11 = A[mid:, mid:]  # Bottom-right block
    
    return B_00, B_01, B_10, B_11
# # Example matrix of size 4x4 (2^2)
# A = np.array([[1, 2, 3, 4],
#               [5, 6, 7, 8],
#               [9, 10, 11, 12],
#               [13, 14, 15, 16]])

# # Split the matrix into 4 named blocks
# B_00, B_01, B_10, B_11 = split_matrix_into_named_blocks(A)

# # Output the blocks
# print("B_00 (Top-left block):")
# print(B_00)
# print("\nB_01 (Top-right block):")
# print(B_01)
# print("\nB_10 (Bottom-left block):")
# print(B_10)
# print("\nB_11 (Bottom-right block):")
# print(B_11)

A_00 = np.array([[1,0],[0,0]])
A_01 = np.array([[0,1],[0,0]])
A_10 = np.array([[0,0],[1,0]])
A_11 = np.array([[0,0],[0,1]])
# print("A_00 (Top-left block):")
# print(A_00)
# print("\nA_01 (Top-right block):")
# print(A_01)
# print("\nA_10 (Bottom-left block):")
# print(A_10)
# print("\nA_11 (Bottom-right block):")
# print(A_11)
def reconstruct_matrix_M_ACB(M_AB, M_C):
    Blocks_B = split_matrix_into_named_blocks(M_AB)
    A_00 = np.array([[1,0],[0,0]])
    A_01 = np.array([[0,1],[0,0]])
    A_10 = np.array([[0,0],[1,0]])
    A_11 = np.array([[0,0],[0,1]])
    M_ABC_00 = np.kron(A_00,np.kron(M_C,Blocks_B[0]))
    # print(M_ABC_00)
    M_ABC_01 = np.kron(A_01,np.kron(M_C,Blocks_B[1]))
    M_ABC_10 = np.kron(A_10,np.kron(M_C,Blocks_B[2]))
    M_ABC_11 = np.kron(A_11,np.kron(M_C,Blocks_B[3]))
    return M_ABC_00+M_ABC_01+M_ABC_10+M_ABC_11
# M_C = np.array([[1,0],[1,0]])
# print(reconstruct_matrix_M_ACB(A,M_C))


EPR_projector_AB = 1/2 * np.array([[1, 0, 0, 1], [0, 0, 0, 0],[0, 0, 0, 0],[1, 0, 0, 1]], dtype=np.complex128)
# print(split_matrix_into_named_blocks(EPR_projector_AB))
EPR_projector_ACB = reconstruct_matrix_M_ACB(EPR_projector_AB,np.eye(2))
# print(EPR_projector_ACB*2)


def compute_local_projections_with_Choi_isomorphism(gates_info, n_qubits, depth, given_dictionary=None,yes_print=None):  #### this is the efficient version of the local projection computation with Choi
    """
    Updates the local projections in the local_projection_dict based on gates_layers.

    Arguments:
    - gates_info (list): List of gate information for the circuit, where each gate includes its matrix.
    - local_projection_dict (list): The local projection dictionary initialized earlier.
    - n_qubits (int): The total number of qubits in the circuit.
    - depth (int): The expected number of layers in the circuit.

    Returns:
    - list: The updated local_projection_dict with updated local projections.
    """
    
    if depth == 0:
        return False

    ###### We now treat qubits as qudits of dimension 4


    # Divide the gates into layers and check the calculated depth against the expected depth.
    gates_layers, depth_check = Manipulate_layers.divide_circuit_into_layers_using_layer_index(gates_info)

    if given_dictionary == None:

        initial_dict = local_projection_computation.initialize_local_projection_dict(gates_layers,n_qubits)
        # print(initial_dict)
        dict_previous = [entry for entry in initial_dict if entry['layer'] == 0]
        for entry in dict_previous:
            layer = entry['layer']  # Extract the current layer from the entry.
            initial_qubit_entry = entry['initial_qubit']
            qubit_indices = entry['qubit_indices']  # Get the qubit indices for this entry.
            if yes_print == True:
                print('computing local projection for initial qubit:',initial_qubit_entry, ' involved:', qubit_indices)
            if layer == 0:
                # For layer 0, we need to update the local projection using the associated unitary matrix.
                U_index = None  # Initialize U_index to None before searching
                for gate in gates_layers[layer]:
                    # Check if the current gate's qubits match the qubit indices for layer 0.
                    if qubit_indices == gate['qubits']:
                        U_index = np.kron(np.eye(2),np.array(gate['matrix'], dtype=np.complex128))   # Get the unitary matrix for this gate.

                if U_index is None:
                    print("No matching gate found for layer 0 and qubit indices:", qubit_indices)
                    return False  # Handle the case where no gate is found
                EPR_vector = 1/np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128)
                EPR_projector_AB = 1/2 * np.array([[1, 0, 0, 1], [0, 0, 0, 0],[0, 0, 0, 0],[1, 0, 0, 1]], dtype=np.complex128)
                if initial_qubit_entry == qubit_indices[0]:
                    A_00 = np.array([[1,0],[0,0]])
                    EPR_projector = np.kron(EPR_projector_AB, np.eye(2)) 
                else:
                    EPR_projector = reconstruct_matrix_M_ACB(EPR_projector_AB,np.eye(2))
                # print(EPR2)

                # Update the local projection using the unitary matrix and a measurement projection.
                entry['local_projection'] = U_index @ EPR_projector @ U_index.conj().T
                # print(entry['local_projection'].shape)
                
    else:
        initial_dict = local_projection_computation.initialize_local_projection_with_initial_dict(gates_info, given_dictionary)  
        # print(initial_dict)
    # Sort the local projection dictionary by layer to ensure correct processing order.
    # initial_dict.sort(key=lambda x: x['layer'])


    if depth_check != depth:
        return False  # Return False if the depths do not match, indicating an inconsistency.
        
    if depth == 0:
        return given_dictionary
    

    # Process each entry in the sorted local projection dictionary.
    for layer_check in range(depth):
        # print(layer_check)
        if layer_check == 0:
            if given_dictionary == None:
                continue  
            else:
                dict_previous = [e for e in initial_dict if e['layer']==0]
            # print(dict_previous)
                
        # else:
        #     print('layer = ', layer_check)
        
        dict_current = [entry for entry in initial_dict if entry['layer'] == layer_check]
        
        for entry in dict_current:
            layer = entry['layer']  # Extract the current layer from the entry.
            initial_qubit_entry = entry['initial_qubit']
            qubit_indices = entry['qubit_indices']  # Get the qubit indices for this entry.

            # For layers k > 0, we need to perform additional checks and updates.
            previous_layer_entry = next(
                (e for e in dict_previous if e['initial_qubit'] == initial_qubit_entry),
                None
            )
            if yes_print == True:
                print('computing local projection for initial qubit:',initial_qubit_entry, ' involved:', qubit_indices)
            # If there's a previous layer entry, we need to update the projection.
            if previous_layer_entry:
                previous_projection = previous_layer_entry['local_projection']  # Get the previous local projection.
                # print('ssss',previous_projection.shape)
                # Determine any additional qubits that are present in the current layer but not in the previous layer.
                additional_qubits_previous = set(qubit_indices) - set(previous_layer_entry['qubit_indices'])
                # print(additional_qubits_previous)
                for index in additional_qubits_previous:
                    # Adjust the previous projection for new qubits added.
                    if index < previous_layer_entry['qubit_indices'][0]:
                        previous_projection = reconstruct_matrix_M_ACB(previous_projection,np.eye(2))  # Add identity for leftmost qubit.
                    elif index > previous_layer_entry['qubit_indices'][-1]:
                        previous_projection = np.kron(previous_projection, np.eye(2))  # Add identity for rightmost qubit.
                    else:
                        # If the additional qubit is not at the end, there is a contradiction.
                        print('contradiction 1')
                        return False
            
            # Initialize matrix for the current layer's projection.
            matrix = np.eye(1, dtype=np.complex128)  # Start with the identity matrix for tensor products.
            qubit_count = []  # List to keep track of qubit counts involved in the current layer's gates.

            # Check the gates in the current layer to build the current projection.
            for gate_info in gates_layers[layer]:
                involved_qubits = gate_info['qubits']  # Get the qubits involved in the current gate.
                # print(involved_qubits)
                # print("checkkkkkkkkk")
                # If any of the involved qubits intersect with the current qubit_indices, update the matrix.
                if set(involved_qubits).intersection(set(qubit_indices)):
                    # print(involved_qubits)
                    unitary_matrix = (np.array(gate_info['matrix'], dtype=np.complex128))  # Get the unitary matrix.
                    matrix = np.kron(matrix, unitary_matrix)  # Update the matrix with the unitary for this gate.
                    # qubit_count.append(involved_qubits[0])  # Track the first qubit.
                    # qubit_count.append(involved_qubits[1])  # Track the second qubit.
                    qubit_count = qubit_count + involved_qubits
            # print(matrix.shape)
            # print(qubit_count)

            # Check if qubit_count is a sequential natural number array.
            if not local_projection_computation.is_sequential_natural_numbers(qubit_count):
                print('contradiction 2')
                return False  # Return False for non-sequential qubit indices.

            # Handle additional qubits not included in the current layer's unitary operations.
            additional_qubits_next = set(qubit_indices) - set(qubit_count)  # Find qubits that need identities.
            # print(qubit_indices, additional_qubits_next,qubit_count)
            for index in additional_qubits_next:
                if len(additional_qubits_next) > 1:
                    print('contradiction 3')  # More than one additional qubit is a contradiction.
                    return False
                elif index < qubit_count[0]:
                    # print("top")
                    matrix = np.kron(np.eye(2), matrix)  # Add identity to the left for new qubits.
                    # print('yyy',matrix.shape)
                elif index > qubit_count[-1]:
                    # print('bottom')
                    matrix = np.kron(matrix, np.eye(2))  # Add identity to the right for new qubits.
                else:
                    print('contradiction 4')  # Found a contradiction if the index is not in expected order.
                    return False
            matrix = np.kron(np.eye(2),matrix)
            
            # print('xxx',matrix.shape)
            # print(previous_projection.shape)

            # Finally, update the local projection for the current entry.
            current_local_projection = matrix @ previous_projection @ matrix.conj().T
            entry['local_projection'] = current_local_projection  # Store the updated projection matrix.
        
        dict_previous = dict_current  # Update dict_previous for the next layer check.

    return dict_previous  # Return the updated local projection dictionary.




# import Checking_equivalence_with_Choi_isomorphism as CEWC
def check_if_two_circuits_are_equal_using_Choi_isomorphism(qc_info_1, qc_info_2,tolerance=1e-15):   
    """ Checks if two quantum circuits are equivalent using the Choi isomorphism."""
    # Step 1: Get circuit properties
    n_qubits_1, depth_1 = Manipulate_layers.find_circuit_properties(qc_info_1)
    n_qubits_2, depth_2 = Manipulate_layers.find_circuit_properties(qc_info_2)

    # Step 2: Check if qubit count and depth are the same
    if n_qubits_1 != n_qubits_2:
        print('Two circuits have different properties')
        return False
    
    # Step 3: Get inverse circuit and combine
    inverse_qc_2, inverse_qc_info_2 = Create_quantum_circuit.load_inverse_circuit_from_gate_info(qc_info_2, n_qubits_1)
    qc_check_info = Create_quantum_circuit.combine_circuits_info(qc_info_1, inverse_qc_info_2)
    depth = depth_1 + depth_2
    
    # Step 4: Compute local projections for the combined circuit
    local_projection_check = compute_local_projections_with_Choi_isomorphism(qc_check_info, n_qubits_1, depth)

    # Step 5: Set up the EPR projector
    EPR_projector_AB = 1/2 * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.complex128)

    # Step 6: Check equivalence using the local projections
    for entry in local_projection_check:
        structure = find_position_of_initial_qubit_in_entry(entry)
        
        # Reconstruct test density matrix based on the structure
        test_density_matrix = EPR_projector_AB
        test_density_matrix = reconstruct_matrix_M_ACB(EPR_projector_AB, np.eye(2**structure[0]))
        test_density_matrix = np.kron(test_density_matrix, np.eye(2**structure[1]))

        # Compute the image after applying the local projection
        image = entry['local_projection'] @ test_density_matrix
        
        # Compare the resulting matrix with the original test_density_matrix

        if np.linalg.norm(image-test_density_matrix,ord=1)>image.shape[0]*tolerance:
            return False
        # if not np.allclose(image, test_density_matrix, atol=1e-14):
        #     # print("Inequivalence found")
        #     return False  # Return immediately if matrices are not equal

    # print("The circuits are equivalent.")
    return True

def benchmark_equivalence_vs_inequivalence(
    file_equivalence="results_equivalence_check_depth_3_retry.txt",
    file_inequivalence="results_inequivalence_check_depth_3_retry.txt",
    min_qubits=4, max_qubits=20,  # Range of qubits to test
    depth=3, n_patterns=10  # Circuit depth and number of patterns
):
    """
    Benchmarks the running time for equivalence and inequivalence checks.

    Arguments:
    - file_equivalence (str): Output file for equivalence check results.
    - file_inequivalence (str): Output file for inequivalence check results.
    - min_qubits (int): Minimum number of qubits.
    - max_qubits (int): Maximum number of qubits.
    - depth (int): Depth of the random circuits.
    - patterns (int): Number of consistent patterns for each qubit count.
    """
    results_equivalence = []
    results_inequivalence = []

    # Iterate over each number of qubits in the specified range
    for n_qubits in range(min_qubits, max_qubits + 1, 4):  # Step by 2 for even qubits only
        eq_times = []
        ineq_times = []

        # Run until we reach the required number of patterns
        for _ in range(n_patterns):
            # Generate circuits for equivalence and inequivalence tests
            qc_1, qc_info_1 = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)
            qc_2, qc_info_2 = qc_1.copy(), qc_info_1  # Identical circuit for equivalence
            qc_3, qc_info_3 = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)  # Different circuit for inequivalence

            # Measure time for equivalence checking
            start_time_eq = time.time()
            check_if_two_circuits_are_equal_using_Choi_isomorphism(qc_info_1, qc_info_2)
            end_time_eq = time.time()
            eq_times.append(end_time_eq - start_time_eq)

            # Measure time for inequivalence checking
            start_time_ineq = time.time()
            check_if_two_circuits_are_equal_using_Choi_isomorphism(qc_info_1, qc_info_3)
            end_time_ineq = time.time()
            ineq_times.append(end_time_ineq - start_time_ineq)

        # Calculate average and standard deviation of times
        avg_eq_time = np.mean(eq_times)
        std_eq_time = np.std(eq_times)
        avg_ineq_time = np.mean(ineq_times)
        std_ineq_time = np.std(ineq_times)

        # Append results for equivalence and inequivalence
        results_equivalence.append((n_qubits, avg_eq_time, std_eq_time))
        results_inequivalence.append((n_qubits, avg_ineq_time, std_ineq_time))

        # Print results
        print(f"{n_qubits} qubits: Equivalence = {avg_eq_time:.6f}s ± {std_eq_time:.6f}s, "
              f"Inequivalence = {avg_ineq_time:.6f}s ± {std_ineq_time:.6f}s")

    # Write equivalence results to file
    with open(file_equivalence, 'w') as fe:
        fe.write("n_qubits\tavg_time\ttime_std\n")
        for n, avg_time, time_std in results_equivalence:
            fe.write(f"{n}\t{avg_time}\t{time_std}\n")

    # Write inequivalence results to file
    with open(file_inequivalence, 'w') as fi:
        fi.write("n_qubits\tavg_time\ttime_std\n")
        for n, avg_time, time_std in results_inequivalence:
            fi.write(f"{n}\t{avg_time}\t{time_std}\n")





def plot_benchmark_results(file_equivalence, file_inequivalence,output_file_name="Running_time_checking_equivalence_difference_using_Choi_state.pdf", depth=3):
    """
    Plots the running time for equivalence and inequivalence checks from benchmark results.

    Arguments:
    - file_equivalence (str): File path for equivalence check results.
    - file_inequivalence (str): File path for inequivalence check results.
    """

    def read_file(file_name):
        # Initialize lists to store data
        n_qubits, avg_times, time_stds = [], [], []
        
        # Read file and extract data
        with open(file_name, 'r') as f:
            next(f)  # Skip header
            for line in f:
                data = line.strip().split("\t")
                n_qubits.append(int(data[0]))
                avg_times.append(float(data[1]))
                time_stds.append(float(data[2]))
                
        return n_qubits, avg_times, time_stds

    # Read data from both files
    n_qubits_eq, avg_times_eq, time_stds_eq = read_file(file_equivalence)
    n_qubits_ineq, avg_times_ineq, time_stds_ineq = read_file(file_inequivalence)
    print("n_qubits_eq",n_qubits_eq)
    print("n_qubits_ineq",n_qubits_ineq)
    # Plot results
    plt.figure(figsize=(10, 10))

    # Plot equivalence checking times with error bars
    plt.errorbar(
        n_qubits_eq, avg_times_eq, yerr=time_stds_eq, fmt='o-', color='blue',
        capsize=10, label="Equivalence Checking"
    )

    # Plot inequivalence checking times with error bars
    plt.errorbar(
        n_qubits_ineq, avg_times_ineq, yerr=time_stds_ineq, fmt='s--', color='red',
        capsize=10, label="Inequivalence Checking"
    )

    # Labeling
    plt.xlabel("Number of Qubits", fontsize=22)
    plt.ylabel("Average Running Time (seconds)", fontsize=22)
    plt.title(f"Checking Equivalence/Inequivalence (Choi state) at depth {depth}", fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(output_file_name)
    plt.show()