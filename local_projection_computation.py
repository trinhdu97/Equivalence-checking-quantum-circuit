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
import itertools

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace, Operator
import Create_quantum_circuit
# import Compare_closeness_quantum_state 
# import Simulate_local_observable
import Manipulate_layers
import time
# import Computing_with_reduced_state
# import Statistics
# import Deal_with_dictionary

def initialize_local_projection_dict(layers_qc, n_qubits):
    """
    Initializes a list to store local projection information for each layer, based on layers in the circuit.

    Arguments:
    - layers_qc (list): List of layers, where each layer contains gate information dictionaries.
    - n_qubits (int): The total number of qubits in the circuit.

    Returns:
    - local_projection_dict (list): A list of dictionaries with 'layer', 'initial_qubit', 'qubit_indices', 
                                    and 'local_projection'.
    """
    local_projection_dict = []

    # Iterate over each qubit in the circuit
    for qubit in range(n_qubits):
        # Get involving qubits for the current qubit across all layers using forward lightcone
        involving_qubits_per_layer = Manipulate_layers.extract_involving_qubits_in_forward_lightcone_using_layer_index(layers_qc, [qubit])

        for layer, involving_qubits in involving_qubits_per_layer:
            # Check if this layer and involving qubits combination already exists
            existing_entry = next(
                (entry for entry in local_projection_dict 
                 if entry['layer'] == layer and entry['qubit_indices'] == list(involving_qubits)),
                None
            )

            if existing_entry is None:
                # Initialize the local projection matrix as a placeholder
                projection_matrix = -1  # Placeholder for projection matrix

                # Add the information as a dictionary to the list
                local_projection_dict.append({
                    'layer': layer,
                    'initial_qubit': qubit,
                    'qubit_indices': list(involving_qubits),  # Keep it as a regular list
                    'local_projection': projection_matrix
                })

    # Sort the dictionary by both 'layer' and 'initial_qubit' to ensure correct order
    local_projection_dict.sort(key=lambda x: (x['layer'], x['initial_qubit']))

    return local_projection_dict

def get_elements_involving_qubit(local_projection_dict, qubit):
    """
    Retrieves all elements from the local projection dictionary that involve a specific qubit and sorts them by layer.

    Arguments:
    - local_projection_dict (list): The list of dictionaries containing local projection information.
    - qubit (int): The qubit index for which to retrieve involved elements.

    Returns:
    - involved_elements (list): A sorted list of dictionaries from local_projection_dict that involve the specified qubit.
    """
    involved_elements = []

    # Iterate through the local projection dictionary
    for entry in local_projection_dict:
        if qubit == entry['initial_qubit']:
            involved_elements.append(entry)

    # Sort the involved elements by the layer number
    involved_elements.sort(key=lambda x: x['layer'])

    return involved_elements

def is_sequential_natural_numbers(num_set):
    """
    Checks if the given set of integers represents a sequential array of natural numbers, including 0.

    Arguments:
    - num_set (set): A set of integers to check.

    Returns:
    - bool: True if the set represents a sequential array of natural numbers (including 0), False otherwise.
    """
    # Convert set to a sorted list
    sorted_numbers = sorted(num_set)
    
    # Check if the sorted list starts with a natural number (0 or greater)
    if not sorted_numbers or sorted_numbers[0] < 0:
        return False  # Must start with 0 or greater

    # Check if the numbers are sequential
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i] + 1 != sorted_numbers[i + 1]:
            return False  # Found a gap between consecutive numbers

    return True  # All numbers are consecutive


def are_consecutive_natural_numbers(local_projection_dict):
    """
    Checks if all qubit_indices in the local_projection_dict form a set of consecutive natural numbers, including 0.

    Arguments:
    - local_projection_dict (list): A list of dictionaries containing 'qubit_indices'.

    Returns:
    - bool: True if all qubit_indices are consecutive natural numbers (including 0), False otherwise.
    """
    all_qubit_indices = set()  # Initialize a set to collect all qubit indices

    # Collect all qubit indices from the local_projection_dict
    for entry in local_projection_dict:
        qubit_indices = entry['qubit_indices']
        all_qubit_indices.update(qubit_indices)

    # Use the is_sequential_natural_numbers function to check the collected indices
    return is_sequential_natural_numbers(all_qubit_indices)


def compute_local_projections(gates_info, n_qubits, depth,dict_previous=None):
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

    
    # Divide the gates into layers and check the calculated depth against the expected depth.
    gates_layers, depth_check = Manipulate_layers.divide_circuit_into_layers_using_layer_index(gates_info)

    initial_dict = initialize_local_projection_dict(gates_layers,n_qubits)
    # Sort the local projection dictionary by layer to ensure correct processing order.
    # initial_dict.sort(key=lambda x: x['layer'])


    if depth_check != depth:
        return False  # Return False if the depths do not match, indicating an inconsistency.

    if dict_previous == None:
        dict_previous = [entry for entry in initial_dict if entry['layer'] == 0]
        for entry in dict_previous:
            layer = entry['layer']  # Extract the current layer from the entry.
            initial_qubit_entry = entry['initial_qubit']
            qubit_indices = entry['qubit_indices']  # Get the qubit indices for this entry.
            print('computing local projection for initial qubit:',initial_qubit_entry, ' involved:', qubit_indices)
            if layer == 0:
                # For layer 0, we need to update the local projection using the associated unitary matrix.
                U_index = None  # Initialize U_index to None before searching
                for gate in gates_layers[layer]:
                    # Check if the current gate's qubits match the qubit indices for layer 0.
                    if qubit_indices == gate['qubits']:
                        U_index = np.array(gate['matrix'], dtype=np.complex128)  # Get the unitary matrix for this gate.

                if U_index is None:
                    print("No matching gate found for layer 0 and qubit indices:", qubit_indices)
                    return False  # Handle the case where no gate is found

                # Update the local projection using the unitary matrix and a measurement projection.
                entry['local_projection'] = U_index @ np.array([[1, 0, 0, 0], 
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0]], dtype=np.complex128) @ U_index.conj().T
    if depth == 0:
        return dict_previous
    

    # Process each entry in the sorted local projection dictionary.
    for layer_check in range(depth):
        if layer_check == 0:
            continue
        else:
            print('layer = ', layer_check)
        
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
            print('computing local projection for initial qubit:',initial_qubit_entry, ' involved:', qubit_indices)
            # If there's a previous layer entry, we need to update the projection.
            if previous_layer_entry:
                previous_projection = previous_layer_entry['local_projection']  # Get the previous local projection.

                # Determine any additional qubits that are present in the current layer but not in the previous layer.
                additional_qubits_previous = set(qubit_indices) - set(previous_layer_entry['qubit_indices'])
                for index in additional_qubits_previous:
                    # Adjust the previous projection for new qubits added.
                    if index < previous_layer_entry['qubit_indices'][0]:
                        previous_projection = np.kron(np.eye(2), previous_projection)  # Add identity for leftmost qubit.
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

                # If any of the involved qubits intersect with the current qubit_indices, update the matrix.
                if set(involved_qubits).intersection(set(qubit_indices)):
                    unitary_matrix = np.array(gate_info['matrix'], dtype=np.complex128)  # Get the unitary matrix.
                    matrix = np.kron(matrix, unitary_matrix)  # Update the matrix with the unitary for this gate.
                    qubit_count.append(involved_qubits[0])  # Track the first qubit.
                    qubit_count.append(involved_qubits[1])  # Track the second qubit.
            # print(qubit_count)

            # Check if qubit_count is a sequential natural number array.
            if not is_sequential_natural_numbers(qubit_count):
                print('contradiction 2')
                return False  # Return False for non-sequential qubit indices.

            # Handle additional qubits not included in the current layer's unitary operations.
            additional_qubits_next = set(qubit_indices) - set(qubit_count)  # Find qubits that need identities.
            for index in additional_qubits_next:
                if len(additional_qubits_next) > 1:
                    print('contradiction 3')  # More than one additional qubit is a contradiction.
                    return False
                elif index < qubit_count[0]:
                    matrix = np.kron(np.eye(2), matrix)  # Add identity to the left for new qubits.
                elif index > qubit_count[-1]:
                    matrix = np.kron(matrix, np.eye(2))  # Add identity to the right for new qubits.
                else:
                    print('contradiction 4')  # Found a contradiction if the index is not in expected order.
                    return False

            # Finally, update the local projection for the current entry.
            current_local_projection = matrix @ previous_projection @ matrix.conj().T
            
            if entry['layer'] == depth-1:
                print('local projection for initial qubit:',initial_qubit_entry, ' involved:', qubit_indices)
                print(current_local_projection)
            else:
                entry['local_projection'] = current_local_projection  # Store the updated projection matrix.
        
        dict_previous = dict_current  # Update dict_previous for the next layer check.

    return dict_current  # Return the updated local projection dictionary.


def print_local_projections_at_depth(local_projection_dict, depth):
    """
    Prints the local projection entries at the specified depth in an elegant format.

    Arguments:
    - local_projection_dict (list): A list of dictionaries containing local projection information.
    - depth (int): The depth for which to print the local projection entries.
    """
    # Filter for entries at the specified depth
    depth_entries = [entry for entry in local_projection_dict if entry['layer'] == depth]

    # Check if there are any entries at the specified depth
    if not depth_entries:
        print(f"No entries found at depth {depth}.")
        return

    # Print the results elegantly
    print(f"Local Projections at Depth {depth}:")
    for entry in depth_entries:
        print(f"Layer: {entry['layer']}, Qubit Indices: {entry['qubit_indices']},\n Local Projection:\n {np.round(entry['local_projection'],10)} \n sss\n{np.round(entry['local_projection']@entry['local_projection'],10)}")


def initialize_local_projection_with_initial_dict(gates_info, initial_dict):
    """
    Initializes a new dictionary for each entry in initial_dict based on gates_info and qubit involvement 
    in forward lightcones, layer by layer.

    Arguments:
    - gates_info (list): List of dictionaries, where each dictionary contains information about a gate, including:
        - 'qubits': List of qubits that the gate acts on.
        - 'layer': The layer in which the gate is applied.
    - initial_dict (list): List of dictionaries, each containing 'qubit_indices' for the initial qubits.

    Returns:
    - new_dict (list): A list of dictionaries, where each dictionary has:
        - 'initial_qubit' (set): The set of 'qubit_indices' from initial_dict.
        - 'layer' (int): The layer index in gates_info.
        - 'qubit_indices' (list): List of qubits involved at that layer.
        - 'local_projection' (int): Placeholder for the projection matrix.
    """
    # Divide gates into layers and retrieve the depth of the circuit
    layers_qc, depth = Manipulate_layers.divide_circuit_into_layers_using_layer_index(gates_info)
    new_dict = []

    # Process each entry in initial_dict
    for entry in initial_dict:
        # Get initial qubits as a set for current entry
        qubit_indices = (entry['qubit_indices'])
        current_local_projection = entry['local_projection']
        
        # Calculate involving qubits layer by layer for the given initial qubits
        involving_qubits_per_layer = Manipulate_layers.extract_involving_qubits_in_forward_lightcone_using_layer_index(layers_qc, qubit_indices)
        
        # Iterate through each layer and add the projection information to new_dict
        for layer, involving_qubits in involving_qubits_per_layer:
            # Check if an entry for this layer and qubit_indices combination already exists
            existing_entry = next(
                (item for item in new_dict if item['layer'] == layer and item['qubit_indices'] == involving_qubits),
                None
            )
            
            # If no existing entry, add a new one
            if existing_entry is None:
                # Initialize the local projection matrix as a placeholder
                if layer == 0:
                    additional_indices = set(involving_qubits) - set(qubit_indices)
                    # print(additional_indices)
                    for additional_index in additional_indices:
                        if additional_index == involving_qubits[-1]:
                            current_local_projection = np.kron(current_local_projection, np.eye(2))
                        if additional_index == involving_qubits[0]:
                            current_local_projection = np.kron(np.eye(2),current_local_projection)
                        else:
                            current_local_projection = current_local_projection
                    projection_matrix=current_local_projection
                    
                else:
                    projection_matrix = -1  # Placeholder for projection matrix

                # Append the new entry to new_dict
                new_dict.append({
                    'layer': layer,
                    'initial_qubit': qubit_indices,
                    'qubit_indices': involving_qubits,
                    'local_projection': projection_matrix
                })

    # Sort by 'layer' and then by the smallest element in 'initial_qubit' for consistency
    new_dict.sort(key=lambda x: (x['layer'], min(x['initial_qubit'])))
    
    return new_dict




def compute_local_projections_fullly_general(gates_info, n_qubits, depth, given_dictionary=None,yes_print=None):
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

    
    # Divide the gates into layers and check the calculated depth against the expected depth.
    gates_layers, depth_check = Manipulate_layers.divide_circuit_into_layers_using_layer_index(gates_info)
    # print(gates_layers)

    if given_dictionary == None:

        initial_dict = initialize_local_projection_dict(gates_layers,n_qubits)
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
                        U_index = np.array(gate['matrix'], dtype=np.complex128)  # Get the unitary matrix for this gate.

                if U_index is None:
                    print("No matching gate found for layer 0 and qubit indices:", qubit_indices)
                    return False  # Handle the case where no gate is found

                # Update the local projection using the unitary matrix and a measurement projection.
                entry['local_projection'] = U_index @ np.array([[1, 0, 0, 0], 
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0]], dtype=np.complex128) @ U_index.conj().T
                
    else:
        initial_dict = initialize_local_projection_with_initial_dict(gates_info, given_dictionary)  
        # print(initial_dict)
    # Sort the local projection dictionary by layer to ensure correct processing order.
    # initial_dict.sort(key=lambda x: x['layer'])


    if depth_check != depth:
        print("depth not matched")
        return False  # Return False if the depths do not match, indicating an inconsistency.
        
    if depth == 0:
        return given_dictionary
    

    # Process each entry in the sorted local projection dictionary.
    for layer_check in range(depth):
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

                # Determine any additional qubits that are present in the current layer but not in the previous layer.
                additional_qubits_previous = set(qubit_indices) - set(previous_layer_entry['qubit_indices'])
                # print(additional_qubits_previous)
                for index in additional_qubits_previous:
                    # Adjust the previous projection for new qubits added.
                    if index < previous_layer_entry['qubit_indices'][0]:
                        previous_projection = np.kron(np.eye(2), previous_projection)  # Add identity for leftmost qubit.
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
                    unitary_matrix = np.array(gate_info['matrix'], dtype=np.complex128)  # Get the unitary matrix.
                    matrix = np.kron(matrix, unitary_matrix)  # Update the matrix with the unitary for this gate.
                    qubit_count =qubit_count + involved_qubits
                    # qubit_count.append(involved_qubits[0])  # Track the first qubit.
                    # qubit_count.append(involved_qubits[1])  # Track the second qubit.
            # print(qubit_count)

            # Check if qubit_count is a sequential natural number array.
            if not is_sequential_natural_numbers(qubit_count):
                # print(qubit_count)
                print('contradiction 2')
                return False  # Return False for non-sequential qubit indices.

            # Handle additional qubits not included in the current layer's unitary operations.
            additional_qubits_next = set(qubit_indices) - set(qubit_count)  # Find qubits that need identities.
            # print(additional_qubits_next)
            # print(qubit_indices, additional_qubits_next,qubit_count)
            for index in additional_qubits_next:
                if len(additional_qubits_next) > 1:
                    print('contradiction 3')  # More than one additional qubit is a contradiction.
                    # return False
                elif index < qubit_count[0]:
                    matrix = np.kron(np.eye(2), matrix)  # Add identity to the left for new qubits.
                elif index > qubit_count[-1]:
                    matrix = np.kron(matrix, np.eye(2))  # Add identity to the right for new qubits.
                else:
                    print('contradiction 4')  # Found a contradiction if the index is not in expected order.
                    return False

            # Finally, update the local projection for the current entry.
            current_local_projection = matrix @ previous_projection @ matrix.conj().T
            entry['local_projection'] = current_local_projection  # Store the updated projection matrix.
        
        dict_previous = dict_current  # Update dict_previous for the next layer check.
    # print(dict_previous)

    return dict_previous  # Return the updated local projection dictionary.

def compute_specific_local_projections(gates_info, n_qubits, depth, given_dictionary=None,yes_print=None, specific_initial_qubit=None):
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
    if specific_initial_qubit == None:
        return False
    if depth == 0:
        return False

    
    # Divide the gates into layers and check the calculated depth against the expected depth.
    gates_layers, depth_check = Manipulate_layers.divide_circuit_into_layers_using_layer_index(gates_info)
    # print(gates_layers)

    if given_dictionary == None:

        initial_dict = initialize_local_projection_dict(gates_layers,n_qubits)
        # print(initial_dict)
        dict_previous = [entry for entry in initial_dict if entry['layer'] == 0]
        for entry in dict_previous:
            layer = entry['layer']  # Extract the current layer from the entry.
            initial_qubit_entry = entry['initial_qubit']
            if initial_qubit_entry != specific_initial_qubit:
                # dict_previous.del
                continue
            qubit_indices = entry['qubit_indices']  # Get the qubit indices for this entry.
            if yes_print == True:
                print('computing local projection for initial qubit:',initial_qubit_entry, ' involved:', qubit_indices)
            if layer == 0:
                # For layer 0, we need to update the local projection using the associated unitary matrix.
                U_index = None  # Initialize U_index to None before searching
                for gate in gates_layers[layer]:
                    # Check if the current gate's qubits match the qubit indices for layer 0.
                    if qubit_indices == gate['qubits']:
                        U_index = np.array(gate['matrix'], dtype=np.complex128)  # Get the unitary matrix for this gate.

                if U_index is None:
                    print("No matching gate found for layer 0 and qubit indices:", qubit_indices)
                    return False  # Handle the case where no gate is found

                # Update the local projection using the unitary matrix and a measurement projection.
                entry['local_projection'] = U_index @ np.array([[1, 0, 0, 0], 
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0],
                                                             [0, 0, 0, 0]], dtype=np.complex128) @ U_index.conj().T
                
    else:
        initial_dict = initialize_local_projection_with_initial_dict(gates_info, given_dictionary)  
        # print(initial_dict)
    # Sort the local projection dictionary by layer to ensure correct processing order.
    # initial_dict.sort(key=lambda x: x['layer'])


    if depth_check != depth:
        return False  # Return False if the depths do not match, indicating an inconsistency.
        
    if depth == 0:
        return given_dictionary
    

    # Process each entry in the sorted local projection dictionary.
    for layer_check in range(depth):
        # print('layer = ', layer_check)
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
            if initial_qubit_entry != specific_initial_qubit:
                continue
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

                # Determine any additional qubits that are present in the current layer but not in the previous layer.
                additional_qubits_previous = set(qubit_indices) - set(previous_layer_entry['qubit_indices'])
                # print(additional_qubits_previous)
                for index in additional_qubits_previous:
                    # Adjust the previous projection for new qubits added.
                    if index < previous_layer_entry['qubit_indices'][0]:
                        previous_projection = np.kron(np.eye(2), previous_projection)  # Add identity for leftmost qubit.
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
                    unitary_matrix = np.array(gate_info['matrix'], dtype=np.complex128)  # Get the unitary matrix.
                    matrix = np.kron(matrix, unitary_matrix)  # Update the matrix with the unitary for this gate.
                    qubit_count.append(involved_qubits[0])  # Track the first qubit.
                    qubit_count.append(involved_qubits[1])  # Track the second qubit.
            # print(qubit_count)

            # Check if qubit_count is a sequential natural number array.
            if not is_sequential_natural_numbers(qubit_count):
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
                    matrix = np.kron(np.eye(2), matrix)  # Add identity to the left for new qubits.
                elif index > qubit_count[-1]:
                    matrix = np.kron(matrix, np.eye(2))  # Add identity to the right for new qubits.
                else:
                    print('contradiction 4')  # Found a contradiction if the index is not in expected order.
                    return False

            # Finally, update the local projection for the current entry.
            current_local_projection = matrix @ previous_projection @ matrix.conj().T
            entry['local_projection'] = current_local_projection  # Store the updated projection matrix.
        
        dict_previous = dict_current  # Update dict_previous for the next layer check.
    # print(dict_previous)

    return current_local_projection  # Return the updated local projection dictionary.


def embed_unitary_matrix(gate_matrix, target_qubits,n_qubits ):
    """
    Compute the full matrix representation of a quantum gate acting on specific qubits.
    
    Parameters:
    -----------
    n_qubits : int
        Total number of qubits in the circuit
    gate_matrix : numpy.ndarray
        Matrix representation of the gate to be applied
    target_qubits : list
        List of target qubit indices where the gate acts
        
    Returns:
    --------
    numpy.ndarray
        Full matrix representation of the gate in the n-qubit space
    """
    
    # Check input validity
    if not isinstance(gate_matrix, np.ndarray):
        raise TypeError("gate_matrix must be a numpy array")
        
    # Verify gate_matrix dimensions match number of target qubits
    gate_size = int(np.log2(gate_matrix.shape[0]))
    if len(target_qubits) != gate_size:
        raise ValueError(f"Gate matrix dimension ({gate_matrix.shape}) doesn't match number of target qubits ({len(target_qubits)})")
    
    # Sort target qubits in ascending order
    target_qubits = sorted(target_qubits)
    
    # Initialize the matrices for each qubit
    matrices = []
    current_target_idx = 0
    
    # Identity matrix for single qubit
    I = np.eye(2)
    
    # Build the list of matrices to tensor together
    for i in range(n_qubits):
        if current_target_idx < len(target_qubits) and i == target_qubits[current_target_idx]:
            # If this qubit is a target, we'll need to handle it specially later
            matrices.append(None)
            current_target_idx += 1
        else:
            # For non-target qubits, we use identity matrices
            matrices.append(I)
    
    # Function to get binary representation of a number
    def get_binary_array(num, width):
        return [int(x) for x in format(num, f'0{width}b')]
    
    # Initialize the full matrix
    dim = 2 ** n_qubits
    full_matrix = np.zeros((dim, dim), dtype=complex)
    
    # Fill the matrix
    for i in range(dim):
        # Get binary representation of input state
        input_state = get_binary_array(i, n_qubits)
        
        # Extract relevant bits for gate application
        gate_input_idx = 0
        for target in target_qubits:
            gate_input_idx = (gate_input_idx << 1) | input_state[target]
        
        # Apply gate
        gate_output = gate_matrix[gate_input_idx]
        
        # Generate all possible output states and their amplitudes
        for gate_output_idx in range(len(gate_output)):
            if abs(gate_output[gate_output_idx]) > 1e-10:  # Ignore very small amplitudes
                # Convert gate output to binary and place in correct positions
                output_state = input_state.copy()
                gate_bits = get_binary_array(gate_output_idx, len(target_qubits))
                for target_idx, target in enumerate(target_qubits):
                    output_state[target] = gate_bits[target_idx]
                
                # Convert binary output state to decimal index
                output_idx = int(''.join(map(str, output_state)), 2)
                
                # Set matrix element
                full_matrix[output_idx, i] = gate_output[gate_output_idx]
    
    return full_matrix


def compute_local_projections_with_Choi_isomorphism(gates_info, n_qubits, depth, given_dictionary=None,yes_print=None):
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

        initial_dict = initialize_local_projection_dict(gates_layers,n_qubits)
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
                        U_index = embed_unitary_matrix(np.array(gate['matrix'], dtype=np.complex128),[1,3],4)   # Get the unitary matrix for this gate.

                if U_index is None:
                    print("No matching gate found for layer 0 and qubit indices:", qubit_indices)
                    return False  # Handle the case where no gate is found
                EPR_vector = 1/np.sqrt(2) * np.array([1, 0, 0, 1], dtype=np.complex128)
                EPR_projector = 1/2 * np.array([[1, 0, 0, 1], [0, 0, 0, 0],[0, 0, 0, 0],[1, 0, 0, 1]], dtype=np.complex128)
                EPR_projector_2 = np.kron(EPR_projector,EPR_projector)
                # print(EPR2)

                # Update the local projection using the unitary matrix and a measurement projection.
                entry['local_projection'] = U_index @ EPR_projector_2 @ U_index.conj().T
                
    else:
        initial_dict = initialize_local_projection_with_initial_dict(gates_info, given_dictionary)  
        # print(initial_dict)
    # Sort the local projection dictionary by layer to ensure correct processing order.
    # initial_dict.sort(key=lambda x: x['layer'])


    if depth_check != depth:
        return False  # Return False if the depths do not match, indicating an inconsistency.
        
    if depth == 0:
        return given_dictionary
    

    # Process each entry in the sorted local projection dictionary.
    for layer_check in range(depth):
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

                # Determine any additional qubits that are present in the current layer but not in the previous layer.
                additional_qubits_previous = set(qubit_indices) - set(previous_layer_entry['qubit_indices'])
                # print(additional_qubits_previous)
                for index in additional_qubits_previous:
                    # Adjust the previous projection for new qubits added.
                    if index < previous_layer_entry['qubit_indices'][0]:
                        previous_projection = np.kron(np.eye(4), previous_projection)  # Add identity for leftmost qubit.
                    elif index > previous_layer_entry['qubit_indices'][-1]:
                        previous_projection = np.kron(previous_projection, np.eye(4))  # Add identity for rightmost qubit.
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
                    unitary_matrix = embed_unitary_matrix(np.array(gate_info['matrix'], dtype=np.complex128),[1,3],4)  # Get the unitary matrix.
                    matrix = np.kron(matrix, unitary_matrix)  # Update the matrix with the unitary for this gate.
                    qubit_count = qubit_count + involved_qubits
                    # qubit_count.append(involved_qubits[0])  # Track the first qubit.
                    # qubit_count.append(involved_qubits[1])  # Track the second qubit.
            # print(qubit_count)

            # Check if qubit_count is a sequential natural number array.
            if not is_sequential_natural_numbers(qubit_count):
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
                    matrix = np.kron(np.eye(4), matrix)  # Add identity to the left for new qubits.
                elif index > qubit_count[-1]:
                    matrix = np.kron(matrix, np.eye(4))  # Add identity to the right for new qubits.
                else:
                    print('contradiction 4')  # Found a contradiction if the index is not in expected order.
                    return False

            # Finally, update the local projection for the current entry.
            current_local_projection = matrix @ previous_projection @ matrix.conj().T
            entry['local_projection'] = current_local_projection  # Store the updated projection matrix.
        
        dict_previous = dict_current  # Update dict_previous for the next layer check.

    return dict_previous  # Return the updated local projection dictionary.





#####---------------------
 ### BENCHMARKING FUNCTIONS



def benchmark_statevector_vs_local_projection(
        file_name="comparison_statevector_vs_local_projection_computation.txt", 
        min_qubits_lp=2, max_qubits_lp=2, 
        min_qubits_sv=2, max_qubits_sv=2, 
        depth=2, n_patterns=2):
    """
    Benchmarks the running time of statevector computation and local projection methods.

    Arguments:
    - file_name (str): Name of the output file to save results.
    - min_qubits_lp (int): Minimum number of qubits to test for local projection (must be even).
    - max_qubits_lp (int): Maximum number of qubits to test for local projection (must be even).
    - min_qubits_sv (int): Minimum number of qubits to test for statevector computation (must be even).
    - max_qubits_sv (int): Maximum number of qubits to test for statevector computation (must be even).
    - depth (int): Depth of the random circuits.
    - num_patterns (int): Number of random patterns per qubit count for averaging.

    This function will run `num_patterns` random circuits for each qubit count and record average times with error bars.
    """
    # Ensure min and max qubits are even for both local projection and statevector tasks
    if min_qubits_lp % 2 != 0: min_qubits_lp += 1
    if max_qubits_lp % 2 != 0: max_qubits_lp -= 1
    if min_qubits_sv % 2 != 0: min_qubits_sv += 1
    if max_qubits_sv % 2 != 0: max_qubits_sv -= 1
    
    results = []

    indice = [index for index in range(min_qubits_lp, max_qubits_lp + 1, 10)]
    # indice.append(100)
    # indice.append(200)
    print("Number of qubits for local projection:",indice)

    # Benchmark Local Projection computation across its range of qubits
    for n_qubits in indice:
        lp_times = []
        inconsistencies = 0

        for _ in range(n_patterns):
            qc, qc_info = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)
            
            # Measure running time for local projection
            start_time_lp = time.time()
            local_projection = compute_local_projections_fullly_general(qc_info, n_qubits, depth, yes_print=False)
            end_time_lp = time.time()
            lp_times.append(end_time_lp - start_time_lp)

        avg_lp_time = np.mean(lp_times)
        std_lp_time = np.std(lp_times)
        results.append((n_qubits, avg_lp_time, std_lp_time, None, None, inconsistencies))

        print(f"{n_qubits} qubits for Local Projection: {avg_lp_time:.6f}s ± {std_lp_time:.6f}s")

    # Benchmark Statevector computation across its range of qubits
    print("Number of qubits for Statevector:", list(range(min_qubits_sv, max_qubits_sv + 1, 2)))
    for n_qubits in range(min_qubits_sv, max_qubits_sv + 1, 2):
        sv_times = []
        inconsistencies = 0

        for _ in range(n_patterns):
            qc, qc_info = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)

            combined_qc = QuantumCircuit(n_qubits)
            combined_qc.append(qc.to_instruction(), range(n_qubits))
            simulator = AerSimulator(method='statevector')
            transpiled_circuit = transpile(combined_qc, simulator)
            transpiled_circuit.save_statevector()
            
            start_time_sv = time.time()
            result = simulator.run(transpiled_circuit).result()
            statevector = result.get_statevector()
            end_time_sv = time.time()
            sv_times.append(end_time_sv - start_time_sv)

        avg_sv_time = np.mean(sv_times)
        std_sv_time = np.std(sv_times)
        results.append((n_qubits, None, None, avg_sv_time, std_sv_time, inconsistencies))

        print(f"{n_qubits} qubits for Statevector: {avg_sv_time:.6f}s ± {std_sv_time:.6f}s")

    # Write all results to the file
    with open(file_name, 'w') as f:
        for n, avg_lp, std_lp, avg_sv, std_sv, incons in results:
            f.write(f"{n}\t{avg_lp}\t{std_lp}\t{avg_sv}\t{std_sv}\t{incons}\n")



def plot_statevector_vs_local_projection(file_name="comparison_statevector_vs_local_projection_computation_fixed_depth.txt", 
                                         output_file_name="Running_time_Statevector_vs_Local_Projection_fixed_depth.pdf",
                                         n_patterns=100):
    """
    Reads benchmark results from a file and plots running time with error bars for each method.
    
    Arguments:
    - file_name (str): Name of the file containing the benchmark results.
    - output_file_name (str): Name of the output file for the plot.
    - n_patterns (int): Number of patterns used for averaging.
    """
    n_qubits = []
    lp_times = []
    lp_errors = []
    sv_times = []
    sv_errors = []

    # Read data from the file, handling `None` values for missing data
    with open(file_name, 'r') as f:
        for line in f:
            data = line.strip().split("\t")
            print(data)
            n_qubit = int(data[0])
            lp_time = float(data[1]) if data[1] != 'None' else None
            lp_error = float(data[2]) if data[2] != 'None' else None
            sv_time = float(data[3]) if data[3] != 'None' else None
            sv_error = float(data[4]) if data[4] != 'None' else None
            
            n_qubits.append(n_qubit)
            lp_times.append(lp_time)
            lp_errors.append(lp_error)
            sv_times.append(sv_time)
            sv_errors.append(sv_error)

    # Plot Local Projection data points (skipping `None` values)
    plt.figure(figsize=(10, 10))
    if any(lp_times):  # Check if there's data to plot for Local Projection
        plt.errorbar(
            [n for n, t in zip(n_qubits, lp_times) if t is not None],
            [t for t in lp_times if t is not None],
            yerr=[e for e in lp_errors if e is not None],
            fmt='o-', capsize=5, label="Local Projection",color = 'blue'
        )

    # Plot Statevector data points (skipping `None` values)
    if any(sv_times):  # Check if there's data to plot for Statevector
        plt.errorbar(
            [n for n, t in zip(n_qubits, sv_times) if t is not None],
            [t for t in sv_times if t is not None],
            yerr=[e for e in sv_errors if e is not None],
            fmt='s-', capsize=5, label="Statevector",color='red'
        )

    # Labeling and visual details
    plt.xlabel("Number of Qubits", fontsize=22)
    plt.ylabel("Average Running Time (seconds)", fontsize=22)
    plt.title(f"Running Time at Depth 6 for {n_patterns} Haar Random Circuits", fontsize=22)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.savefig(output_file_name, dpi=400)
    plt.show()




# Benchmarking function with varying depth
def benchmark_statevector_vs_local_projection_by_depth(
        file_name="comparison_statevector_vs_local_projection_fixed_n_qubits.txt",
        n_qubits=20, min_depth=1, max_depth=4, n_patterns=1):
    """
    Benchmarks the running time of statevector computation and local projection methods for varying circuit depths.
    
    Arguments:
    - file_name (str): Name of the output file to save results.
    - n_qubits (int): Number of qubits to test (must be even).
    - min_depth (int): Minimum circuit depth to test.
    - max_depth (int): Maximum circuit depth to test.
    - num_patterns (int): Number of random patterns per depth for averaging.

    This function will run `num_patterns` random circuits for each depth and record average times with error bars.
    """
    if n_qubits % 2 != 0:
        raise ValueError("Number of qubits must be even.")
    
    results = []
    print(f"Number of qubits for both methods: {n_qubits}")
    print(f"Depth range: {min_depth} to {max_depth}")
    for depth in range(min_depth, max_depth + 1):
        lp_times = []
        sv_times = []
        inconsistencies = 0

        for _ in range(n_patterns):
            # Generate a random Haar circuit
            qc, qc_info = Create_quantum_circuit.create_random_haar_circuit(n_qubits, depth)
            
            # Measure running time for local projection
            start_time_lp = time.time()
            local_projection = compute_local_projections_fullly_general(qc_info, n_qubits, depth, yes_print=False)
            end_time_lp = time.time()
            lp_times.append(end_time_lp - start_time_lp)

            # Measure running time for statevector computation
            combined_qc = QuantumCircuit(n_qubits)
            combined_qc.append(qc.to_instruction(), range(n_qubits))
            simulator = AerSimulator(method='statevector')
            transpiled_circuit = transpile(combined_qc, simulator)
            transpiled_circuit.save_statevector()
            
            start_time_sv = time.time()
            result = simulator.run(transpiled_circuit).result()
            statevector = result.get_statevector()
            end_time_sv = time.time()
            sv_times.append(end_time_sv - start_time_sv)

        # Calculate average times and standard deviations
        avg_lp_time = np.mean(lp_times)
        std_lp_time = np.std(lp_times)
        avg_sv_time = np.mean(sv_times)
        std_sv_time = np.std(sv_times)
        results.append((depth, avg_lp_time, std_lp_time, avg_sv_time, std_sv_time, inconsistencies))

        # Write results to the file
        with open(file_name, 'w') as f:
            for d, avg_lp, std_lp, avg_sv, std_sv, incons in results:
                f.write(f"{d}\t{avg_lp}\t{std_lp}\t{avg_sv}\t{std_sv}\t{incons}\n")
        print(f"Depth {depth}: Local Projection = {avg_lp_time:.6f}s ± {std_lp_time:.6f}s, "
              f"Statevector = {avg_sv_time:.6f}s ± {std_sv_time:.6f}s, Inconsistencies = {inconsistencies}")

# Plotting function
def plot_statevector_vs_local_projection_by_depth(file_name="comparison_statevector_vs_local_projection_fixed_n_qubits.txt",
                                                  output_file_name="Running_time_Statevector_vs_Local_Projection_fixed_n_qubits.pdf",
                                                  num_patterns=100, n_qubits=20):
    """
    Reads benchmark results from a file and plots running time with error bars for each method.
    
    Arguments:
    - file_name (str): Name of the file containing the benchmark results.
    """
    depths = []
    lp_times = []
    lp_errors = []
    sv_times = []
    sv_errors = []

    with open(file_name, 'r') as f:
        for line in f:
            data = line.strip().split("\t")
            depths.append(int(data[0]))
            lp_times.append(float(data[1]))
            lp_errors.append(float(data[2]))
            sv_times.append(float(data[3]))
            sv_errors.append(float(data[4]))

    plt.figure(figsize=(10, 10))
    plt.errorbar(depths, lp_times, yerr=lp_errors, fmt='o-', capsize=15, label="Local Projection", color = 'blue')
    plt.errorbar(depths, sv_times, yerr=sv_errors, fmt='s-', capsize=15, label="Statevector",color = 'red')
    plt.xlabel("Circuit Depth", fontsize=22)
    plt.ylabel("Average Running Time (seconds)", fontsize=22)
    plt.title(f"Running Time for {num_patterns} Haar Random Circuits of {n_qubits} qubits", fontsize=22)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xticks(np.arange(int(min(depths)), int(max(depths)) + 1, 1))
    plt.grid(True)
    plt.savefig(output_file_name, dpi=400)
    plt.show()