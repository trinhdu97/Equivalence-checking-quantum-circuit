from qiskit_ibm_runtime import EstimatorV2 as EstimatorV2
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
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
import Create_quantum_circuit
import local_projection_computation
import Manipulate_layers


from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

def map_2d_to_1d(x, y, L):
    """Map 2D coordinates (x, y) to 1D index."""
    return x * L + y
def map_1d_to_2d(index, L):
    """Map 1D index to 2D coordinates (x, y)."""
    x = index // L
    y = index % L
    return x, y

def create_2d_circuit(L, depth):
    if L % 2 != 0:
        raise ValueError("L must be an even number for 2D circuit construction.")
    
    gates_info = []
    qc = QuantumCircuit(L * L)

    for layer in range(depth):
        if layer % 4 == 0:
            for i in range(L):
                for j in range(L//2):
                    unitary_matrix = random_unitary(4).data
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(i, 2*j, L), map_2d_to_1d(i, 2*j+1, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer
        elif layer % 4 == 1:
            for i in range(L//2):
                for j in range(L):
                    unitary_matrix = random_unitary(4).data
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(2*i, j, L), map_2d_to_1d(2*i+1, j, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer
        elif layer % 4 == 2:
            for i in range(L):
                for j in range(L//2-1):
                    unitary_matrix = random_unitary(4).data
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(i, 2*j+1, L), map_2d_to_1d(i, 2*j+2, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer          
        elif layer % 4 == 3:
            for i in range(L//2-1):
                for j in range(L):
                    unitary_matrix = random_unitary(4).data
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(2*i+1, j, L), map_2d_to_1d(2*i+2, j, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer  
    return qc, gates_info

def create_2d_circuit_identity(L, depth):
    if L % 2 != 0:
        raise ValueError("L must be an even number for 2D circuit construction.")
    
    gates_info = []
    qc = QuantumCircuit(L * L)

    for layer in range(depth):
        if layer % 4 == 0:
            for i in range(L):
                for j in range(L//2):
                    unitary_matrix = np.eye(4)
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(i, 2*j, L), map_2d_to_1d(i, 2*j+1, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer
        elif layer % 4 == 1:
            for i in range(L//2):
                for j in range(L):
                    unitary_matrix = np.eye(4)#random_unitary(4).data
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(2*i, j, L), map_2d_to_1d(2*i+1, j, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer
        elif layer % 4 == 2:
            for i in range(L):
                for j in range(L//2-1):
                    unitary_matrix = np.eye(4)#random_unitary(4).data
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(i, 2*j+1, L), map_2d_to_1d(i, 2*j+2, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer          
        elif layer % 4 == 3:
            for i in range(L//2-1):
                for j in range(L):
                    unitary_matrix = np.eye(4)#random_unitary(4).data
                    qc, gates_info = Create_quantum_circuit.add_gate(
                        'unitary',
                        [map_2d_to_1d(2*i+1, j, L), map_2d_to_1d(2*i+2, j, L)],
                        gates_info,
                        qc,
                        unitary_matrix
                    )
                    gates_info[-1]['layer'] = layer  
    return qc, gates_info

def visualize_2d_circuit_layout(gates_info, L, depth):
    fig, axes = plt.subplots(1, depth, figsize=(4 * depth, 4))
    if depth == 1:
        axes = [axes]

    for layer in range(depth):
        ax = axes[layer]
        ax.set_title(f'Layer {layer}')
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(-0.5, L - 0.5)
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_aspect('equal')
        ax.grid(True)

        for gate in gates_info:
            if gate['layer'] == layer:
                q1, q2 = gate['qubits']
                x1, y1 = map_1d_to_2d(q1, L)
                x2, y2 = map_1d_to_2d(q2, L)
                ax.plot([y1, y2], [x1, x2], 'ro-')

    plt.tight_layout()
    plt.show()



def light_cone_2d_circuit(gates_info, L, depth):
    """
    Show the light cone of the qubits in the 2D circuit based on the gates_info.
    """
    light_cones = np.zeros((L*L,depth, L*L), dtype=int)   # L*L qubits, depth, L*L light cone states    
    # light_cones[q, d, k] = 1 means: qubit `q` is influenced by qubit `k` at layer `d`
    for layer in range(depth):
        if layer != 0:
            light_cones[:,layer,:] = light_cones[:,layer-1,:].copy()
        # Initialize the light cone for the current layer
        for gate in gates_info:
            if gate['layer'] == layer:
                q1, q2 = gate['qubits']
                x1, y1 = map_1d_to_2d(q1, L)
                x2, y2 = map_1d_to_2d(q2, L)
                if layer == 0:
                    light_cones[q1, layer, q1] = 1
                    light_cones[q1, layer, q2] = 1
                    light_cones[q2, layer, q1] = 1
                    light_cones[q2, layer, q2] = 1
                else:
                    for qubit in range(L*L):
                        if light_cones[qubit,layer-1, q1] == 1 or light_cones[qubit, layer-1, q2] == 1:
                            light_cones[qubit, layer, q1] = 1
                            light_cones[qubit, layer, q2] = 1
                # Update the light cone for the qubits involved in the gate
    return light_cones


def light_cone_arbitrary_connectivity_circuit(gates_info, N, depth):
    """
    Show the light cone of the qubits in the 2D circuit based on the gates_info.
    """
    light_cones = np.zeros((N, depth, N), dtype=int)   # N qubits, depth, N light cone states
    # light_cones[q, d, k] = 1 means: qubit `q` is influenced by qubit `k` at layer `d`
    for layer in range(depth):
        if layer != 0:
            light_cones[:,layer,:] = light_cones[:,layer-1,:].copy()
        # Initialize the light cone for the current layer
        for gate in gates_info:
            if gate['layer'] == layer:
                q1, q2 = gate['qubits']
                if layer == 0:
                    light_cones[q1, layer, q1] = 1
                    light_cones[q1, layer, q2] = 1
                    light_cones[q2, layer, q1] = 1
                    light_cones[q2, layer, q2] = 1
                else:
                    for qubit in range(N):
                        if light_cones[qubit,layer-1, q1] == 1 or light_cones[qubit, layer-1, q2] == 1:
                            light_cones[qubit, layer, q1] = 1
                            light_cones[qubit, layer, q2] = 1
                # Update the light cone for the qubits involved in the gate
    return light_cones


def reduce_light_cone(light_cones, N):
    """
    Reduce the light cone array by removing duplicate cones across qubits.

    Arguments:
    - light_cones: numpy array of shape (N, depth, N)
    - L: lattice size (L x L)

    Returns:
    - reduced_light_cones: numpy array of unique light cones, shape (num_unique, depth, N)
    """
    N, D, _ = light_cones.shape  # N qubits, depth, N light cone states

    # Reshape each light cone to a flat vector of length D * N
    flattened = light_cones.reshape(N, -1)

    # Use numpy's unique to efficiently remove duplicates
    unique_flattened = np.unique(flattened, axis=0)

    # Reshape back to 3D: (num_unique, depth, N)
    reduced_light_cones = unique_flattened.reshape(-1, D, N)

    return reduced_light_cones



def reduce_light_cone_dictionary(light_cones):
    """
    Reduce light cones by removing duplicates and grouping qubits with identical cones.

    Returns:
    - grouped_cones: list of dicts, each with:
        - 'qubits': list of qubit indices
        - 'light_cone': numpy array of shape (depth, N)
    """
    N, D, _ = light_cones.shape
    flattened = light_cones.reshape(N, -1)

    cone_map = {}
    grouped_cones = []

    for qubit in range(N):
        cone_key = tuple(flattened[qubit])  # hashable representation
        if cone_key not in cone_map:
            # First time seeing this light cone → create new group
            cone_map[cone_key] = len(grouped_cones)
            grouped_cones.append({
                'qubits': [qubit],
                'light_cone': light_cones[qubit].copy(),
                'local_projection': -1
            })
        else:
            # Add qubit to existing group
            group_id = cone_map[cone_key]
            grouped_cones[group_id]['qubits'].append(qubit)
    # print(cone_map)

    return grouped_cones

    
def plot_light_cone_matrix(light_cones, layer, L):
    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.grid(True)
    ax.set_ylim(-1, L*L)
    ax.set_xlim(-1, L*L)
    ax.set_xticks(range(0, L*L, 1))
    ax.set_yticks(range(0, L*L, 1))
    # ax.set_aspect('equal')
    # Create a grid for the light cone matrix
    # light_cones is a 3D numpy array of shape (L*L, depth, L*L)
    # Plot the light cone matrix for the specified layer
    # The light cone matrix is a 2D slice of the 3D array at the specified layer
    # The x-axis represents the influencing qubit, and the y-axis represents the affected qubit
    ax.imshow(light_cones[:, layer, :], cmap='Greys')
    ax.set_title(f"Light Cone Matrix at Layer {layer}")
    ax.set_xlabel("Influencing Qubit")
    ax.set_ylabel("Affected Qubit")
    plt.show()

def plot_light_cone_points(light_cones, layer, L):
    """
    Plot the light cone as a set of discrete points (not filled squares) for a given layer.

    Each point (i, j) represents that qubit `i` is influenced by qubit `j` at this layer.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract the binary matrix for the given layer
    data = light_cones[:, layer, :]  # Shape: (L*L, L*L)

    # Get coordinates of entries with value 1
    rows, cols = np.where(data == 1)

    # Scatter plot for all '1' entries
    ax.scatter(cols, rows, c='black', s=10)  # (x=columns, y=rows)

    # Formatting
    ax.set_title(f"Light Cone Points at Layer {layer}")
    ax.set_xlabel("Influencing Qubit Index")
    ax.set_ylabel("Affected Qubit Index")
    ax.set_xticks(range(0, L*L, max(1, L)))
    ax.set_yticks(range(0, L*L, max(1, L)))
    ax.set_xlim(-1, L*L)
    ax.set_ylim(-1, L*L)
    ax.set_aspect('equal')
    ax.grid(True, color='lightgray', linewidth=0.5)
    plt.tight_layout()
    plt.show()



def get_subset_indices(subset, big_set):
    return [big_set.index(item) for item in subset]

def compute_local_projection_2d_circuit(N, depth, gates_info):
    """
    Compute local projection matrices for groups of qubits sharing the same light cone
    in a 2D circuit with arbitrary connectivity.

    Parameters:
    - N: Total number of qubits
    - depth: Circuit depth
    - gates_info: List of gates with associated qubit indices and matrices

    Returns:
    - A list of reduced light cone groups with updated 'local_projection_matrix' for each group
    """
    def get_subset_indices(subset, big_set):
        return [big_set.index(item) for item in subset]
    # Step 1: Compute the light cone for each qubit
    light_cones = light_cone_arbitrary_connectivity_circuit(gates_info, N, depth)

    # Step 2: Group qubits that share the same light cone structure
    reduce_light_cones = reduce_light_cone_dictionary(light_cones)

    # Step 3: Organize the gates into circuit layers
    layers, depth = Manipulate_layers.divide_circuit_into_layers_using_layer_index(gates_info)

    # print("Layers divided:", layers)

    # Step 4: Process each group of qubits with identical light cone
    for group_index, light_cone_group in enumerate(reduce_light_cones):
        # print("----------------")
        # print("Light cone group:", group_index)
        # print("Qubits in light cone group:", light_cone_group['qubits'])

        # Step 5: Iterate through each layer in depth
        for i in range(depth):
            # print("Layer", i)
            cone = light_cone_group['light_cone']

            if i == 0:
                # First layer: initialize projection matrix to |00...0⟩⟨00...0|
                # print(cone[i])
                qubits_in_current_layer = sorted(set(np.where(cone[i] == 1)[0]))
                previous_local_projection_matrix = np.zeros(
                    (2**len(qubits_in_current_layer), 2**len(qubits_in_current_layer)), dtype=complex)
                previous_local_projection_matrix[0, 0] = 1  # |00⟩⟨00|

                # print("Qubits in current layer:", qubits_in_current_layer)

                # Apply the first valid gate in the layer (for efficiency)
                for gates in layers[i]:
                    # print("Gate in layer", i, ":", gates['qubits'])

                    # Sanity check: gate must be fully within the current light cone
                    if set(gates['qubits']).intersection(qubits_in_current_layer) and not set(gates['qubits']).issubset(qubits_in_current_layer):
                        print("Error: gate intersects but is not fully inside the cone")
                        return False

                    if set(gates['qubits']).issubset(qubits_in_current_layer):
                        gate_index_in_current_layer = get_subset_indices(gates['qubits'], qubits_in_current_layer)
                        gate_full_matrix = embed_unitary_matrix(
                            operator_matrix=np.array(gates['matrix']),
                            target_qubits=gate_index_in_current_layer,
                            n_qubits=len(qubits_in_current_layer))
                        # print("current unitary: ", gate_full_matrix)
                        # print("dsdsdd")
                        # print("original gate: ", gates['matrix'])
                        break  # Only one gate applied for now

                # Update the projection matrix
                current_local_projection_matrix = gate_full_matrix @ previous_local_projection_matrix @ np.conjugate(np.transpose(gate_full_matrix))

            else:
                # print(cone[i])
                # For subsequent layers, update light cone and embed previous projection
                qubit_in_previous_layer = qubits_in_current_layer.copy()
                qubits_in_current_layer = sorted(set(np.where(cone[i] == 1)[0]))
                index_in_current_layer = get_subset_indices(qubit_in_previous_layer, qubits_in_current_layer)

                # Embed previous projection matrix into new (larger) space
                previous_local_projection_matrix = embed_unitary_matrix(
                    current_local_projection_matrix, index_in_current_layer, len(qubits_in_current_layer))

                # print("Qubits in previous layer:", qubit_in_previous_layer)
                # print("Qubits in current layer:", qubits_in_current_layer)
                # print("Index in current layer:", index_in_current_layer)

                count = 0  # Count valid gates in the current light cone
                for gates in layers[i]:
                    if set(gates['qubits']).intersection(qubits_in_current_layer) and not set(gates['qubits']).issubset(qubits_in_current_layer):
                        print("Error: gate intersects but is not fully inside the cone")
                        return False

                    if set(gates['qubits']).issubset(qubits_in_current_layer):
                        count += 1
                        # print("Gate in layer", i, ":", gates['qubits'])

                        if count == 1:
                            # Initialize unitary with the first gate
                            current_embedded_qubits = sorted(set(gates['qubits']))
                            current_unitary = embed_unitary_matrix(
                                operator_matrix=np.array(gates['matrix']),
                                target_qubits=get_subset_indices(gates['qubits'], current_embedded_qubits),
                                n_qubits=len(current_embedded_qubits))
                        else:
                            # Merge and embed multiple gates into one operator
                            previous_embedded_qubits = current_embedded_qubits.copy()
                            current_embedded_qubits.extend(gates['qubits'])
                            current_embedded_qubits = sorted(set(current_embedded_qubits))

                            current_unitary = embed_unitary_matrix(
                                operator_matrix=current_unitary,
                                target_qubits=get_subset_indices(previous_embedded_qubits, current_embedded_qubits),
                                n_qubits=len(current_embedded_qubits))

                            gate_full_matrix = embed_unitary_matrix(
                                operator_matrix=np.array(gates['matrix']),
                                target_qubits=get_subset_indices(gates['qubits'], current_embedded_qubits),
                                n_qubits=len(current_embedded_qubits))

                            current_unitary = gate_full_matrix @ current_unitary
                        # print("current unitary: ", current_unitary)
                        # print("current_embbeded:", current_embedded_qubits)

                # print("count:", count)

                # Embed the composed unitary into the full light cone
                current_unitary = embed_unitary_matrix(
                    operator_matrix=current_unitary,
                    target_qubits=get_subset_indices(current_embedded_qubits, qubits_in_current_layer),
                    n_qubits=len(qubits_in_current_layer))
                # print("current unitary: ", current_unitary)
                # print("Final embedding for this layer:", current_embedded_qubits, qubits_in_current_layer)

                # Update the local projection matrix
                # print("Previous:", previous_local_projection_matrix)
                current_local_projection_matrix = current_unitary @ previous_local_projection_matrix @ np.conjugate(np.transpose(current_unitary))

        # Assign the resulting matrix back to the group
        light_cone_group['local_projection'] = current_local_projection_matrix

    return reduce_light_cones
# def plot_light_cone_2d(light_cones, layer, target_qubit, L):
#     """
#     Plot the 2D light cone of a target qubit at a given layer.

#     Arguments:
#     - light_cones: numpy array of shape (L*L, depth, L*L)
#     - layer: layer index (int)
#     - target_qubit: index of the target qubit (int)
#     - L: lattice size (int)
#     """
#     influence_map = light_cones[target_qubit, layer, :].reshape(L, L)

#     fig, ax = plt.subplots()
#     im = ax.imshow(influence_map, cmap='Greys', origin='upper')

#     x, y = map_1d_to_2d(target_qubit, L)
#     ax.plot(y, x, 'ro')  # Highlight the target qubit in red
#     ax.set_title(f"2D Light Cone of Qubit {target_qubit} at Layer {layer}")
#     ax.set_xticks(range(L))
#     ax.set_yticks(range(L))
#     ax.set_xlabel("Column")
#     ax.set_ylabel("Row")
#     ax.grid(True)
#     plt.colorbar(im, ax=ax, label='Influence (0 or 1)')
#     plt.show()


# def plot_light_cone_matrix(light_cones, layer, L):
#     """
#     Plot the full light cone matrix with correctly aligned pixel squares.
#     """
#     data = light_cones[:, layer, :]
#     fig, ax = plt.subplots(figsize=(8, 8))

#     im = ax.imshow(data, 
#                    cmap='Greys', 
#                    interpolation='none',
#                    origin='upper',
#                    extent=[-0.5, L*L - 0.5, L*L - 0.5, -0.5])

#     ax.set_title(f"Light Cone Matrix at Layer {layer}")
#     ax.set_xlabel("Influencing Qubit Index")
#     ax.set_ylabel("Affected Qubit Index")
#     ax.set_xticks(range(0, L*L, L))
#     ax.set_yticks(range(0, L*L, L))
#     ax.grid(True, color='lightgray', linewidth=0.5)

#     plt.colorbar(im, ax=ax, label='Influence (0 or 1)')
#     plt.tight_layout()
#     plt.show()

def state_vector_simulation(qc, backend_name='aer_simulator'):
    """
    Simulate the statevector of a quantum circuit using Qiskit's Aer simulator.

    Parameters:
    - qc (QuantumCircuit): The quantum circuit to simulate.
    - backend_name (str): Backend name, typically 'aer_simulator' or 'statevector_simulator'.

    Returns:
    - statevector (Statevector): The final statevector of the quantum circuit.
    """
    simulator = AerSimulator(method='statevector')
    transpiled_circuit = transpile(qc, simulator)

    # Add a save_statevector instruction at the end of the transpiled circuit
    transpiled_circuit.save_statevector()

    # Run the circuit on the simulator and obtain the result (statevector)
    result = simulator.run(transpiled_circuit).result()
    statevector = result.get_statevector()

    # Ensure the circuit is set to use statevector simulation

    return statevector


def embed_unitary_matrix(operator_matrix, target_qubits, n_qubits):
    # print("embedded:", local_projection_computation.embed_unitary_matrix(operator_matrix,target_qubits,n_qubits))
    return local_projection_computation.embed_unitary_matrix(operator_matrix,target_qubits,n_qubits)

    # if not isinstance(operator_matrix, np.ndarray):
    #     raise TypeError("operator_matrix must be a numpy array")
    
    # gate_size = int(np.log2(operator_matrix.shape[0]))
    # if len(target_qubits) != gate_size:
    #     raise ValueError("Mismatch between operator dimension and number of target qubits")

    # target_qubits = sorted(target_qubits)

    # def get_binary_array(num, width):
    #     return [int(x) for x in format(num, f'0{width}b')]

    # dim = 2 ** n_qubits
    # full_matrix = np.zeros((dim, dim), dtype=complex)

    # for i in range(dim):
    #     input_state = get_binary_array(i, n_qubits)
    #     gate_input_idx = 0
    #     for target in target_qubits:
    #         gate_input_idx = (gate_input_idx << 1) | input_state[target]

    #     gate_output = operator_matrix[gate_input_idx]

    #     for gate_output_idx in range(len(gate_output)):
    #         if abs(gate_output[gate_output_idx]) > 1e-16:
    #             output_state = input_state.copy()
    #             gate_bits = get_binary_array(gate_output_idx, len(target_qubits))
    #             for target_idx, target in enumerate(target_qubits):
    #                 output_state[target] = gate_bits[target_idx]

    #             output_idx = int(''.join(map(str, output_state)), 2)
    #             full_matrix[output_idx, i] = gate_output[gate_output_idx]

    # return full_matrix
    


import local_projection_computation
def local_projection_check_if_two_circuits_are_equal(qc_info_1, qc_info_2, tolerance=1e-15):
    """
    Check if two quantum circuits are equivalent using local projection matrices.
    
    This function works by:
    1. Computing the inverse of the second circuit.
    2. Combining the first circuit with the inverse of the second to form a "difference circuit."
    3. Computing local projections of the difference circuit.
    4. Verifying that the resulting projections all preserve the |0...0⟩ state.

    Parameters:
    - qc_info_1: List of gate information from circuit 1
    - qc_info_2: List of gate information from circuit 2
    - tolerance: Tolerance for comparing eigenvector preservation (default = 1e-15)

    Returns:
    - True if circuits are locally equivalent
    - False otherwise
    """

    # Step 1: Get circuit size and depth
    n_qubits_1, depth_1 = Manipulate_layers.find_circuit_properties(qc_info_1)
    n_qubits_2, depth_2 = Manipulate_layers.find_circuit_properties(qc_info_2)

    # Optional check for same circuit structure (commented out in case inverse makes them match)
    # if n_qubits_1 != n_qubits_2 or depth_1 != depth_2:
    #     print('Two circuits have different qubit counts or depths')
    #     return False

    # Step 2: Invert the second circuit and concatenate it with the first
    _, inverse_qc_info_2 = Create_quantum_circuit.load_inverse_circuit_from_gate_info(
        qc_info_2, n_qubits=n_qubits_1)
    Identity_circuit_info = Create_quantum_circuit.combine_circuits_info(
        qc_info_1, inverse_qc_info_2)

    # Step 3: Compute local projections of the combined "difference circuit"
    # local_projections = local_projection_computation.compute_local_projections_fullly_general(gates_info=Identity_circuit_info,n_qubits=n_qubits_1,depth=depth_1+depth_2)
    local_projections = compute_local_projection_2d_circuit(
        n_qubits_1,
        depth=depth_1 + depth_2,
        gates_info=Identity_circuit_info
    )

    # Step 4: For each projection matrix, verify it preserves |0...0⟩ state
    for count, entry in enumerate(local_projections, start=1):
        local_proj = entry['local_projection']

        # Sanity check: should be square matrix
        if not isinstance(local_proj, np.ndarray) or local_proj.shape[0] != local_proj.shape[1]:
            print(f"Invalid projection matrix in group {count}")
            return False

        # Define target state |0...0⟩
        target_eigenvector = np.zeros(local_proj.shape[0], dtype=np.complex128)
        target_eigenvector[0] = 1

        # Apply projection
        projection_result = local_proj @ target_eigenvector

        # Check if projection result matches the original vector
        if np.linalg.norm(projection_result - target_eigenvector, ord=1) > local_proj.shape[0] * tolerance:
            print("Projection check failed at group", count)
            return False

    # All local projections passed
    return True
