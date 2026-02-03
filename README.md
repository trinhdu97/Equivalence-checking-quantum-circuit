# Scalable Equivalence Checking and Verification of Shallow Quantum Circuits – Artifact README

This repository contains the implementation and configuration files accompanying our paper **“Scalable Equivalence Checking and Verification of Shallow Quantum Circuits” (OOPSLA2425-paper861)**. It provides code for:

- **Local Projection Computation**
- **Equivalence Checking**
- **Benchmarking Local Projection vs. Statevector Computation**
- **Error Evaluation**
- **Micro-benchmarks for Controlled Unitary Gates and Their Decompositions**

---

## Compatibility and Environment

This project has been developed and tested using the following software environment:

### Python Version
- Python **3.13.2**

### Required Libraries
- `qiskit` (version **1.4.2**)
- `qiskit-aer` (version **0.17.0**)
- `matplotlib` (any version)
- `pylatexenc` (version: 2.10)
---

## Important Notes on Hardware and Usage

1. **Hardware Information**  
   Hardware specifications used in our experiments are described in the paper. While actual runtimes will vary across systems, the **scaling trends with respect to the number of qubits and circuit depth** should remain consistent.

2. **Memory Constraints**  
   The **maximum number of qubits** and **circuit depth** you can simulate will depend on your system’s RAM:
   - For **statevector simulations**, memory usage grows exponentially with `n_qubits`. On our setup with 1TB of Memory, we were able to simulate up to **34 qubits**.
   - For **local projection computation**, memory grows **linearly with qubit count** and **exponentially with depth**, allowing larger `n_qubits` in practice.

   > ⚠️ **Tip:** Qiskit will automatically raise a warning if a state vector simulation exceeds available resources.

3. **Execution Strategy**  
   Before running any notebook cell, please read the comments to understand:
   - What each block computes
   - Whether pre-computed results are already shown, and how we expect on the results
   - How long the block may take to run

   For blocks with **long runtime**, we recommend executing them via standalone Python scripts using a persistent session tool like `tmux`. These blocks are generally **independent** of each other (except for initial imports and core module loading). Then they can be cut out from the notebook to run easily.

4. **Erratum**  
   We acknowledge a typo in the paper (Line 879):  
   - The correct range for `n` is: **20 ≤ n ≤ 34**, not **10 ≤ n ≤ 34**.  
   - This will be corrected in the revised submission (due **July 29**).

---

## Reproducing Figures from the Paper

The following Jupyter notebooks reproduce key experimental results:

| Figure(s)        | Notebook File                           |
|------------------|------------------------------------------|
| Figure 2a, 2b    | `Run_experiment_local_projection.ipynb`  |
| Figure 3         | `Micro_benchmark_FIG_3.ipynb`            |
| Figure 4         | `Micro_benchmark_FIG_4.ipynb`            |
| Figure 5a, 6     | `Checking_Weak_Equivalence.ipynb`        |
| Figure 5b        | `Checking_Equivalence_with_Choi.ipynb`   |
| Figure 7         | `Evaluate_error_local_projection_computation.ipynb`|

# The following is a list of Core python files that support the experiments in the notebook files: 

+ `Create_quantum_circuit.py`
+ `Manipulate_layers.py`
+ `local_projection_computation.py`
+ `Check_Weak_Equivalence.py`
+ `Check_Equivalence_with_Choi.py`


Reader can find the interpretation for the functions in these python file below:

## --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Core File: `Create_quantum_circuit.py`

This module provides a comprehensive toolkit for creating, manipulating, perturbing, saving, and loading quantum circuits, especially those structured in 2D layouts or composed of Haar-random gates.

### Key Functions and Their Roles

### 1. Circuit Construction
- `create_random_haar_circuit(n_qubits, depth)`  
  Generates a quantum circuit using 2-qubit Haar-random unitary gates, alternating across layers.

- `create_random_circuit(n_qubits, depth)`  
  Constructs a circuit using structured gate compositions (e.g., Rz, square-root X, and CZ).

- `create_one_shot_random_circuit(n_qubits, depth, basis)`  
  Creates a randomized circuit and appends measurements in a given Pauli basis (X or Z).

### 2. Gate Handling and Metadata
- `add_gate(...)`  
  Adds a quantum gate to a circuit and logs the gate’s metadata (e.g., type, qubits, optional matrix or angle).

- `combine_circuits_info(circuit_info_U1, circuit_info_U2)`  
  Combines the metadata of two circuits, adjusting layers so the second follows the first.

### 3. Perturbation Utilities
- `apply_perturbation_to_haar_circuit(...)`  
  Applies random perturbations (via Rz gates) to qubits in unitary operations using a configurable noise parameter `sigma`.

- `apply_and_save_perturbation(...)`  
  Loads a circuit, applies perturbations, and saves the perturbed version to a JSON file.

### 4. Saving and Loading Circuits
- `save_circuit_to_json(...)` and `load_gates_info(...)`  
  Save and load circuit metadata with complex number support via JSON encoding/decoding.

- `load_circuit_from_json(...)`  
  Reconstructs a `QuantumCircuit` from gate metadata stored in JSON, including support for measurements and barriers.

### 5. Measurement Support
- `add_measurement(...)`  
  Adds measurement operations in Pauli-X, Pauli-Y, or Pauli-Z basis, with proper rotation gates.

- `load_circuit_and_add_measurements(...)`  
  Loads a circuit and applies measurement logic according to a provided Pauli string (e.g., 'XYZX').

### 6. Inverse Circuit Generation
- `load_inverse_circuit_from_gate_info(...)`  
  Reconstructs the inverse of a given circuit by reversing and conjugating unitary gates and flipping layer indices.

### 7. Circuit Concatenation
- `extend_and_concatenate_gates_info(json_files)`  
  Combines multiple JSON-based circuit files into a single larger circuit with adjusted qubit indices.

## -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Core File: `Manipulate_layers.py`

This module provides utility functions for analyzing the structure and dependencies within quantum circuits, especially to compute light cones and reduce circuits based on qubit interactions.

### Key Functions and Their Roles

### 1. Layer Structuring
- `divide_circuit_into_layers_using_layer_index(gates_info)`  
  Organizes gate operations into layers based on the `layer` attribute in each gate's metadata.

### 2. Light Cone Computation
- `extract_involving_qubits_at_each_layer_using_layer_index(...)`  
  Traces backward through layers to determine which qubits influence a target qubit (i.e., backward light cone).

- `extract_involving_qubits_in_forward_lightcone_using_layer_index(...)`  
  Traces forward from a set of target qubits to determine their influence across layers.

- `compute_reduced_indices(circuit_info, n_qubits)`  
  Computes reduced sets of qubit indices based on their light cones, then filters for unique sets not nested in others.

### 3. Circuit Reduction
- `get_reduced_circuit_from_layers(...)`  
  Constructs a new circuit including only the gates relevant to the evolving set of qubits (based on backward light cones).

- `get_reduced_circuit_from_forward_lightcone(...)`  
  Builds a reduced circuit using forward light cone information, retaining original qubit indices.

### 4. Metadata Extraction
- `find_number_of_qubits_involved(qc_info)`  
  Determines the total number of qubits used in a given circuit based on gate metadata.

- `find_circuit_properties(qc_info)`  
  Returns both the number of qubits and the circuit depth by analyzing `layer` and `qubits` attributes.

### 5. Utility
- `retain_container_lists(...)`  
  Removes lists that are strict subsets of other lists from a collection of qubit index groups, often used to simplify light cone results.

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Core File: `local_projection_computation.py`

This module implements the core logic for simulating and analyzing **local projections** in quantum circuits. It supports initialization, propagation across circuit layers, and benchmarking against full statevector simulation to assess performance and scalability.

### Key Functionalities

### 1. Local Projection Initialization and Propagation
- `initialize_local_projection_dict(...)`  
  Initializes projection entries layer-by-layer using forward lightcone analysis.
  
- `compute_local_projections(...)` and `compute_local_projections_fullly_general(...)`  
  Propagate local projection matrices across layers using unitaries and partial measurements.

- `compute_specific_local_projections(...)`  
  Targets a single qubit's local projection for fine-grained analysis.

- `compute_local_projections_with_Choi_isomorphism(...)`  
  Variant using the Choi-Jamiolkowski isomorphism and EPR states.

- `initialize_local_projection_with_initial_dict(...)`  
  Reinitializes local projections with externally provided starting conditions.

### 2. Projection Analysis and Utility
- `get_elements_involving_qubit(...)`  
  Extracts projection entries related to a specific qubit.

- `is_sequential_natural_numbers(...)` and `are_consecutive_natural_numbers(...)`  
  Validates the structure of qubit indices in projection analysis.

- `print_local_projections_at_depth(...)`  
  Prints projection matrices cleanly for a given circuit depth.

- `embed_unitary_matrix(...)`  
  Embeds a gate acting on a subset of qubits into the full system Hilbert space.

### 3. Benchmarking and Plotting
- `benchmark_statevector_vs_local_projection(...)`  
  Compares runtime of local projection vs. full statevector simulation across different qubit counts.

- `benchmark_statevector_vs_local_projection_by_depth(...)`  
  Compares performance as a function of circuit depth for a fixed number of qubits.

- `plot_statevector_vs_local_projection(...)` and `plot_statevector_vs_local_projection_by_depth(...)`  
  Generates performance plots with error bars for benchmarking results.

### Purpose

This file is essential for evaluating the computational trade-offs between local projection methods and traditional statevector simulation. It is particularly useful for assessing scalability of quantum circuit analysis in noisy or large-scale settings.

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Core File: `Check_Weak_Equivalence.py`

This module benchmarks and compares two approaches to quantum circuit equivalence checking: local projection analysis and full statevector simulation. It provides tools to verify circuit equality, evaluate runtime performance across circuit sizes, and visualize the results.

### Key Functionalities

### 1. Circuit Equivalence Checking
- `local_projection_check_if_two_circuits_are_equal(...)`  
  Uses lightcone propagation and local projection tracking to determine if two circuits are equivalent.

- `state_vector_computation_check_if_two_circuits_are_equal(...)`  
  Compares two circuits by simulating their combined unitary (appending one with the inverse of the other) and checking if the result approximates the identity action on the |0⟩ state.

### 2. Benchmarking Routines
- `benchmark_methods_comparison(...)`  
  Benchmarks the runtime of both equivalence-checking methods over increasing qubit counts and circuit depths. Supports both equivalence and inequivalence testing scenarios.

- `benchmark_methods_comparison_old(...)`  
  Legacy benchmarking function used for comparison on older experiments.

### 3. Visualization
- `plot_statevector_vs_local_projection(...)`  
  Reads benchmark results from output files and plots average runtimes with error bars for both equivalence and inequivalence checking. Differentiates methods using colors and line styles.

### 4. Distance Metrics
- `matrix_distances(A, B)`  
  Computes both the Frobenius and spectral norm distances between two matrices. Useful for numerical comparison.

### Purpose

This module is central to validating the effectiveness and scalability of the local projection method against conventional statevector simulation. It helps quantify performance trade-offs and supports empirical claims made in the paper.

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Core File: `Checking_Equivalence_with_Choi.py`

This script implements **quantum circuit equivalence checking** using the **Choi isomorphism**, and benchmarks its performance. It is part of the artifact for evaluating shallow quantum circuits efficiently using local projections.

---

## Utility Functions

### `kron_power(vector, times)`
Computes the Kronecker product of a vector with itself multiple times (used to build composite quantum states like `|000...0⟩`).

---

### `find_position_of_initial_qubit_in_entry(entry)`
Given a dictionary entry from the local projection data, determines how many qubits come **before** and **after** the `initial_qubit` within its local block.

---

### `split_matrix_into_named_blocks(A)`
Splits a square matrix into 4 blocks labeled `B_00`, `B_01`, `B_10`, `B_11`. Used in reconstructing higher-order entangled matrices.

---

### `reconstruct_matrix_M_ACB(M_AB, M_C)`
Reconstructs a matrix acting on three subsystems from one that acts on two, using tensor logic like `M_ABC = A ⊗ C ⊗ B`.

---

##  Core Computation

### `compute_local_projections_with_Choi_isomorphism(...)`
Efficiently computes local projection matrices using the Choi isomorphism for a given quantum circuit:
- Initializes EPR-based projections for the first layer
- Updates projections layer by layer with gate applications
- Handles tensor reshaping to embed gates across qubit positions

---

##  Equivalence Checking

### `check_if_two_circuits_are_equal_using_Choi_isomorphism(qc_info_1, qc_info_2, tolerance)`
- Combines a circuit with the inverse of the second circuit
- Computes the global local projection using Choi
- Applies EPR projectors to verify if each local block maps the initial test state back to itself
- If **all blocks pass**, the circuits are declared **equivalent**

---

##  Benchmarking Tools

### `benchmark_equivalence_vs_inequivalence(...)`
Runs timed experiments to evaluate:
- **Equivalence Check**: Comparing a circuit with itself
- **Inequivalence Check**: Comparing two different circuits

Results (average time ± std) are saved in `.txt` files for plotting.

---

### `plot_benchmark_results(...)`
Visualizes the benchmarking results:
- Plots average time vs. number of qubits
- Includes error bars for standard deviation
- Saves figure as a PDF


