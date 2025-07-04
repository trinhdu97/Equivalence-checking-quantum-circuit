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
import numpy as np

import Create_quantum_circuit
import Manipulate_layers
import time
import local_projection_computation
import Checking_Equivalence_with_Choi



from scipy.stats import unitary_group

# Generate a Haar-random 4x4 unitary matrix
W = unitary_group.rvs(2)
U = W @ W
print(W)



def construct_control_W(W):
    A_00 = np.array([[1,0],
                     [0,0]])
    A_01 = np.array([[0,1],
                     [0,0]])
    A_10 = np.array([[0,0],
                     [1,0]])
    A_11 = np.array([[0,0],
                     [0,1]])
    # print("A_00 (Top-left block):")
    # print(A_00)
    # print("\nA_01 (Top-right block):")
    # print(A_01)
    # print("\nA_10 (Bottom-left block):")
    # print(A_10)
    # print("\nA_11 (Bottom-right block):")
    # print(A_11)
    control_W = np.kron(A_00,np.eye(2))+np.kron(A_11,W)
    # print(control_W)

    return control_W
np.set_printoptions(linewidth = 200)
# Construct the control version of W
control_W = construct_control_W(W)
print("control_W:")
print(control_W)
# Define the X gate
X = np.array([[0,1],[1,0]])
# Construct the control version of X
control_X = construct_control_W(X)
print("control_X:")
print(control_X)
# Define the SWAP gate
SWAP = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]])
print("SWAP gate:")
print(SWAP)

A_00 = np.array([[1,0],
                     [0,0]])
A_01 = np.array([[0,1],
                     [0,0]])
A_10 = np.array([[0,0],
                     [1,0]])
A_11 = np.array([[0,0],
                     [0,1]])
toffoli = np.kron(A_00,np.eye(4))+np.kron(A_11,control_X) 
print(toffoli)          # Toffoli gate is printed here to check the corretness of the control-control gate construction
print(U)
control_control_U_inverse = np.kron(A_00,np.eye(4))+np.kron(A_11,np.transpose(np.conjugate(control_W@control_W)))     
print(control_control_U_inverse)

#Deutsch gate circuit
qc_Deutsch = QuantumCircuit(20)
qc_info_Deutsch = []

qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[0,1],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))
qc_info_Deutsch[-1]['layer'] = 0
qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2,3],qc_info_Deutsch,qc_Deutsch,gate_matrix=control_W) # control_W is applied on qubits (2,3) instead of (0,1)
qc_info_Deutsch[-1]['layer'] = 0
print(control_W)
# print(qc_info_Deutsch[-1])
for i in range(2,10):
    qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2*i,2*i+1],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))  # other gates are identity gates
    qc_info_Deutsch[-1]['layer'] = 0
# qc_Deutsch.draw('mpl')


# 2nd layer
qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[1,2],qc_info_Deutsch,qc_Deutsch,gate_matrix=control_X) # add control_X gate on qubits (1,2)
qc_info_Deutsch[-1]['layer'] = 1
# print(qc_info_Deutsch[-1])


for i in range(1,9):
    qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2*i+1,2*i+2],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))  # other gates are identity gates
    qc_info_Deutsch[-1]['layer'] = 1

# 3rd layer

qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[0,1],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))     # add identity gate on qubits (0,1)
qc_info_Deutsch[-1]['layer'] = 2
qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2,3],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.transpose(np.conjugate(control_W))) # control_W is applied on qubits (2,3)
qc_info_Deutsch[-1]['layer'] = 2

for i in range(2,10):
    qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2*i,2*i+1],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4)) # other gates are identity gates
    qc_info_Deutsch[-1]['layer'] = 2

# 4th layer

qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[1,2],qc_info_Deutsch,qc_Deutsch,gate_matrix=SWAP@control_X) # add control_X and SWAP gate on qubits (1,2)
qc_info_Deutsch[-1]['layer'] = 3
# print(qc_info_Deutsch[-1])
for i in range(1,9):
    qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2*i+1,2*i+2],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))
    qc_info_Deutsch[-1]['layer'] = 3

# 5th layer

qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[0,1],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))
qc_info_Deutsch[-1]['layer'] = 4
qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2,3],qc_info_Deutsch,qc_Deutsch,gate_matrix=control_W)
qc_info_Deutsch[-1]['layer'] = 4
for i in range(2,10):
    qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2*i,2*i+1],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))
    qc_info_Deutsch[-1]['layer'] = 4

# 6th layer

qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[1,2],qc_info_Deutsch,qc_Deutsch,gate_matrix=SWAP)
qc_info_Deutsch[-1]['layer'] = 5
for i in range(1,9):
    qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2*i+1,2*i+2],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))
    qc_info_Deutsch[-1]['layer'] = 5

# 7th layer
qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[1,2,3],qc_info_Deutsch,qc_Deutsch,gate_matrix=control_control_U_inverse)
qc_info_Deutsch[-1]['layer'] = 6
# print(qc_info_Deutsch[-1])
for i in range(2,10):
    qc_Deutsch, qc_info_Deutsch = Create_quantum_circuit.add_gate("unitary",[2*i,2*i+1],qc_info_Deutsch,qc_Deutsch,gate_matrix=np.eye(4))
    qc_info_Deutsch[-1]['layer'] = 6

qc_Deutsch.draw('mpl')

print("Computed Choi local projection")
Choi_local_projection = Checking_Equivalence_with_Choi.compute_local_projections_with_Choi_isomorphism(qc_info_Deutsch,20,7)
np.savetxt("large_matrix.txt", Choi_local_projection[1]['local_projection'], fmt="%.3f")

tolerance = 1e-15
EPR_projector_AB = 1/2 * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.complex128)
count =0
check = True
for entry in Choi_local_projection:
    print(entry["qubit_indices"])
    print(entry['initial_qubit'])
    count+=1
    structure = Checking_Equivalence_with_Choi.find_position_of_initial_qubit_in_entry(entry)
    print(structure)  # Get the positions before and after the qubit
    test_density_matrix = EPR_projector_AB
    
    # Reconstruct the matrix based on the structure
    test_density_matrix = Checking_Equivalence_with_Choi.reconstruct_matrix_M_ACB(EPR_projector_AB, np.eye(2**structure[0]))
    test_density_matrix = np.kron(test_density_matrix, np.eye(2**structure[1]))
    # print(test_density_matrix)
    # Compute the image after applying the local projection
    image = test_density_matrix @ test_density_matrix
    
    # Compare the resulting matrix with the original test_density_matrix
    if np.linalg.norm(image-test_density_matrix,ord=1)>image.shape[0]*tolerance:
        check = False
        break

# If all checks pass    
if check == True:
    print("Number of entries in Choi_local_projection:", count)
    print("All local projections are correct.")
    print("Checking successfull.")
else:
    print("Not all local projections are correct.")
    print("Number of entries in Choi_local_projection:", count)
    print("Checking failed.")