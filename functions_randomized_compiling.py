from qiskit import QuantumCircuit,transpile
import numpy as np


#Functions used for randomized compiling

def standard_twirl(qc):
    """
    Parameter: 
    circuit (QuantumCircuit)
        
    Returns: circuit (QuantumCircuit): one random twirl configuration of the circuit, all the CNOT gates have been randomly Pauli twirl once
    the Pauli twirls are applied only on the control and target qubit
    """
    qct=transpile(qc,optimization_level=0,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)#transpilation in terms CNOT gates
    qc2=QuantumCircuit(qc.num_qubits,qc.num_clbits)
    for gate in qct:#loop on all the gate of the circuit
        if gate[0].name=='cx':
            qb_ctrl=gate[1][0].index #control qubit
            qb_trgt=gate[1][1].index #target qubit
            rand_ctrl,rand_trgt=np.random.randint(4, size=2) #Random choice of the Pauli gates
            if rand_ctrl==1:#implement the random Pauli gates before the CNOT
                qc2.x(qb_ctrl)
            if rand_ctrl==2:
                qc2.y(qb_ctrl) 
            if rand_ctrl==3:
                qc2.z(qb_ctrl)
            if rand_trgt==1:
                qc2.x(qb_trgt)
            if rand_trgt==2:
                qc2.y(qb_trgt)
            if rand_trgt==3:
                qc2.z(qb_trgt)
            qc2.barrier([qb_ctrl,qb_trgt])#to avoid merging of Pauli gates before and after the CNOT
            qc2.cx(qb_ctrl,qb_trgt)#The original CNOT gate
            qc2.barrier([qb_ctrl,qb_trgt])
            qc2.cx(qb_ctrl,qb_trgt)#undo the action of the previously introduced Pauli gate. 
            if rand_ctrl==1:#Here instead of computing directly the resulting Pauli gate,
                qc2.x(qb_ctrl)#we frame the Pauli gate choose earlier by 2CNOTS
            if rand_ctrl==2:#the final transpilation will annihilate both CNOTS and compute the resulting Pauli gate automatically
                qc2.y(qb_ctrl)
            if rand_ctrl==3:
                qc2.z(qb_ctrl)
            if rand_trgt==1:
                qc2.x(qb_trgt)
            if rand_trgt==2:
                qc2.y(qb_trgt)
            if rand_trgt==3:
                qc2.z(qb_trgt)
            qc2.cx(qb_ctrl,qb_trgt)
        if gate[0].name=='x':#copy simply the 1-qubit gate
            qc2.x(gate[1][0].index)
        if gate[0].name=='sx':
            qc2.sx(gate[1][0].index)
        if gate[0].name=='rz':
            theta=gate[0].params[0]
            qc2.rz(theta,gate[1][0].index)
        if gate[0].name=='measure':
            qb_meas=gate[1][0].index #measured qubit
            cb=gate[2][0].index #classical bit
            qc2.measure(qb_meas,cb)
        if gate[0].name=='barrier':
            qc2.barrier(list(gate[1][k].index for k in range(len(gate[1]))))
    qct2=transpile(qc2,optimization_level=3,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    return qct2

def used_qubits(qc):
    """
    Parameter: 
    circuit (QuantumCircuit)
        
    Returns: list: indices of the qubits used in the circuit
    """
    L=qc.num_qubits
    used_qb=[]
    qct=transpile(qc,optimization_level=0,basis_gates=['cx','rx','rz'])
    for gate in qct:
        if gate[0].name=='rx' or gate[0].name=='rz':
            if gate[1][0].index not in used_qb:
                used_qb.append(gate[1][0].index)
        if gate[0].name=='cx':
            if gate[1][0].index not in used_qb:
                used_qb.append(gate[1][0].index)
            if gate[1][1].index not in used_qb:
                used_qb.append(gate[1][1].index)
    return used_qb

def neighbours(qb_index,coupling_map):
    """
    Parameters: 
    integer: index of a qubit 
    list: coupling map of the backend (same format as in Qiskit)
        
    Returns: list: indices of neighbouring qubits of qb_index
    """
    neigh=[]
    for qb_1,qb_2 in coupling_map:
        if qb_1==qb_index and qb_2 not in neigh:
            neigh.append(qb_2)
        if qb_2==qb_index and qb_1 not in neigh:
            neigh.append(qb_1)
    return neigh

def CNOT_used(qc,coupling_map):
    """
    Parameters: 
    circuit (QuantumCircuit)
    list: coupling map of the backend (same format as in Qiskit)
        
    Return: list: indices of possible CNOT used in the circuit
    """
    list_CNOT=[]
    used_qb=used_qubits(qc)
    for qb_ctrl in used_qb:
        for qb_trgt in used_qb:
            if [qb_ctrl,qb_trgt] in coupling_map:
                list_CNOT.append([qb_ctrl,qb_trgt])
    return (list_CNOT)
                
    
def neighbours_CNOT_used(qb_ctrl,qb_trgt,qc,coupling_map):
    """
    Parameters: 
    integer: index of the control qubit of the CNOT
    integer: index of the target qubit of the CNOT
    circuit (QuantumCircuit)
    list: coupling map of the backend (same format as in Qiskit)
        
    Return: list: neighbouring qubits of the CNOT used in the circuit
    """
    neigh_ctrl=neighbours(qb_ctrl,coupling_map)
    neigh_trgt=neighbours(qb_trgt,coupling_map)
    used_qb=used_qubits(qc)
    neigh_CNOT=[]
    for qb in used_qb:
        if qb!=qb_ctrl and qb!=qb_trgt and (qb in neigh_ctrl or qb in neigh_trgt):
            neigh_CNOT.append(qb)
    return neigh_CNOT            
                
            

def crosstalk_twirl(qc,coupling_map):
    """
    Parameter: 
    circuit (QuantumCircuit)
    list: coupling map of the backend (same format as in Qiskit)
        
    Returns: circuit (QuantumCircuit): one random twirl configuration of the circuit, all the CNOT gates have been randomly Pauli twirl once
    neighbouring qubits of the CNOT are randomly Pauli twirled +Pi/2 rotation twirled
    """
    qct=transpile(qc,optimization_level=0,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    used_qb=used_qubits(qct)
    list_CNOT=CNOT_used(qct,coupling_map)
    list_neighbours_CNOT={}
    for qb_ctrl,qb_trgt in list_CNOT:
        list_neighbours_CNOT[qb_ctrl,qb_trgt]=neighbours_CNOT_used(qb_ctrl,qb_trgt,qc,coupling_map)
                                  
    qc2=QuantumCircuit(qc.num_qubits,qc.num_clbits)
    for gate in qct:
        if gate[0].name=='cx':
            qb_ctrl=gate[1][0].index
            qb_trgt=gate[1][1].index
            neigh=list_neighbours_CNOT[qb_ctrl,qb_trgt]
            rand_ctrl,rand_trgt=np.random.randint(4, size=2) #Random choice of the Pauli gates for control and target qubit
            rand_pauli=np.random.randint(4, size=len(neigh)) #Random choice of the Pauli gates for neighbouring qubits
            rand_rot=np.random.randint(3, size=len(neigh)) #Random choice of the direction for the Pi/2 rotation
            if rand_ctrl==1:#implement the random Pauli gates on control and target qubits before the CNOT 
                qc2.x(qb_ctrl)
            if rand_ctrl==2:
                qc2.y(qb_ctrl) 
            if rand_ctrl==3:
                qc2.z(qb_ctrl)
            if rand_trgt==1:
                qc2.x(qb_trgt)
            if rand_trgt==2:
                qc2.y(qb_trgt)
            if rand_trgt==3:
                qc2.z(qb_trgt)
            for k in range(len(neigh)):#implement the random Pauli gates on neighbouring qubits before the CNOT
                if rand_pauli[k]==1:
                    qc2.x(neigh[k])
                if rand_pauli[k]==2:
                    qc2.y(neigh[k])
                if rand_pauli[k]==3:
                    qc2.z(neigh[k])
                if rand_rot[k]==0: #implement the random pi/2 rotation gate on neighbouring qubits before the CNOT 
                    qc2.rx(np.pi/2,neigh[k])
                if rand_rot[k]==1:
                    qc2.ry(np.pi/2,neigh[k])
                if rand_rot[k]==2:
                    qc2.rz(np.pi/2,neigh[k])
            qc2.barrier(neigh+[qb_ctrl,qb_trgt])
            qc2.cx(qb_ctrl,qb_trgt)#CNOT of the original circuit
            qc2.barrier(neigh+[qb_ctrl,qb_trgt])
            qc2.cx(qb_ctrl,qb_trgt) #undo the action of the previous twirl
            if rand_ctrl==1:
                qc2.x(qb_ctrl)
            if rand_ctrl==2:
                qc2.y(qb_ctrl) 
            if rand_ctrl==3:
                qc2.z(qb_ctrl)
            if rand_trgt==1:
                qc2.x(qb_trgt)
            if rand_trgt==2:
                qc2.y(qb_trgt)
            if rand_trgt==3:
                qc2.z(qb_trgt)
            for k in range(len(neigh)):
                if rand_rot[k]==0:
                    qc2.rx(-np.pi/2,neigh[k])
                if rand_rot[k]==1:
                    qc2.ry(-np.pi/2,neigh[k])
                if rand_rot[k]==2:
                    qc2.rz(-np.pi/2,neigh[k])
                if rand_pauli[k]==1:
                    qc2.x(neigh[k])
                if rand_pauli[k]==2:
                    qc2.y(neigh[k])
                if rand_pauli[k]==3:
                    qc2.z(neigh[k])
            qc2.cx(qb_ctrl,qb_trgt)
        if gate[0].name=='x':#copy of the 1-qubit gates
            qc2.x(gate[1][0].index)
        if gate[0].name=='sx':
            qc2.sx(gate[1][0].index)
        if gate[0].name=='rz':
            theta=gate[0].params[0]
            qc2.rz(theta,gate[1][0].index)
        if gate[0].name=='measure':
            qb_meas=gate[1][0].index #measured qubit
            cb=gate[2][0].index #classical bit
            qc2.measure(qb_meas,cb)
        if gate[0].name=='barrier':
            qc2.barrier(list(gate[1][k].index for k in range(len(gate[1]))))
    qct2=transpile(qc2,optimization_level=3,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    return qct2

