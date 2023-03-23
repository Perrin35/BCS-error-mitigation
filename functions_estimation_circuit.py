from qiskit import QuantumCircuit,transpile
import numpy as np

from functions_randomized_compiling import used_qubits

#Functions used for building noise estimation circuits

def preserve_cnot(qc):
    """
    Parameter: 
    circuit (QuantumCircuit)
        
    Return: circuit (QuantumCircuit): circuit where only the CNOT are conserved (the circuit is first transpiled in terms of CNOT)
    """
    qct=transpile(qc,optimization_level=0,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    qc2=QuantumCircuit(qc.num_qubits,qc.num_clbits)
    for gate in qct:
        if gate[0].name=='cx':
            qc2.cx(gate[1][0].index,gate[1][1].index)
        if gate[0].name=='measure':
            qb=gate[1][0].index
            cb=gate[2][0].index 
            qc2.measure(qb,cb)
        if gate[0].name=='barrier':
            qc2.barrier(list(gate[1][k].index for k in range(len(gate[1]))))
    qc2=transpile(qc2,optimization_level=0,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    return qc2

def coupling_list(qc):
    """
    Parameter: 
    circuit (QuantumCircuit)
        
    Return: list: build the coupling map induced from the circuit by at the position of the CNOT
    """
    qct=transpile(qc,optimization_level=0,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    couplings=[]
    for gate in qct:
        if gate[0].name=='cx':
            qb_ctrl=gate[1][0].index
            qb_trgt=gate[1][1].index
            if [qb_ctrl,qb_trgt] not in couplings:
                couplings.append([qb_ctrl,qb_trgt])
                couplings.append([qb_trgt,qb_ctrl])
    return couplings
    

def detect_swap(qc):
    """
    Parameters: 
    circuit (QuantumCircuit)
        
    Return: list : sequence of indices of qubits swapped in the circuit
    """
    swap=[]
    qct=transpile(qc,optimization_level=0,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    coupling_map=coupling_list(qc)
    count_cnot=list(0 for _ in range(len(coupling_map)))#count the number of CNOT in a row head to tail
    for gate in qct:
        if gate[0].name=='cx':
            qb_ctrl=gate[1][0].index
            qb_trgt=gate[1][1].index
            head=coupling_map.index([qb_ctrl,qb_trgt])#index of the coupling (i,j) in the coupling map list
            tail=coupling_map.index([qb_trgt,qb_ctrl])#index of the coupling (j,i) in the coupling map list
            if count_cnot[head]==0 and count_cnot[tail]==0:#this algorithm detect if there is a SWAP transpile in terms of CNOT (3 CNOTS)
                count_cnot[head]=1
            elif count_cnot[head]==0 and count_cnot[tail]==1:
                count_cnot[head]=2
            elif count_cnot[head]==1 and count_cnot[tail]==0:
                count_cnot[head]=0
            elif count_cnot[head]==1 and count_cnot[tail]==2:
                swap.append([qb_ctrl,qb_trgt])
                count_cnot[head]=0
                count_cnot[tail]=0
            elif count_cnot[head]==2 and count_cnot[tail]==1:
                count_cnot[head]=0
            for k in range(len(coupling_map)):
                cnot_index=coupling_map[k]
                if cnot_index[0]==qb_ctrl or cnot_index[0]==qb_trgt or cnot_index[1]==qb_ctrl or cnot_index[1]==qb_trgt:
                    if k!=head and k!=tail:
                        count_cnot[k]=0
        if gate[0].name=='x' or gate[0].name=='sx' or gate[0].name=='rz' or gate[0].name=='measure':
            qb=gate[1][0].index
            for k in range(len(coupling_map)):
                cnot_index=coupling_map[k]
                if cnot_index[0]==qb or cnot_index[1]==qb:
                    count_cnot[k]=0
    return swap


def final_layout(qc):
    """
    Parameters 
    circuit (QuantumCircuit)
        
    Return: list : infer the final layout of the circuit by detecting the SWAP gates in the circuit
    """
    qubs=used_qubits(qc)
    swap=detect_swap(qc)
    for [i,j] in swap:
        k=qubs.index(i)
        r=qubs.index(j)
        qubs[k]=j
        qubs[r]=i
    return qubs


def add_random(qc):
    """
    Parameter: 
    circuit (QuantumCircuit)
        
    Return: circuit (QuantumCircuit): Modify the circuit by adding at the beginning initial random 1-qubit unitaries and 
                                     undo ther action at the end by taking into account possible SWAP
    """
    qct=transpile(qc,optimization_level=0,basis_gates=['cx','x','sx','rz','id'],approximation_degree=1)
    qc2=QuantumCircuit(qc.num_qubits,qc.num_clbits)#circuit for the initial random unitaries
    qc3=QuantumCircuit(qc.num_qubits,qc.num_clbits)#circuit at the end to undo the random gates
    qb_ini=used_qubits(qct)
    qb_fin=final_layout(qc)
    for i in range(len(qb_ini)):
        th=np.random.uniform(0, np.pi)#random angles
        phi=np.random.uniform(0, 2*np.pi)
        lam=np.random.uniform(0, 2*np.pi)
        qc2.u(th,phi,lam,qb_ini[i])#initial random unitary gates
        qc3.u(-th,-lam,-phi, qb_fin[i])#undo the action of random gates by taking into account the swap
    qc2+=qct #add to the initial rotation the circuit
    gate=qc2[-1]#start by the endof the circuit
    meas=[]
    while gate[0].name=='measure':#suppose the last actions are only measurement
        meas.append([gate[1][0].index,gate[2][0].index])#register the qubit measured and its corresponding classical bit
        qc2.data.pop(len(qc2.data)-1)#remove measurement
        gate=qc2[-1]
    if gate[0].name=='barrier':
        qc2.data.pop(len(qc2.data)-1)
    qc2.barrier()
    qc2+=qc3#add the last random unitaries wich undo the effect of the initial ones
    qc2.barrier()
    for ind in meas:#put back the measurement
        qc2.measure(ind[0],ind[1])
    qc2=transpile(qc2,optimization_level=0,approximation_degree=1)
    return qc2
