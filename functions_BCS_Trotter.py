from qiskit import QuantumCircuit,transpile
from sympy import symbols, solve
from scipy import linalg
import math , cmath
import numpy as np



#Functions used to build  the Trotter circuit

def interaction(theta):
    """
    Parameter=: 
    float: theta is the strength of the interaction e^{-i*theta/2(XX+YY)}
    
    Return: circuit (QuantumCircuit): corresponding to this operator
    """
    qc=QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0,1)
    qc.rz(theta,1)
    qc.cx(0,1)
    qc.h(0)
    qc.h(1)
    qc.rx(-np.pi/2,0)
    qc.rx(-np.pi/2,1)
    qc.cx(0,1)
    qc.rz(theta,1)
    qc.cx(0,1)
    qc.rx(np.pi/2,0)
    qc.rx(np.pi/2,1)
    interaction=qc.to_gate()
    interaction.name = "interaction gate"
    return interaction

def initial_angle(energies,g):
    """
    Parameters:
        list: on-site energies
        float: coupling strength

    Return:
        list: angles to initialize the state into the mean field ground state (we suppose the gap is real)
    """
    d = symbols('d',positive=True)
    expr = abs(d)*(1+d**2)**0.5/(abs(d)+(1+d**2)**0.5/2)-g
    delta = solve(expr)[0]
    return list(2*math.atan(((math.sqrt(en**2+delta**2))-en)/abs(delta)) for en in energies)


#Functions used to simulate classically the BCS model

def pauli_x(qubit,system_size):
    """
    Parameters: 
    integer: the qubit on which is applied the Pauli Matrix X (from 0 to system_size-1)
    integer: number of qubits of the whole system 
    
    Return: numpy array: the corresponding matrix operator
    """
    if qubit==0:
        S=np.array([[0,1],[1,0]])
    else:
        S=np.array([[1,0],[0,1]])
    for j in range(1,system_size):
        if j==qubit:
            S=np.kron(np.array([[0,1],[1,0]]),S)
        else:
            S=np.kron(np.array([[1,0],[0,1]]),S)
    return S

def pauli_y(qubit,system_size):
    """
    Parameters: 
    integer: the qubit on which is applied the Pauli Matrix Y (from 0 to system_size-1)
    integer: number of qubits of the whole system 
    
    Return: numpy array: the corresponding matrix operator
    """
    if qubit==0:
        S=np.array([[0,-1j],[1j,0]])
    else:
        S=np.array([[1,0],[0,1]])
    for j in range(1,system_size):
        if j==qubit:
            S=np.kron(np.array([[0,-1j],[1j,0]]),S)
        else:
            S=np.kron(np.array([[1,0],[0,1]]),S)
    return S
    
    
def pauli_z(qubit,system_size):
    """
    Parameters: 
    integer: the qubit on which is applied the Pauli Matrix Z (from 0 to system_size-1)
    integer: number of qubits of the whole system 
    
    Return: numpy array: the corresponding matrix operator
    """
    if qubit==0:
        S=np.array([[1,0],[0,-1]])
    else:
        S=np.array([[1,0],[0,1]])
    for j in range(1,system_size):
        if j==qubit:
            S=np.kron(np.array([[1,0],[0,-1]]),S)
        else:
            S=np.kron(np.array([[1,0],[0,1]]),S)
    return S

def BCS_classic(energies,g):
    """
    Parameters: 
    list: on-site energies of the system
    float: coupling strength
    
    Return: numpy array: the BCS hamiltonian operator
    """
    L=len(energies)
    hamiltonian=np.zeros([2**L,2**L],dtype=np.complex128)
    for i in range(L):
        for j in range(i+1,L):
            hamiltonian-=g/2*(np.matmul(pauli_x(i,L),pauli_x(j,L))+np.matmul(pauli_y(i,L),pauli_y(j,L)))
        hamiltonian-=(energies[i]-g/2)*pauli_z(i,L)
    return hamiltonian

def observable_operator(obs):
    """
    Parameter: 
    list: Pauli String observable decomposed on the Pauli basis X0 Z0 X1 Z1 ... (Y=1*j*X*Z) 
        length of the list is twice the number of qubits in the system
        
    Return: numpy array: the corresponding observable operator written in matrix form
    """
    if obs[0]!=0:
        if obs[1]!=0:
            obs_op=-1j*obs[0]*obs[1]*np.array([[0,-1j],[1j,0]])
        else:
            obs_op=obs[0]*np.array([[0,1],[1,0]])
    else:
        if obs[1]!=0:
            obs_op=obs[1]*np.array([[1,0],[0,-1]])
        else:
            obs_op=np.array([[1,0],[0,1]])

    for i in range(1,len(obs)//2):
        if obs[2*i]!=0:
            if obs[2*i+1]!=0:
                obs_op=-1j*obs[2*i]*obs[2*i+1]*np.kron(np.array([[0,-1j],[1j,0]]),obs_op)
            else:
                obs_op=obs[2*i]*np.kron(np.array([[0,1],[1,0]]),obs_op)
        else:
            if obs[2*i+1]!=0:
                obs_op=obs[2*i+1]*np.kron(np.array([[1,0],[0,-1]]),obs_op)
            else:
                obs_op=np.kron(np.array([[1,0],[0,1]]),obs_op)
    return obs_op
            
    
