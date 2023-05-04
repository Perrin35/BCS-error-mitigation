from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer.backends import AerSimulator
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise import NoiseModel
from functions_randomized_compiling import used_qubits
import numpy as np

#Functions used in the post-analysis notebook

def IBU(ymes,t0,Rin,n):
    #This is the iterative Bayesian unfolding method.
    #Rin is a matrix where the first coordinate is the measured value and the second coordinate is the true value.
    #n is the number of iterations.
    
    tn = t0
    for q in range(n):
        out = []
        for j in range(len(t0)):
            mynum = 0.
            for i in range(len(ymes)):
                myden = 0.
                for k in range(len(t0)):
                    myden+=Rin[i][k]*tn[k]
                    pass
                mynum+=Rin[i][j]*tn[j]*ymes[i]/myden
                pass
            out+=[mynum]
        tn = out
        pass
    '''
    tn = t0
    for i in range(n):
        Rjitni = [np.array(Rin[:][i])*tn[i] for i in range(len(tn))]
        Pm_given_t = Rjitni / np.matmul(Rin,tn)
        tn = np.dot(Pm_given_t,ymes)
        pass
    '''
    return tn



def noisy_trotter(errors,initial_step,final_step,meas_list,measurement, dt, PATH):
    """
    Parameter=: 
    list: float representing the errors of a depolarising model such that: the first and second elements
        are the depolarising parameters of the active qubits of the CNOT respectively in the junction (0,1) and (1,2).
        The third and fourth  elements are the depolarising parameter for the neighbouring qubits (resp. 2 and 0). 
        The last element is the global depolarising parameter. 
        (Warning they are defined differently as in the article, see transf_noise_params functions)
    
     initial_step: Integer:  initial step of the evolution
     
     final_step: Integer : final step of the evolution
     
     meas_list: list of the different observables (see Analysis notebook for an example)
     
     measurement: string (either 'XYZ' or 'ZZZ')
     
     dt: float Trotter time step
     
     PATH: to get the corresponding Trotter circuit  in data
     
    
    Return: list: The noisy Trotter evolution of the 7 different observables with such error channel.
                      
                        
                
    """
    error_loc_0=errors[0]
    error_loc_1=errors[1]
    error_neigh_0=errors[2]
    error_neigh_1=errors[3]
    error_global=errors[4]
    
    #Define the noise model
    #2 qubit depolarising channel on qubits of CNOT
    local_0=noise.depolarizing_error(error_loc_0,2)
    local_1=noise.depolarizing_error(error_loc_1,2)
    
    #1-qubit depolarising channel on neighbouring qubit
    neighbour_0=noise.depolarizing_error(error_neigh_0,1)
    neighbour_1=noise.depolarizing_error(error_neigh_1,1)
    
    #3-qubit (global) depolarising channel
    global_error=noise.depolarizing_error(error_global,3)
    
    #Here the depolarising noise channel (local, neighbouring and global are defined succesively unlike the article)
    noise_model = noise.NoiseModel()
    noise_model.add_quantum_error(local_0, ['cx'],[0,1],warnings=False)
    noise_model.add_quantum_error(local_0, ['cx'],[1,0],warnings=False)
    noise_model.add_quantum_error(local_1, ['cx'],[1,2],warnings=False)
    noise_model.add_quantum_error(local_1, ['cx'],[2,1],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_0, 'cx', [0,1], [2],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_0, 'cx', [1,0], [2],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_1, 'cx', [1,2], [0],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_1, 'cx', [2,1], [0],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [0,1], [0,1,2],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [1,0], [0,1,2],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [1,2], [0,1,2],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [2,1], [0,1,2],warnings=False)
    
    #Define a simulator with such noise channel
    sim=AerSimulator(noise_model=noise_model,method='density_matrix')
    
    trotter_noisy=list(list(0. for _ in range(final_step-initial_step+1)) for _ in range(len(meas_list)))
    for t in range(initial_step,final_step+1):
        if t//2*2!=t:
            lay=[0,2,1]
        else:
            lay=[0,1,2]
        T=dt*t
        qc=QuantumCircuit.from_qasm_file('data/'+measurement+'/T='+str(round(T,1))+PATH)
        qb_used=used_qubits(qc)
        nb_qubits=len(qb_used)
        
        #Copy the circuit to remove the idle qubits and measurements
        qc2=QuantumCircuit(nb_qubits)
        for gate in qc:
            if gate[0].name=='x':
                qc2.x(qb_used.index(gate[1][0].index))
            if gate[0].name=='sx':
                qc2.sx(qb_used.index(gate[1][0].index))
            if gate[0].name=='rz':
                theta=gate[0].params[0]
                qc2.rz(theta,qb_used.index(gate[1][0].index))
            if gate[0].name=='barrier':
                qc2.barrier()
            if gate[0].name=='cx':
                qc2.cx(qb_used.index(gate[1][0].index),qb_used.index(gate[1][1].index))
        qc2=transpile(qc2,sim, optimization_level=0)
                       
        #Compute the density matrix
        qc2.save_density_matrix()
        counts=sim.run(qc2).result()
        density=counts.data()['density_matrix']
                       
        #Compute the different observable from the density matrix
        for obs in range(len(meas_list)):
            m=list(lay[meas] for meas in meas_list[obs])
            trotter_noisy[obs][t-initial_step]=sum(np.real(density[i,i]*(-1)**(sum(i//2**m[k] for k in range(len(m))))) for i in range(8))
    return trotter_noisy

                       
                       
                       
def fit_trotter(errors,real_trotter,initial_step,final_step,meas_list,measurement, dt, PATH):
    """
    Parameters=: 
    list: floats representing the errors of a depolarising model such that: the first and second elements
        are the depolarising parameters of the active qubits of the CNOT respectively in the junction (0,1) and (1,2).
        The third and fourth  elements are the depolarising parameter for the neighbouring qubits (resp. 2 and 0). 
        The last element is the global depolarising parameter. 
        (Warning they are defined differently as in the article, see transf_noise_params functions)
    
    list: the real noisy Trotter evolution on a quantum computer 
        (the average on the different Pauli randomized instances when applying error mitigation protocols)
        
    initial_step: Integer:  initial step of the evolution
     
    final_step: Integer : final step of the evolution
     
    meas_list: list of the different observables (see Analysis notebook for an example)
     
    measurement: string (either 'XYZ' or 'ZZZ')
    
    dt: float Trotter time step
     
    PATH: to get the corresponding Trotter circuit  in data
    
    
    Return: list: All the discrepancies at each time step and for the 7 different observables
                  between the noise model and the output of the QC
    
                                
    """
    trotter_noisy=noisy_trotter(errors,initial_step,final_step,meas_list,measurement, dt, PATH)
    diff=np.ravel(list(list((trotter_noisy[obs][t]-real_trotter[obs][t]) for t in range(final_step-initial_step+1)) 
                       for obs in range(len(meas_list))))
    return diff

def transf_noise_params(errors):
    """
    Parameter=: 
    list: float representing the errors of a depolarising model such that: the first and second elements
        are the depolarising parameters of the active qubits of the CNOT respectively in the junction (0,1) and (1,2).
        The third and fourth  elements are the depolarising parameter for the neighbouring qubits (resp. 2 and 0). 
        The last element is the global depolarising parameter. 
       
    
     These errors are not defined as in the article beacuse in the noise model of Qiskit we define them successively
     (on top of each other while in the article they are defined at once).
     
     Return:  This function transforms the error defined in the Qiskit noise model onto the one of the article.
                
    """
    error_loc_0=errors[0]
    error_loc_1=errors[1]
    error_neigh_0=errors[2]
    error_neigh_1=errors[3]
    error_global=errors[4]
    error_loc_0_bis= error_loc_0*(1-error_global)*(1-error_neigh_0)
    error_loc_1_bis= error_loc_1*(1-error_global)*(1-error_neigh_1)
    error_neigh_0_bis=error_neigh_0*(1-error_global)*(1-error_loc_0)
    error_neigh_1_bis=error_neigh_1*(1-error_global)*(1-error_loc_1)
    error_global_bis=error_global+(1-error_global)*(4/9*error_loc_0*error_neigh_0+5/9*error_loc_1*error_neigh_1)
    return [error_loc_0_bis,error_loc_1_bis,error_neigh_0_bis,error_neigh_1_bis,error_global_bis]


def noisy_estimation_circuit(errors,initial_step, final_step, meas_list, dt, PATH):
    """
    Parameter=: 
    list: float representing the errors of a depolarising model such that: the first and second elements
        are the depolarising parameters of the active qubits of the CNOT respectively in the junction (0,1) and (1,2).
        The third and fourth  elements are the depolarising parameter for the neighbouring qubits (resp. 2 and 0). 
        The last element is the global depolarising parameter. 
        (Warning they are defined differently as in the article, see transf_noise_params functions)
        
    initial_step: Integer:  initial step of the evolution
     
    final_step: Integer : final step of the evolution
     
    meas_list: list of the different observables (see Analysis notebook for an example)
    
    dt: float Trotter time step
    
    PATH: to get the corresponding noise estimation circuit  in the data folder
    
    Return: list: The noisy evolution of the 7 different observables of the estimation circuits with such error channel.
    

    """
    error_loc_0=errors[0]
    error_loc_1=errors[1]
    error_neigh_0=errors[2]
    error_neigh_1=errors[3]
    error_global=errors[4]
    
    #Define the noise model
    #2 qubit depolarising channel on qubits of CNOT
    local_0=noise.depolarizing_error(error_loc_0,2)
    local_1=noise.depolarizing_error(error_loc_1,2)
    
    #1-qubit depolarising channel on neighbouring qubit
    neighbour_0=noise.depolarizing_error(error_neigh_0,1)
    neighbour_1=noise.depolarizing_error(error_neigh_1,1)
    
    #3-qubit (global) depolarising channel
    global_error=noise.depolarizing_error(error_global,3)
    
    #Here the depolarising noise channel (local, neighbouring and global are defined succesively unlike the article)
    noise_model = noise.NoiseModel()
    noise_model.add_quantum_error(local_0, ['cx'],[0,1],warnings=False)
    noise_model.add_quantum_error(local_0, ['cx'],[1,0],warnings=False)
    noise_model.add_quantum_error(local_1, ['cx'],[1,2],warnings=False)
    noise_model.add_quantum_error(local_1, ['cx'],[2,1],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_0, 'cx', [0,1], [2],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_0, 'cx', [1,0], [2],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_1, 'cx', [1,2], [0],warnings=False)
    noise_model.add_nonlocal_quantum_error(neighbour_1, 'cx', [2,1], [0],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [0,1], [0,1,2],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [1,0], [0,1,2],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [1,2], [0,1,2],warnings=False)
    noise_model.add_nonlocal_quantum_error(global_error, 'cx', [2,1], [0,1,2],warnings=False)
    
    sim=AerSimulator(noise_model=noise_model,method='density_matrix')
    
    estimation_noisy=list(list(0. for _ in range(final_step-initial_step+1)) for _ in range(len(meas_list)))
    for t in range(initial_step,final_step+1):
        if t//2*2!=t:
            lay=[0,2,1]
        else:
            lay=[0,1,2]
        T=dt*t
        
        qc=QuantumCircuit.from_qasm_file('data/ZZZ/T='+str(round(T,1))+PATH)
        qb_used=used_qubits(qc)
        nb_qubits=len(qb_used)
                          
        #Copy the circuit to remove the idle qubits and measurements
        qc2=QuantumCircuit(nb_qubits)
        for gate in qc:
            if gate[0].name=='x':
                qc2.x(qb_used.index(gate[1][0].index))
            if gate[0].name=='sx':
                qc2.sx(qb_used.index(gate[1][0].index))
            if gate[0].name=='rz':
                theta=gate[0].params[0]
                qc2.rz(qb_used.index(gate[1][0].index))
            if gate[0].name=='barrier':
                qc2.barrier()
            if gate[0].name=='cx':
                qc2.cx(qb_used.index(gate[1][0].index),qb_used.index(gate[1][1].index))
                        
        #Compute the density matrix
        qc2=transpile(qc2,sim, optimization_level=0)
        qc2.save_density_matrix()
        counts=sim.run(qc2).result()
        density=counts.data()['density_matrix']
        
        #Compute the different observable from the density matrix
        for obs in range(len(meas_list)):
            m=list(lay[meas] for meas in meas_list[obs])
            estimation_noisy[obs][t-initial_step]=sum(np.real(density[i,i]*(-1)**(sum(i//2**m[k] 
                                                                                      for k in range(len(m))))) for i in range(8))
    return estimation_noisy

def fit_estimation(errors,real_estimation,initial_step,final_step,meas_list,dt,PATH):
    """
    Parameters=: 
    list: floats representing the errors of a depolarising model such that: the first and second elements
        are the depolarising parameters of the active qubits of the CNOT respectively in the junction (0,1) and (1,2).
        The third and fourth  elements are the depolarising parameter for the neighbouring qubits (resp. 2 and 0). 
        The last element is the global depolarising parameter. 
        (Warning they are defined differently as in the article, see transf_noise_params functions)
    
    list: the real noisy evolution on a quantum computer of the noise estimation circuits
        (the average on the different Pauli randomized instances when applying error mitigation protocols)
        
    initial_step: Integer:  initial step of the evolution
     
    final_step: Integer : final step of the evolution
     
    meas_list: list of the different observables (see Analysis notebook for an example)
     
    dt: float Trotter time step
     
    PATH: to get the corresponding noise estimation circuit  in the data folder
    
    
    Return: list: All the discrepancies at each time step and for the 7 different observables
                  between the noise model and the output of the QC
    
                        
                
    """
    estimation_noisy=noisy_estimation_circuit(errors,initial_step,final_step,meas_list,dt, PATH)
    diff=np.ravel(list(list((estimation_noisy[obs][t]-real_estimation[obs][t]) 
                            for t in range(final_step-initial_step+1)) for obs in range(len(meas_list))))
    return diff