#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
import time
start_time = time.time()

def variational_ansatz(params, wires):
    """The variational ansatz circuit.

    Fill in the details of your ansatz between the # QHACK # comment markers. Your
    

        a_0 |10...0> + a_1 |01..0> + ... + a_{n-2} |00...10> + a_{n-1} |00...01>

    where {a_i} are real-valued coefficients.

    Args:
         params (np.array): The variational parameters.
         wires (qml.Wires): The device wires that this circuit will run on.
    """

    # QHACK #
    #qml.templates.layers.StronglyEntanglingLayers(params,wires=wires)
    qml.templates.state_preparations.ArbitraryStatePreparation(params,wires=wires)
    #n_qubits = len(wires)
    #n_rotations = len(params)
    '''print(wires)
    print(params)
    for i in wires:
        print(params[i])
        qml.Rot(*params[i], wires=i)'''
    
    #if n_rotations > 1:
    '''n_layers = n_rotations // n_qubits
        n_extra_rots = n_rotations - n_layers * n_qubits

        for layer_idx in range(n_layers):
            layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
            qml.broadcast(qml.CNOT, wires, pattern="ring")'''
        #qml.PauliX(1)
        #print(wires)
        #for i in wires:
        #    qml.Rot(*params[i], wires=i)
    '''
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[0,1])'''
        #qml.CNOT(wires=[1,2])
        #qml.CNOT(wires=[2,0])
        #qml.CNOT(wires=[2,0])
        #qml.CNOT(wires=[3,1])
        
    #else:
    #    qml.Rot(*params[0], wires=wires[0])
    
    # QHACK #


def run_vqe(H):
    """Runs the variational quantum eigensolver on the problem Hamiltonian using the
    variational ansatz specified above.

    Fill in the missing parts between the # QHACK # markers below to run the VQE.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The ground state energy of the Hamiltonian.
    """
    energy = 0

    # QHACK #
    size = (2 ** (len(H.wires) + 1) - 2,)#len(H.wires),3)
    
    dev = qml.device('default.qubit',wires=len(H.wires))
    # Initialize the quantum device
    np.random.seed(0)
    #params = np.random.uniform(low=-np.pi / 2, high=np.pi , size=size)
    params = np.random.normal(0.001, 0.01, size)
    #params = np.random.normal(0.01, np.pi, size)
    
    # Randomly choose initial parameters (how many do you need?)
    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev, optimize=True)
    
    # Set up a cost function
    #opt = qml.GradientDescentOptimizer(stepsize=0.01)
    opt = qml.AdagradOptimizer(stepsize=0.2)
    # Set up an optimizer
    
#    max_iterations=500
    #previous = 0
    #for i in range(500):
    #index = 0
    while True:
        #previous = energy
        params = opt.step(cost_fn, params)
        
        if time.time() - start_time > 55:
            break
        #if index % 25 == 0:
            #if round(energy, 7) == round(previous, 7):
            #    break
            #print(energy)
            #print(i)
        #index += 1
            
    energy = cost_fn(params)
    '''n=0
    while True:
        params = opt.step(cost_fn, params)
        prev = float(energy)
        energy = cost_fn(params)
        #if n%20==0:
        #    print(energy)
        if round(float(energy), 8) == round(prev, 8) or n == 500:
            break
        n+=1'''

#    conv_tol = 1e-06
    
#    print(params)
#    for n in range(max_iterations):
#        params, prev_energy = opt.step_and_cost(cost_fn, params)
#        energy = cost_fn(params)
#        conv = np.abs(energy - prev_energy)

#        if n % 20 == 0:
#            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

#        if conv <= conv_tol:
#            break
    # Run the VQE by iterating over many steps of the optimizer
    # QHACK #
    return energy
    # Return the ground state energy


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    ground_state_energy = run_vqe(H)
    print(f"{ground_state_energy:.6f}")
    #print(time.time()-start_time)
