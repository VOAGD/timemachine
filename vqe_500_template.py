#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    s_wires=[[0, 1, 2], [1, 2, 3]]
    d_wires=[[[0, 1], [2, 3]]]
    hfstate=[1,1,0,0]

    dev = qml.device("default.qubit", wires=len(H.wires))
    
    #def new_ansatz(params, wires):
    #    qml.templates.UCCSD(weights=params,wires=H.wires,s_wires=s_wires,d_wires=d_wires,init_state=hfstate)

    def ansatz(params,wires):
        qml.templates.state_preparations.ArbitraryStatePreparation(params,wires=wires)
    cost_fn = qml.ExpvalCost(ansatz, H, dev, optimize=True)

    
    #opt = qml.GradientDescentOptimizer(stepsize=0.1)
    opt = qml.AdagradOptimizer(stepsize=0.4)
#    opt = qml.optimize.QNGOptimizer(stepsize=0.1)
    np.random.seed(0)  # for reproducibility
    size = (2 ** (len(H.wires) + 1) - 2,)#len(H.wires),3)
    #params = np.random.normal(0, np.pi, len(singles) + len(doubles))
    params = np.random.normal(0.001, 0.01, size=size)
    print(params)

    max_iterations = 140
#    conv_tol = 1e-06
#    index=0
    energy=0

    for n in range(max_iterations):
        params, prev_energy = opt.step_and_cost(cost_fn, params)
        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)
        if n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

        if float(energy) == float(prev_energy) or n == 100:
            break



    # QHACK #

    #return ",".join([str(E) for E in energies])


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
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
