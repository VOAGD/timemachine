#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    def state(params):
        qstate = np.zeros([8,])
        
        variational_circuit(params)
        prob0 = qml.probs(0)
        prob1 = qml.probs(1)
        prob2 = qml.probs(2)
        
        index = 0
        for i in range(len(prob0)):
            for j in range(len(prob1)):
                for k in range(len(prob2)):
                    qstate[index] = prob0[i] @ prob1[j] @ prob2[k]
        
        return qstate
        
    def getTensorValue(params, i, j, qstate):
        
        params1 = params.copy()
        params1[i], params1[j] = params[i] + np.pi/2, params[j] + np.pi/2
        
        params2 = params.copy()
        params2[i], params2[j] = params[i] + np.pi/2, params[j] - np.pi/2
        
        params3 = params.copy()
        params3[i], params3[j] = params[i] - np.pi/2, params[j] + np.pi/2
        
        params4 = params.copy()
        params4[i], params4[j] = params[i] - np.pi/2, params[j] - np.pi/2
        
        return (1/8)*(-np.abs(np.dot(qstate.conj().T, state(params1)))**2 + np.abs(np.dot(qstate.conj().T, state(params2)))**2 + np.abs(np.dot(qstate.conj().T, state(params3)))**2 - np.abs(np.dot(qstate.conj().T, state(params4)))**2)
    
    def PST(w, i):
        shifted_g = w.copy()
        shifted_g[i] += np.pi/2
        pst_g_plus = qnode(shifted_g)
        
        shifted_g[i] -= np.pi
        pst_g_minus = qnode(shifted_g)
        
        return (pst_g_plus - pst_g_minus)/(2* np.sin(pi/2))
        
    qstate = state(params)
    
    for i in range(len(tensor)):
        for j in range(len(tensor)):
            tensor[i][j] = getTensorValue(params, i, j, qstate)
            
    for k in range(len(params)):
        gradient[k] = PST(params, i)
        
    tensor_inv = np.linalg.inv(tensor)
        
    natural_grad = tensor_inv.dot(gradient)
    
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
