import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.
    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.
    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).
            * gradient is a real NumPy array of size (5,).
            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)
    
    def PST(w, i):
        shifted_g = w.copy()
        shifted_g[i] += np.pi/2
        pst_g_plus = circuit(shifted_g)
        
        shifted_g[i] -= np.pi
        pst_g_minus = circuit(shifted_g)
        
        return 0.5 * (pst_g_plus - pst_g_minus)
    
    def second_PST(w, i, j):
        shifted_nd = w.copy()
        
        #shifted_nd[i] += np.pi/2
        #pst_nd_plus = PST(shifted_nd, j)
        
        shifted_nd[i] -= np.pi
        pst_nd_minus = PST(shifted_nd, j)
        
        return (gradient[j] - pst_nd_minus) / (np.sin(np.pi/2))
    
    for k in range(len(gradient)):
            gradient[k] = PST(weights, k)
    
    arbitrary = 0
    for i in range(len(hessian)):
            for j in range(len(hessian[i])):
                if hessian[i][j] == 0 and j >= arbitrary:
                    hessian[i][j] = second_PST(weights, i, j)
                    hessian[j][i] = hessian[i][j]
            arbitrary += 1
    
    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
