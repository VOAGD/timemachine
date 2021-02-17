#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pennylane as qml
from pennylane import numpy as np
import sys

def simple_circuits_30(angle):
    qml.RY(angle, wires= 0)
    return qml.expval(qml.PauliX(0))

    x_expectation = 0.0

    # QHACK #

    # Step 1 : initialize a device
    dev = qml.device('default.qubit', wires= 2)

    # Step 2 : Create a quantum circuit and qnode
    circuit = qml.QNode(simple_circuits_30, dev)

    # Step 3 : Run the qnode
    result = circuit(angle)
    
    x_expectation = circuit
    

    # QHACK #
    return x_expectation


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    angle_str = sys.stdin.read()
    angle = float(angle_str)

    ans = simple_circuits_30(angle)

    if isinstance(ans, np.tensor):
        ans = ans.item()

    if not isinstance(ans, float):
        raise TypeError("the simple_circuits_30 function needs to return a float")

    print(ans)


# In[ ]:




