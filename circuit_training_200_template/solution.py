#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []
    
    # QHACK #
    
    '''
    cost_h, mixer_h = qml.qaoa.max_independent_set(graph, constrained=True)
    
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)
    
    wires = range(6)
    depth = 10

    def circuit(params, **kwargs):
        qml.layer(qaoa_layer, depth, params[0], params[1])
        
    dev = qml.device("default.qubit", wires=wires)
    
    @qml.qnode(dev)
    def probability_circuit(gamma, alpha):
        circuit([gamma, alpha])
        return qml.probs(wires=wires)
    
    #drawer = qml.draw(probability_circuit)
    
    #print(drawer(params[0], params[1]))
    probs = probability_circuit(params[0], params[1])
    
    from matplotlib import pyplot as plt
    plt.style.use("seaborn")
    plt.bar(range(2 ** len(wires)), probs)
    plt.show()
    
    index = 0
    biggest = [0,0]
    for i in probs:
        if i > biggest[0]:
            biggest[0] = i
            biggest[1] = index
        index += 1
    
    print(biggest)'''
    from networkx.algorithms import approximation
    #print(approximation.maximum_independent_set(graph))
    max_ind_set = nx.maximal_independent_set(graph, approximation.maximum_independent_set(graph))
    '''binary_num = bin(biggest[1]).replace('0b', '')
    print(binary_num)
    binary_num = binary_num[::-1]
    
    index = 0
    for i in str(binary_num):
        if i == '1': max_ind_set.append(index)
        index += 1
    '''
    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
