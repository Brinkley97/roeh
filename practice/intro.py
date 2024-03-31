"""Intro to the Nengo library

Tutorial: https://www.youtube.com/watch?v=sgu9l_bqAHM

"""
import nengo

import numpy as np

model = nengo.Network()
with model:
    # stim = nengo.Node([0])
    # a = nengo.Ensemble(n_neurons=50, dimensions=1)
    # nengo.Connection(stim, a)
    
    """Frontend: Ensemble
    
    A population of nodes
    """
    N, d = 100, 1 # N neurons with d dimension
    neuron_type = nengo.LIF() # spiking version of the leaky integrate-and-fire neuron model
    noise = nengo.processes.WhiteNoise()
    lif_model = nengo.Ensemble(n_neurons=N, dimensions=1)
    
    spiking_type = nengo.PoissonSpiking(nengo.Tanh())
    poisson = nengo.Ensemble(n_neurons=N, dimensions=1, neuron_type=spiking_type)
    
    """Frontend: Node, runs an arbitrary python function
    
    Provide non-neural inputs
    Run non-neural functions
    Run route signals (ie: link vision model and motor models with node)
    Connect to external processes/devices (on PC)
    """
    const_node = nengo.Node([0, 0]) # pass in a list or np array
    time_function_node = nengo.Node(lambda t: np.sin(t)) # function of time t to perform operation sin(t)
    # input_function_node = nengo.Node(lambda t, x: x[0] * x[1]) # function of time t to perfrom operation x[0] * x[1]
    passthrough = nengo.Node(None, size_in=3)
