import nengo
import numpy as np


model = nengo.Network()
with model:
    stim_gradient = nengo.Node([0,1,0])
    
    gradient = nengo.networks.EnsembleArray(n_ensembles=3, n_neurons=50)
    
    
    
    image = nengo.networks.EnsembleArray(n_ensembles=4, n_neurons=50)
    
    transform = np.zeros((4, 3))
    transform[:3,:] += -np.eye(3)
    transform[1:4,:] += np.eye(3)
    
    input_s = 1.0
    s = 0.1
    syn = 0.01
    
    nengo.Connection(stim_gradient, gradient.input, transform=input_s, synapse=syn)
    nengo.Connection(gradient.output, gradient.input, transform=1.0-input_s, synapse=syn)


    nengo.Connection(gradient.output, image.input, transform=transform*s, synapse=syn)
    
    nengo.Connection(image.output, gradient.input, transform=transform.T*s, synapse=syn)
    
    nengo.Connection(gradient.output, gradient.input, transform=1.0-s, synapse=syn)
    nengo.Connection(image.output, image.input, transform=1.0-s, synapse=syn)
    