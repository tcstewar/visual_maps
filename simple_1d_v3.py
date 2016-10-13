import nengo
import numpy as np

D = 4

model = nengo.Network()
#model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    stim_gradient = nengo.Node([0]*(D-1))
    
    gradient = nengo.networks.EnsembleArray(n_ensembles=D-1, n_neurons=50, radius=1)
    
    
    
    image = nengo.networks.EnsembleArray(n_ensembles=D, n_neurons=50, radius=1)
    
    transform = np.zeros((D, D-1))
    transform[:D-1,:] += -np.eye(D-1)
    transform[1:D,:] += np.eye(D-1)
    
    input_s = 2.0
    s = 0.3
    syn = 0.01
    
    t2 = np.zeros((D, D-1))
    for i in range(D-1):
        t2[i+1:,i] = 1
    print t2
    t3 = np.zeros((D, D-1))
    for i in range(D-1):
        t3[:i+1,i] = -1
    print t3
    
    nengo.Connection(stim_gradient, gradient.input, transform=input_s, synapse=syn)
    nengo.Connection(gradient.output, gradient.input, transform=1.0-input_s, synapse=syn)


    nengo.Connection(gradient.output, image.input, transform=(t2+t3)*s, synapse=syn)
    
    nengo.Connection(image.output, gradient.input, transform=transform.T*s, synapse=syn)
    
    nengo.Connection(gradient.output, gradient.input, transform=1.0-s, synapse=syn)
    nengo.Connection(image.output, image.input, transform=1.0-s, synapse=syn)
    