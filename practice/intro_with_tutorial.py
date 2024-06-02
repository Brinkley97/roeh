import nengo

# a container to put stuff/components in
model = nengo.Network()

# auto put into container
with model:
    
    # Define a group of neurons
    # We 100 neurons working together to represent one thing
    ensemble_1 = nengo.Ensemble(n_neurons=100, dimensions=1)