import nengo
import numpy as np

# Helper function for showing the values as images
def make_display(shape):
    def display_func(t, x, shape=shape):
        import base64
        import PIL
        import cStringIO

        values = x.reshape(shape)
        values = np.clip((values+1)*128, 0, 255)
        values = values.astype('uint8')

        png = PIL.Image.fromarray(values)
        buffer = cStringIO.StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())

        display_func._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%"
                   xlink:href="data:image/png;base64,%s"
                   style="image-rendering: pixelated;">
            </svg>''' % (''.join(img_str))
    return display_func

D = 8

model = nengo.Network()
# uncomment the following line to switch to "Direct" mode (i.e. just do the
#  math, rather than using neurons.  You can also set neuron_type=nengo.LIFRate()
#  for rate-mode neurons, or even nengo.Sigmoid() for sigmoid rate neurons
#model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    stim_gradient = nengo.Node([0]*(D-1))
    
    N = 25  # number of neurons per pixel per map
    
    gradient = nengo.networks.EnsembleArray(n_ensembles=D-1, n_neurons=N, radius=1)

    image = nengo.networks.EnsembleArray(n_ensembles=D, n_neurons=N, radius=1)

    input_s = 1.5  # strength of input stimulus
    s_G2I = 0.3    # strength of coupling from G to I
    s_I2G = 0.3    # strength of coupling from I to G
    syn = 0.01     # synaptic time constant

    # the function going from I to G is G = dot(t_I2G, I)
    t_I2G = np.zeros((D-1, D))
    t_I2G[:,:D-1] += -np.eye(D-1)
    t_I2G[:,1:D] += np.eye(D-1)
    
    # the function going from G to I is I = dot(t_G2I, G)
    t_G2I = np.zeros((D, D-1))
    for i in range(D-1):
        t_G2I[i+1:,i] = 1
        t_G2I[:i+1,i] = -1

    #Connect the input to the gradient
    nengo.Connection(stim_gradient, gradient.input, transform=input_s, synapse=syn)
    nengo.Connection(gradient.output, gradient.input, transform=1.0-input_s, synapse=syn)

    # Connect the gradient to the image and vice-versa
    nengo.Connection(gradient.output, image.input, transform=t_G2I*s_G2I, synapse=syn)
    nengo.Connection(image.output, gradient.input, transform=t_I2G*s_I2G, synapse=syn)
    nengo.Connection(gradient.output, gradient.input, transform=1.0-s_I2G, synapse=syn)
    nengo.Connection(image.output, image.input, transform=1.0-s_G2I, synapse=syn)
    
    # create special populations for showing the values as an image
    # (as we don't have that built in to the GUI yet)
    show_image = nengo.Node(make_display((1, D)), size_in=D)
    nengo.Connection(image.output, show_image, synapse=0.01)
    
    show_grad = nengo.Node(make_display((1, D-1)), size_in=D-1)
    nengo.Connection(gradient.output, show_grad, synapse=0.01)
    