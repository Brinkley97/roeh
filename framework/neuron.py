
import numpy as np

import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from dataclasses import dataclass
# from __future__ import print_function


class SimpleNeurons(object):
    #
    # A class representing a population of simple neurons
    #

    def __init__(self, n=1, membrane_recovery=None, recovery_rate=None, reset_membrane_voltage=None, membrane_post_spike_recovery=None):

        ####################
        # Model parameters #
        ####################

        # The number of neurons
        self.n = n

        # Scale of the membrane recovery (lower values lead to slow recovery)
        if membrane_recovery is None:
            self.membrane_recovery = np.full((n), 0.02, dtype=np.float32)
        else:
            self.membrane_recovery = membrane_recovery

        # Sensitivity of recovery towards membrane potential (higher values lead to higher firing rate)
        if recovery_rate is None:
            self.recovery_rate = np.full((n), 0.2, dtype=np.float32)
        else:
            self.recovery_rate = recovery_rate

        # Membrane voltage reset value
        if reset_membrane_voltage is None:
            self.reset_membrane_voltage = np.full((n), -65.0, dtype=np.float32)
        else:
            self.reset_membrane_voltage = reset_membrane_voltage

        # Membrane recovery 'boost' after a spike
        if membrane_post_spike_recovery is None:
            self.membrane_post_spike_recovery = np.full((n), 8.0, dtype=np.float32)
        else:
            self.membrane_post_spike_recovery = membrane_post_spike_recovery

        # Spiking threshold
        self.SPIKING_THRESHOLD = 35.0
        # self.SPIKING_THRESHOLD = 1.0

        # Resting potential
        self.RESTING_POTENTIAL = -70.0

        # Instantiate a specific tensorflow graph for the Neuron Model
        self.graph = tf.Graph()

        ################################
        # Build the neuron model graph #
        ################################
        with self.graph.as_default():

            ##############################
            # Variables and placeholders #
            ##############################
            self.get_vars_and_ph()

            ##############
            # Operations #
            ##############

            # Operations to evaluate the membrane response (potential v and recovery u)
            self.potential, self.recovery = self.get_response_ops()

    ###############################################
    # Define the graph Variables and placeholders #
    ###############################################
    def get_vars_and_ph(self):

            # Membrane potential
            # All neurons start at the resting potential
            self.v = tf.Variable(tf.constant(self.RESTING_POTENTIAL, shape=[self.n]), name='v')

            # Membrane recovery
            # All neurons start with a value of recovery_rate * reset_membrane_voltage
            self.u = tf.Variable(self.recovery_rate*self.reset_membrane_voltage, name='u')

            # We need a placeholder to pass the input current
            self.I = tf.placeholder(tf.float32, shape=[self.n])

            # We also need a placeholder to pass the length of the time interval
            self.dt = tf.placeholder(tf.float32)

    #######################################################
    # Define the graph of operations to update v and u:   #
    # has_fired_op                                        #
    #   -> (v_reset_op, u_rest_op)      <- I              #
    #      -> (dv_op, du_op)          <- i_op             #
    #        -> (v_op, u_op)                              #
    # We only need to return the leaf operations as their #
    # graph include the others.                           #
    #######################################################

    # This method for future use when we introduce synaptic currents
    def get_input_ops(self, has_fired_op, v_op):

        return tf.add(self.I, 0.0)

    def get_response_ops(self):

        has_fired_op, v_reset_op, u_reset_op = self.get_reset_ops()

        i_op = self.get_input_ops(has_fired_op, v_reset_op)

        v_op, u_op = self.get_update_ops(has_fired_op, v_reset_op, u_reset_op, i_op)

        return v_op, u_op

    def get_reset_ops(self):

        # Evaluate which neurons have reached the spiking threshold
        has_fired_op = tf.greater_equal(self.v, tf.constant(self.SPIKING_THRESHOLD, shape=[self.n]))

        # Neurons that have spiked must be reset, others simply evolve from their initial value

        # Membrane potential is reset to reset_membrane_voltage
        v_reset_op = tf.where(has_fired_op, self.reset_membrane_voltage, self.v)

        # Membrane recovery is increased by membrane_post_spike_recovery
        u_reset_op = tf.where(has_fired_op, tf.add(self.u, self.membrane_post_spike_recovery), self.u)

        return has_fired_op, v_reset_op, u_reset_op

    def get_update_ops(self, has_fired_op, v_reset_op, u_reset_op, i_op):

        # Evaluate membrane potential increment for the considered time interval
        # dv = 0 if the neuron fired, dv = 0.04v*v + 5v + 140 + I -u otherwise
        dv_op = tf.where(has_fired_op,
                         tf.zeros(self.v.shape),
                         tf.subtract(tf.add_n([tf.multiply(tf.square(v_reset_op), 0.04),
                                               tf.multiply(v_reset_op, 5.0),
                                               tf.constant(140.0, shape=[self.n]),
                                               i_op]),
                                     self.u))

        # Evaluate membrane recovery decrement for the considered time interval
        # du = 0 if the neuron fired, du = a*(b*v -u) otherwise
        du_op = tf.where(has_fired_op,
                         tf.zeros([self.n]),
                         tf.multiply(self.membrane_recovery, tf.subtract(tf.multiply(self.recovery_rate, v_reset_op), u_reset_op)))

        # Increment membrane potential, and clamp it to the spiking threshold
        # v += dv * dt
        v_op = tf.assign(self.v, tf.minimum(tf.constant(self.SPIKING_THRESHOLD, shape=[self.n]),
                                                 tf.add(v_reset_op, tf.multiply(dv_op, self.dt))))

        # Decrease membrane recovery
        u_op = tf.assign(self.u, tf.add(u_reset_op, tf.multiply(du_op, self.dt)))

        return v_op, u_op

    def simulate_session(self, neuron_model_tf_graph, T: int, dt: float, steps: int):
        # Array of input current values
        I_in = []

        # Array of evaluated membrane potential values
        v_out = []

        with tf.Session(graph=neuron_model_tf_graph) as sess:

            # Initialize global variables to their default values
            sess.run(tf.global_variables_initializer())

            # Run the simulation at each time step
            for step in steps:

                t = step*dt

                # We generate a current step of 7 A (membrane_recovery) between 200 and 700 ms
                if t > 200 and t < 700:
                    i_in = 7.0
                else:
                    i_in = 0.0

                # Create the dictionary of parameters to use for this time step
                feed = {self.I: np.full((1), i_in), self.dt: [dt]}

                # Run the neuron response operations, passing our parameters
                v, u = sess.run([self.potential, self.recovery], feed_dict=feed)

                # Store values
                I_in.append((t, i_in))
                v_out.append((t, v))

        return I_in, v_out


class SimpleSynapticNeurons(SimpleNeurons):
    #
    # A class representing a population of simple neurons with synaptic inputs
    #

    def __init__(self, n=1, m=100, membrane_recovery=None, recovery_rate=None, reset_membrane_voltage=None, membrane_post_spike_recovery=None, W_in=None):

        # Additional model parameters
        self.m = m
        self.tau = 10.0
        if W_in is None:
            self.W_in = np.full((n,m), 0.07, dtype=np.float32)
        else:
            self.W_in = W_in
        # The reason this one is different is to allow broadcasting when subtracting v
        self.E_in = np.zeros((m), dtype=np.float32)

        # Call the parent contructor
        # This will call the methods we have overidden when building the graph
        super(SimpleSynapticNeurons, self).__init__(n, membrane_recovery, recovery_rate, reset_membrane_voltage, membrane_post_spike_recovery)

    ########################################################
    # Override the parent graph Variables and placeholders #
    ########################################################
    def get_vars_and_ph(self):

        # Get parent grah variables and placeholders
        super(SimpleSynapticNeurons, self).get_vars_and_ph()

        # Input synapse conductance dynamics (increases on each synapse spike)
        self.g_in = tf.Variable(tf.zeros(dtype=tf.float32, shape=[self.m]),
                                    dtype=tf.float32,
                                    name='g_in')

        # We need a new placeholder to pass the input synapses behaviour at each timestep
        self.syn_has_spiked = tf.placeholder(tf.bool, shape=[self.m])

    #######################################################
    # Modify i_op in the graph of operations:             #
    #     syn_has_spiked -> g_in_op -> i_op               #
    #######################################################
    def get_input_ops(self, has_fired_op, v_op):

        # First, update synaptic conductance dynamics:
        # - increment by one the current factor of synapses that fired
        # - decrease by tau the conductance dynamics in any case
        g_in_update_op = tf.where(self.syn_has_spiked,
                                  tf.add(self.g_in, tf.ones(shape=self.g_in.shape)),
                                  tf.subtract(self.g_in, tf.multiply(self.dt,tf.divide(self.g_in, self.tau))))

        # Update the g_in variable
        g_in_op = tf.assign(self.g_in, g_in_update_op)

        # We can now evaluate the synaptic input currents
        # Isyn = Σ w_in(j)g_in(j)E_in(j) - (Σ w_in(j)g_in(j)).v(t)
        i_op = tf.subtract(tf.einsum('nm,m->n', tf.constant(self.W_in), tf.multiply(g_in_op, tf.constant(self.E_in))),
                           tf.multiply(tf.einsum('nm,m->n', tf.constant(self.W_in), g_in_op), v_op))

        # Store a reference to this operation for easier retrieval
        self.input = i_op

        return i_op

    def simulate_session(self, neuron_model_tf_graph, T: int, dt: float, steps: int, synapses_firing_rate: float):
        # Array of input current values
        I_in = []

        # Array of evaluated membrane potential values
        v_out = []

        with tf.Session(graph=neuron_model_tf_graph) as sess:

            # Initialize v and u to their default values
            sess.run(tf.global_variables_initializer())

            # Run the simulation at each time step
            for step in steps:

                t = step * dt
                # We generate random spikes on the input synapses between 200 and 700 ms
                if t > 200 and t < 700:
                    # Generate a random matrix
                    r = np.random.uniform(0,1,(self.m))
                    # A synapse has spiked when r is lower than the spiking rate
                    p_syn_spike = r < synapses_firing_rate * dt
                else:
                    # No synapse activity during that period
                    p_syn_spike = np.zeros((self.m), dtype=bool)

                feed = {self.syn_has_spiked: p_syn_spike, self.dt: [dt]}

                # Run the graph corresponding to our update ops, with our parameters
                i, v, u = sess.run([self.input, self.potential, self.recovery], feed_dict=feed)

                # Store values
                I_in.append((t,i))
                v_out.append((t,v))

        return I_in, v_out

@dataclass
class CreatePlots:

    def plot_simulation(I_in: list, v_out, dt, simulation_type):
        plt.rcParams["figure.figsize"] =(6,6)
        # Draw the input current and the membrane potential
        plt.figure()
        plt.title('Input current')
        plt.ylabel('Current (mA)')
        plt.xlabel('Time (msec)')

        if simulation_type == "single_neuron_synaptic_input":
            _, i_mean = np.mean(np.array(I_in)[int(200/dt):int(700/dt),:], axis=0)
            plt.axhline(y=i_mean, color='y', linestyle='--')

        plt.plot(*zip(*I_in))
        plt.figure()
        plt.title('Neuron response')
        plt.ylabel('Membrane Potential (mV)')
        plt.xlabel('Time (msec)')
        plt.plot(*zip(*v_out))
