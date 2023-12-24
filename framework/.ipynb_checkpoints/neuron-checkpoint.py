#
# A class representing a population of simple neurons
#
class SimpleNeurons(object):

    def __init__(self, n=1, A=None, B=None, C=None, D=None):

        ####################
        # Model parameters #
        ####################

        # The number of neurons
        self.n = n

        # Scale of the membrane recovery (lower values lead to slow recovery)
        if A is None:
            self.A = np.full((n), 0.02, dtype=np.float32)
        else:
            self.A = A
        # Sensitivity of recovery towards membrane potential (higher values lead to higher firing rate)
        if B is None:
            self.B = np.full((n), 0.2, dtype=np.float32)
        else:
            self.B = B
        # Membrane voltage reset value
        if C is None:
            self.C = np.full((n), -65.0, dtype=np.float32)
        else:
            self.C = C
        # Membrane recovery 'boost' after a spike
        if D is None:
            self.D = np.full((n), 8.0, dtype=np.float32)
        else:
            self.D = D
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
            # All neurons start with a value of B * C
            self.u = tf.Variable(self.B*self.C, name='u')

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

        # Membrane potential is reset to C
        v_reset_op = tf.where(has_fired_op, self.C, self.v)

        # Membrane recovery is increased by D
        u_reset_op = tf.where(has_fired_op, tf.add(self.u, self.D), self.u)

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
                         tf.multiply(self.A, tf.subtract(tf.multiply(self.B, v_reset_op), u_reset_op)))

        # Increment membrane potential, and clamp it to the spiking threshold
        # v += dv * dt
        v_op = tf.assign(self.v, tf.minimum(tf.constant(self.SPIKING_THRESHOLD, shape=[self.n]),
                                                 tf.add(v_reset_op, tf.multiply(dv_op, self.dt))))

        # Decrease membrane recovery
        u_op = tf.assign(self.u, tf.add(u_reset_op, tf.multiply(du_op, self.dt)))

        return v_op, u_op
