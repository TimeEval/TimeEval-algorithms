import logging
import numpy as np

from torsk.models.initialize import dense_esn_reservoir
from torsk.models.numpy_map_esn import NumpyMapESNCell, NumpyMapSparseESNCell
from torsk.imed import metric_matrix
from torsk.timing import Timer
from torsk.data.utils import eigh
import torsk.models.numpy_optimize as opt
from torsk.numpy_accelerate import bh, to_bh, to_np, bh_dot

logger = logging.getLogger(__name__)


class NumpyESN(object):
    """Complete ESN with output layer. Only supports batch=1 for now!!!

    Parameters
    ----------
    params : torsk.utils.Params
        The network hyper-parameters

    Inputs
    ------
    inputs : Tensor
        Inputs of shape (seq, batch, input_size)
    state : Tensor
        Initial state of the ESN with shape (batch, hidden_size)
    nr_predictions : int
        Number of steps to predict into the future

    Outputs
    -------
    outputs : Tensor
        Predicitons nr_predictions into the future
        shape (seq, batch, input_size)
    states' : Tensor
        Accumulated states of the ESN with shape (seq, batch, hidden_size)
    """
    def __init__(self, params):
        super(NumpyESN, self).__init__()
        self.params = params

        if params.reservoir_representation == "dense":
            ESNCell = NumpyMapESNCell
        elif params.reservoir_representation == "sparse":
            ESNCell = NumpyMapSparseESNCell
        else:
            raise ValueError("Unknown reservoir representation; use one of 'dense' or 'sparse'")

        self.timer  = Timer(
            timing_depth=params.timing_depth,
            root_context="numpy_esn", flush=True)

        self.esn_cell = ESNCell(
            input_shape=params.input_shape,
            input_map_specs=params.input_map_specs,
            spectral_radius=params.spectral_radius,
            density=params.density,
            dtype=params.dtype, timer=self.timer)

        input_size = params.input_shape[0] * params.input_shape[1]
        wout_shape = [input_size, self.esn_cell.hidden_size + input_size + 1]
        self.wout = bh.zeros(wout_shape, dtype=self.esn_cell.dtype)

        self.ones = bh.ones([1], dtype=self.esn_cell.dtype)
        self.imed_G = None
        self.imed_w = None
        self.imed_V = None

    def forward(self, inputs, state=None, states_only=True):
        if state is None:
            state = bh.zeros([self.esn_cell.hidden_size], dtype=self.esn_cell.dtype)
        if self.params.debug:
            logger.debug("Calling forward function in debug mode")
            return self._forward_debug(inputs, state)
        if states_only:
            return self._forward_states_only(inputs, state)
        else:
            return self._forward(inputs, state)

    def _forward_states_only(self, inputs, state):
        self.timer.begin("forward_states_only")
        (T, H) = (inputs.shape[0], state.shape[0])

        inputs = to_bh(inputs)
        states = bh.empty((T,H),dtype=state.dtype)
        state  = to_bh(state)

        for i in range(T):
            state = self.esn_cell.forward(inputs[i], state)
            states[i] = to_bh(state)

        self.timer.end()
        return None, states

    def _forward(self, inputs, state):
        self.timer.begin("forward")
        (T,H)  = (inputs.shape[0],state.shape[0])
        states  = bh.empty((T,H),dtype=state.dtype)
        outputs = bh.empty(inputs.shape,dtype=inputs.dtype)

        for i in range(T):
            inp        = inputs[i]
            state      = self.esn_cell.forward(inp, state)
            ext_state  = bh.concatenate((self.ones, inp.reshape(-1), state), axis=0)
            outputs[i] = bh.dot(self.wout, ext_state).reshape(inputs.shape[1:])
            states[i]  = state

        self.timer.end()
        return outputs, states

    def _forward_debug(self, inputs, state):
        from torsk.visualize import plot_iteration
        self.timer.begin("forward_debug")

        (T,H) = (inputs.shape[0],state.shape[0])
        states  = bh.empty((T,H),dtype=state.dtype)
        outputs = bh.empty(inputs.shape,dtype=inputs.dtype)

        for i in range(T):
            inp        = inputs[i]
            new_state  = self.esn_cell.forward(inp, state)
            ext_state  = bh.concatenate([self.ones, inp.reshape(-1), new_state], axis=0)
            outputs[i] = bh.dot(self.wout, ext_state).reshape(inputs.shape[1:])
            states[i]  = state

            if (i == 200):
                plot_iteration(self, i, inp, state)

            state = new_state   # TODO: What's going on here?

        self.timer.end()
        return outputs, states

    def predict(self, initial_input, initial_state, nr_predictions):
        self.timer.begin("predict")

        (T,H) = (nr_predictions,initial_state.shape[0])
        (M,N) = initial_input.shape
        inp   = to_bh(initial_input)
        state = to_bh(initial_state)

        states  = bh.empty((T,H),dtype=state.dtype)
        outputs = bh.empty((T,M,N),dtype=inp.dtype)

        for i in range(T):
            state = self.esn_cell.forward(inp, state)
            S = to_np(state)
            I = to_np(inp).reshape(-1)
            O = to_np(self.ones)
            #            ext_state = bh.concatenate([self.ones, inp.reshape(-1), state], axis=0)
            ext_state = to_bh(np.concatenate([O,I,S], axis=0))
            output = bh_dot(self.wout, ext_state).reshape((M,N))

            inp        = output
            outputs[i] = to_bh(output)
            states[i]  = to_bh(state)

        self.timer.end()
        return outputs, states

    def optimize(self, inputs, states, labels):
        """Train the output layer.

        Parameters
        ----------
        inputs : Tensor
            A batch of inputs with shape (batch, ydim, xdim)
        states : Tensor
            A batch of hidden states with shape (batch, hidden_size)
        labels : Tensor
            A batch of labels with shape (batch, ydim, xdim)
        """
        self.timer.begin("optimize")
        method = self.params.train_method
        beta = self.params.tikhonov_beta

        train_length = inputs.shape[0]
        flat_inputs = inputs.reshape([train_length, -1])
        flat_labels = labels.reshape([train_length, -1])

        if self.params.imed_loss:
            if self.imed_G is None:
                self.timer.begin("IMED initialize")
                self.imed_G              = metric_matrix(inputs.shape[1:])
                self.imed_w, self.imed_V = eigh(self.imed_G)
                self.timer.end()

            self.timer.begin("IMED prepare")
            w,V = self.imed_w, self.imed_V
            s   = np.sqrt(w)
            G12 = np.dot(V,s[:,None]*V.T)
            self.timer.end()

            self.timer.begin("IMED matmul on inputs and labels")
            flat_inputs = np.matmul(G12, flat_inputs[:,:,None])[:,:,0]
            flat_labels = np.matmul(G12, flat_labels[:,:,None])[:,:,0]
            self.timer.end()

        if method == 'tikhonov':
            if beta is None:
                raise ValueError(
                    'For Tikhonov training the beta parameter cannot be None.')
            logger.debug(f"Tikhonov optimizing with beta={beta}")
            wout = opt.tikhonov(flat_inputs, states, flat_labels, beta)

        elif 'pinv' in method:
            if beta is not None:
                logger.debug("With pseudo inverse training the "
                             "beta parameter has no effect.")
            logger.debug(f"Pinv optimizing with mode={method}")
            wout = opt.pseudo_inverse(
                flat_inputs, states, flat_labels,
                mode=method.replace("pinv_", ""), timer=self.timer)

        else:
            raise ValueError(f'Unkown training method: {method}')

        if self.params.imed_loss:
            self.timer.begin("IMED matmul on wout")
            s      = 1/bh.sqrt(w)
            invG12 = bh_dot(V,s[:,None]*V.T)
            wout   = bh_dot(invG12,wout)
            self.timer.end()

        if(wout.shape != self.wout.shape):
            raise ValueError("Optimized and original Wout shape do not match."
                             f"{wout.shape} / {self.wout.shape}")

        self.timer.end()        # optimize
        self.wout = wout


class NumpyStandardESNCell(object):
    """An Echo State Network (ESN) cell.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of features in the hidden state
    spectral_radius : float
        Largest eigenvalue of the reservoir matrix
    in_weight_init : float
        Input matrix will be chosen from a random uniform like
        (-in_weight_init, in_weight_init)
    in_bias_init : float
        Input matrix will be chosen from a random uniform like
        (-in_bias_init, in_bias_init)


    Inputs
    ------
    input : array
        contains input features of shape (batch, input_size)
    state : array
        current hidden state of shape (batch, hidden_size)

    Outputs
    -------
    state' : array
        contains the next hidden state of shape (batch, hidden_size)
    """
    def __init__(self, input_size, hidden_size, spectral_radius,
                 in_weight_init, in_bias_init, density, dtype):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = bh.dtype(dtype)

        self.weight_ih = to_bh(np.random.uniform(
            low=-in_weight_init,
            high=in_weight_init,
            size=[hidden_size, input_size])).astype(dtype)

        self.weight_hh = dense_esn_reservoir(
            dim=hidden_size, spectral_radius=spectral_radius,
            density=density, symmetric=False)
        self.weight_hh = self.weight_hh.astype(dtype)

        self.bias_ih = to_bh(np.random.uniform(
            low=-in_bias_init,
            high=in_bias_init,
            size=[hidden_size])).astype(dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def forward(self, inputs, state):
        self.check_dtypes(inputs, state)

        # next state
        x_inputs  = bh.dot(self.weight_ih, inputs)
        x_state   = bh.dot(self.weight_hh, state)
        new_state = bh.tanh(x_inputs + x_state + self.bias_ih)

        return new_state
