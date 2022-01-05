import logging

from scipy.signal import convolve2d

from torsk.data.conv import get_kernel
from torsk.data.dct import dct2
from torsk.data.utils import resample2d, normalize
from torsk.models.initialize import dense_esn_reservoir, sparse_nzpr_esn_reservoir
from torsk.numpy_accelerate import bh, to_np, to_bh, bh_dot
from torsk.timing import start_timer, end_timer

logger = logging.getLogger(__name__)


def apply_input_map(image, F):
    if F["type"] == "pixels":
        _features = resample2d(image, F["size"])
        F["dbg_size"] = F["size"]
    elif F["type"] == "dct":
        _features = dct2(image, *F["size"]).reshape(-1)
        F["dbg_size"] = F["size"]
    elif F["type"] == "gradient":
        grad = bh.concatenate(bh.gradient(image))
        _features = normalize(grad.reshape(-1)) * 2 - 1
        F["dbg_size"] = grad.shape
    elif F["type"] == "conv":
        _features = convolve2d(
            image, F["kernel"], mode=F['mode'], boundary="symm")
        F["dbg_size"] = _features.shape
        _features = normalize(_features.reshape(-1)) * 2 - 1
    elif F["type"] == "random_weights":
        _features = bh_dot(F["weight_ih"], image.reshape(-1))
        _features += F["bias_ih"]
        F["dbg_size"] = F["size"]
    elif F["type"] == "compose":
        for f in F["operations"]:
            _features = apply_input_map(image, f)
        F["dbg_size"] = hidden_size_of(image.shape,F["operations"][-1])
    else:
        raise ValueError(F)

    if "input_scale" in F:
        scale = F["input_scale"]
    else:
        scale = 1

    _features = scale * _features
    return _features


def input_map(image, operations, timer=None):
    start_timer(timer,"input_map")
    features = []
    for F in operations:
        start_timer(timer,F['type'])
        features.append(apply_input_map(image,F).reshape(-1))
        end_timer(timer)
    end_timer(timer)
    return features


def init_input_map_specs(input_map_specs, input_shape, dtype):
    for spec in input_map_specs:
        if spec["type"] == "conv":
            spec["kernel"] = get_kernel(spec["size"], spec["kernel_type"], dtype)
        elif spec["type"] == "random_weights":
            assert len(spec["size"]) == 1
            weight_ih = bh.random.uniform(low=-1., high=1.,
                size=[spec["size"][0], input_shape[0] * input_shape[1]])
            bias_ih = bh.random.uniform(low=-1., high=1.,
                size=spec["size"])
            spec["weight_ih"] = weight_ih.astype(dtype)
            spec["bias_ih"] = bias_ih.astype(dtype)
        elif spec["type"] == "compose":
            spec["operations"] = init_input_map_specs(
                spec["operations"], input_shape, dtype)
    return input_map_specs

def hidden_size_of(input_shape, F):
    shape = input_shape
    if F["type"] == "conv":
        if F["mode"] == "valid":
#            shape = conv2d_output_shape(input_shape, F["size"])!
            (m,n) = F["size"]
            shape = (input_shape[0]-m+1,input_shape[1]-n+1)
            size  = shape[0] * shape[1]
        elif F["mode"] == "same":
            size  = shape[0] * shape[1]
    elif F["type"] == "random_weights":
        size = F["size"][0]
    elif F["type"] == "gradient":
        size = input_shape[0] * input_shape[1] * 2  # For 2d pictures
    elif F["type"] == "compose":
        shape = input_shape
        for f in F["operations"]:
            size, shape = hidden_size_of(shape,f);
    else:
        shape = F["size"]
        size  = shape[0] * shape[1]
    return size, shape

def get_hidden_size(input_shape, input_map_specs):
    hidden_size = 0
    for F in input_map_specs:
        hidden_size += hidden_size_of(input_shape,F)[0]
    return hidden_size


class NumpyMapESNCell(object):
    """An Echo State Network (ESN) cell that enables custom input -> state
    mappings.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images
    input_map_specs : list
        a list of dicts that specify the input mapping. example:
        [{"type": "random_weights", "size": 500, "weight_scale": 1.0},
         {"type": "pixels", "size": [10, 10]},
         {"type": "dct", "size": [10, 10]},
         {"type": "conv", "size": [2, 2], "kernel_type": "gauss"}]
    spectral_radius : float
        Largest eigenvalue of the reservoir matrix

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

    def __init__(self, input_shape, input_map_specs, spectral_radius, density, dtype, timer=None):
        self.input_shape = input_shape
        self.spectral_radius = spectral_radius
        self.density = density
        self.dtype = bh.dtype(dtype)
        self.input_map_specs = input_map_specs
        self.timer = timer

        self.hidden_size = self.get_hidden_size(input_shape)
        logger.debug(f"ESN hidden size: {self.hidden_size}")
        self.weight_hh = dense_esn_reservoir(
            dim=self.hidden_size, spectral_radius=self.spectral_radius,
            density=self.density, symmetric=False)
        self.weight_hh = self.weight_hh.astype(self.dtype)

        self.input_map_specs = init_input_map_specs(input_map_specs, input_shape, dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return get_hidden_size(input_shape, self.input_map_specs)

    def input_map(self, image):
        return input_map(image, self.input_map_specs)

    def cat_input_map(self, input_stack):
        return bh.concatenate(input_stack, axis=0)

    def state_map(self, state):
        return bh.dot(self.weight_hh, state)

    def forward(self, image, state):
        self.check_dtypes(image, state)

        input_stack = self.input_map(image)
        x_input = self.cat_input_map(input_stack)
        x_state = self.state_map(state)
        new_state = bh.tanh(x_input + x_state)

        return new_state


class NumpyMapSparseESNCell(object):
    def __init__(self, input_shape, input_map_specs, spectral_radius, density, dtype, timer=None):
        self.input_shape = input_shape
        self.input_map_specs = input_map_specs
        self.spectral_radius = spectral_radius
        self.density = density
        self.dtype = bh.dtype(dtype)
        self.timer = timer

        self.hidden_size = self.get_hidden_size(input_shape)
        nonzeros_per_row = int(self.hidden_size * density)
        logger.debug(f"ESN hidden size: {self.hidden_size}, nonzeros_per_row: {nonzeros_per_row}")
        if nonzeros_per_row <= 0:
            raise ValueError(f"There must be at least one connection within the ESN cell, but nonzeros_per_row was {nonzeros_per_row}! "
                             f"Please change the hidden size ({self.hidden_size}) or the density ({density}) parameter.")
        self.weight_hh = sparse_nzpr_esn_reservoir(
            dim=self.hidden_size,
            spectral_radius=self.spectral_radius,
            nonzeros_per_row=nonzeros_per_row,
            dtype=self.dtype, timer=timer)
        #self.weight_hh = sparse_esn_reservoir(
        #    dim=self.hidden_size,
        #    spectral_radius=self.spectral_radius,
        #    #nonzeros_per_row=nonzeros_per_row,
        #    density=density,
        #    symmetric=True)
        #    #dtype=self.dtype)

        self.input_map_specs = init_input_map_specs(input_map_specs, input_shape, dtype)

    def check_dtypes(self, *args):
        for arg in args:
            assert arg.dtype == self.dtype

    def get_hidden_size(self, input_shape):
        return get_hidden_size(input_shape, self.input_map_specs)

    def input_map(self, image):
        return input_map(image, self.input_map_specs, self.timer)

    def state_map(self, state):
        start_timer(self.timer,"state_map")
        new_state = self.weight_hh.sparse_dense_mv(state)
        end_timer(self.timer)
        return new_state

    def forward(self, image, state):
        start_timer(self.timer,"forward")

        self.check_dtypes(image, state)

        input_stack = self.input_map(to_np(image))       # np
        x_input     = to_bh(bh.concatenate(input_stack)) # np -> bh
        x_state     = self.state_map(to_bh(state))       # bh
        start_timer(self.timer,"tanh")
        new_state   = bh.tanh(x_input+x_state)      # bh
        end_timer(self.timer) # /tanh
        end_timer(self.timer) # /forward
        return new_state
