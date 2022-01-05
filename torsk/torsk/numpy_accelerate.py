import logging
import numpy as np
from torsk import config

logger = logging.getLogger(__name__)


def Id(A):
    return A


def void():
    return


if config.numpy_acceleration == "bohrium":
    import bohrium

    bh = bohrium
    bh_type = bh.ndarray
    bh_check = bh.bhary.check
    to_bh = bh.array
    bh_flush = bh.flush
    bh_diag = bh.diag
    bh_dot = bh.dot


    def to_np(A):
        if bh_check(A):
            return A.copy2numpy()
        else:
            return A


    accel_capabilities = ["streaming"]

elif config.numpy_acceleration == "bh107":
    import bh107

    bh = bh107
    bh_type = bh.BhArray
    bh_check = lambda A: isinstance(A, bh.BhArray)
    to_bh = bh.BhArray.from_numpy
    bh_flush = bh.flush
    bh_diag = lambda A: to_bh(np.diag(A.asnumpy()))
    bh_dot = lambda A, B: to_bh(np.dot(to_np(A), to_np(B)))


    def to_np(A):
        if isinstance(A, bh_type):
            bh.flush()  # Make sure everything pending is already written to A
            return A.asnumpy()
        else:
            return A


    accel_capabilities = ["streaming"]

else:
    bh = np
    bh_type = bh.ndarray
    bh_check = lambda A: isinstance(A, bh_type);
    to_bh = Id
    to_np = Id
    bh_flush = void
    bh_diag = np.diag
    bh_dot = np.dot
    accel_capabilities = []


def before_storage(model):
    numpyize(model)


def after_storage(model, old_state=None):
    bohriumize(model, old_state)


def _get_all_class_attrs(model):
    """Returns a list of dicts with most of the model attribute/value pairs."""
    attrs = [model.__dict__, model.esn_cell.__dict__]
    attrs += model.esn_cell.input_map_specs
    if model.params.reservoir_representation == "sparse":
        attrs += [model.esn_cell.weight_hh.__dict__]
    return attrs


def bohriumize(model, old_state=None):
    logger.debug("Bohriumizing...")
    # TODO: is the old_state still needed?
    # if old_state is None:
    #    if model.params.reservoir_representation == "dense":
    #        model.esn_cell.weight_hh = to_bh(model.esn_cell.weight_hh)
    #    else:
    #        model.esn_cell.weight_hh.values  = to_bh(model.esn_cell.weight_hh.values)
    #        model.esn_cell.weight_hh.col_idx = to_bh(model.esn_cell.weight_hh.col_idx)
    #    model.ones = to_bh(model.ones)
    #    model.wout = to_bh(model.wout)
    # else:
    #    if model.params.reservoir_representation == "dense":
    #        model.esn_cell.weight_hh = old_state["esn_cell.weight_hh"]
    #    else:
    #        model.esn_cell.weight_hh.values  = old_state["esn_cell.weight_hh.values"]
    #        model.esn_cell.weight_hh.col_idx = old_state["esn_cell.weight_hh.col_idx"]
    #    model.ones = old_state["ones"]
    #    model.wout = old_state["wout"]

    attrs = _get_all_class_attrs(model)
    for D in attrs:
        for k, v in D.items():
            if isinstance(v, np.ndarray):
                if old_state is None:
                    D[k] = to_bh(v)
                else:
                    D[k] = old_state[k]

    logger.debug("Done bohriumizing...")


def numpyize(model):
    logger.debug("Numpyizing...")
    # old_state = {};
    # TODO: is the old_state still needed?
    # if model.params.reservoir_representation == "dense":
    #    old_state["esn_cell.weight_hh"] = model.esn_cell.weight_hh
    #    model.esn_cell.weight_hh = to_np(model.esn_cell.weight_hh)
    # else:
    #    old_state["esn_cell.weight_hh.values"]  = model.esn_cell.weight_hh.values;
    #    old_state["esn_cell.weight_hh.col_idx"] = model.esn_cell.weight_hh.col_idx;
    #    model.esn_cell.weight_hh.values  = to_np(model.esn_cell.weight_hh.values)
    #    model.esn_cell.weight_hh.col_idx = to_np(model.esn_cell.weight_hh.col_idx)

    # old_state["ones"] = model.ones;
    # old_state["wout"] = model.wout;
    # model.ones = to_np(model.ones)
    # model.wout = to_np(model.wout)

    attrs = _get_all_class_attrs(model)

    for D in attrs:
        for k, v in D.items():
            if isinstance(v, bh_type):
                logger.debug(f"Numpyizing `{k}`")
                D[k] = to_np(v)
    # return old_state
