import torch
from torch.autograd import Variable
from .variational_autoencoder import VAE


def mcmc_missing_imputation(observe_normal, vae: VAE, n_iteration=10, **inputs):
    assert "observe_x" in inputs
    observe_x = inputs["observe_x"]

    if isinstance(observe_x, Variable):
        test_x = Variable(observe_x.data)
    else:
        test_x = Variable(observe_x)
    with torch.no_grad():
        for mcmc_step in range(n_iteration):
            p_xz, _, _ = vae(**inputs, n_sample=1)
            test_x[observe_normal == 0.] = p_xz.sample()[0][observe_normal == 0.]
    return test_x

