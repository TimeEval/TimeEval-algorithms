import torch
import torch.distributions as dist


def m_elbo(observe_x, observe_z, normal, p_xz: dist.Distribution, q_zx: dist.Distribution, p_z: dist.Distribution):
    """
    :param observe_x: (batch_size, x_dims)
    :param observe_z: (sample_size, batch_size, z_dims) or (batch_size, z_dims,)
    :param normal: (batch_size, x_dims)
    :param p_xz: samples in shape (sample_size, batch_size, x_dims)
    :param q_zx: samples in shape (sample_size, batch_size, z_dims)
    :param p_z: samples in shape (z_dims, )
    :return:
    """
    observe_x = torch.unsqueeze(observe_x, 0)  # (1, batch_size, x_dims)
    normal = torch.unsqueeze(normal, 0)  # (1, batch_size, x_dims)
    log_p_xz = p_xz.log_prob(observe_x)  # (1, batch_size, x_dims)
    if observe_z.dim() == 2:
        torch.unsqueeze(observe_z, 0, observe_z)  # (sample_size, batch_size, z_dims)
    # noinspection PyArgumentList
    log_q_zx = torch.sum(q_zx.log_prob(observe_z), -1)  # (sample_size, batch_size)
    # noinspection PyArgumentList
    log_p_z = torch.sum(p_z.log_prob(observe_z), -1)  # (sample_size, batch_size)
    # noinspection PyArgumentList
    radio = (torch.sum(normal, -1) / float(normal.size()[-1]))  # (1, batch_size)
    # noinspection PyArgumentList
    return - torch.mean(torch.sum(log_p_xz * normal, -1) + log_p_z * radio - log_q_zx)
