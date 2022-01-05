from typing import *
from scipy.special import erf
import numpy as np
import torch
import torch.distributions as dist
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from .evaluation_metric import ignore_missing
from .network import MultiLinearGaussianStatistic
from .network.loop import Loop, TestLoop
from .torch_util import VstackDataset
from .donutx import CVAE, m_elbo, mcmc_missing_imputation, VAE, BasicVAE
from .kpi_frame_dataloader import KpiFrameDataLoader
from .kpi_frame_dataset import TimestampDataset, KpiFrameDataset
from .kpi_series import KPISeries
from .evaluation_metric import range_lift_with_delay
from .early_stopping import EarlyStopping
import sklearn


def threshold_ml(indicators: np.ndarray, labels: np.ndarray, delay=None, factor=100, prior_best=None, return_statistics=False, return_fscore=False):
    assert np.shape(indicators) == np.shape(labels), f"indicator and label's shape must be equal. indicator:{np.shape(indicators)}, label:{np.shape(labels)}"
    assert np.ndim(indicators) == 1, f"indicator and label must be 1-d array like object. indicator:{np.shape(indicators)}"
    _min, _max = np.min(indicators), np.max(indicators)
    indicators_ = (indicators - _min) / (_max - _min + 1e-8)

    labeled_idx = np.where(labels != -1)[0]
    unlabeled_idx = np.where(labels == -1)[0]

    alpha = len(labeled_idx) / len(labels)

    _indicators_labeled = range_lift_with_delay(indicators_[labeled_idx], labels[labeled_idx], delay=delay)
    _ps, _rs, _ts = sklearn.metrics.precision_recall_curve(labels[labeled_idx], _indicators_labeled)
    _fs = 2.0 * _ps * _rs / np.clip(_ps + _rs, a_min=1e-4, a_max=None)
    thresholds = np.concatenate([_ts, [2.0]])
    f1_scores = _fs

    idx = np.argsort(thresholds)
    thresholds = thresholds[idx]
    f1_scores = f1_scores[idx]
    # print("\nbefore", np.max(f1_scores), thresholds[np.argmax(f1_scores)])
    # _ts = np.linspace(np.min(thresholds), np.max(thresholds), 10000)
    _ts = np.linspace(0, 1, 10000)
    _fs = f1_scores[np.searchsorted(thresholds, _ts, side="left")]
    thresholds = np.concatenate([_ts, thresholds])
    f1_scores = np.concatenate([_fs, f1_scores])

    idx = np.argsort(thresholds)[::-1]
    thresholds = thresholds[idx]
    f1_scores = f1_scores[idx]
    # print("after", np.max(f1_scores), thresholds[np.argmax(f1_scores)])

    if len(unlabeled_idx) > 0:
        if prior_best is None:
            unlabeled_prior_best = threshold_prior(indicators_[unlabeled_idx])
        else:
            unlabeled_prior_best = (prior_best - _min) / (_max - _min)
        unlabeled_prior = np.abs(thresholds - unlabeled_prior_best)
        unlabeled_prior = 1. - unlabeled_prior
        likelihood = (f1_scores * alpha + unlabeled_prior * (1.-alpha))
    else:
        likelihood = f1_scores
    idx = np.argmax(likelihood)
    # print("f1 score", sklearn.metrics.f1_score(labels[labeled_idx], indicators_[labeled_idx] >= thresholds[idx]))
    # if idx > 0:
    #     threshold = (thresholds[idx - 1] * (factor - 1) + thresholds[idx]) / factor
    # else:
    threshold = thresholds[idx]
    # print("f1 score", sklearn.metrics.f1_score(labels[labeled_idx], indicators_[labeled_idx] >= threshold))
    threshold = threshold * (_max - _min) + _min
    # print("fscore", sklearn.metrics.f1_score(labels, indicators >= threshold))
    if return_statistics:
        return threshold, likelihood, thresholds
    elif return_fscore:
        return threshold, np.max(likelihood)
    else:
        return threshold


class DonutX:
    def __init__(self, max_epoch: int = 150, batch_size: int = 128, network_size: List[int] = None,
                 latent_dims: int = 8, window_size: int = 120, cuda: bool = True, condition_dropout_left_rate=0.9,
                 print_fn=print, early_stopping_patience: int = 10, early_stopping_delta: float = 1e-2):
        if network_size is None:
            network_size = [100, 100]

        self.print_fn = print_fn
        self.condition_size = 60 + 24 + 7
        self.window_size = window_size
        self.latent_dims = latent_dims
        self.network_size = network_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.cuda = cuda
        self.condition_dropout_left_rate = condition_dropout_left_rate
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        self._model = CVAE(
            MultiLinearGaussianStatistic(
                self.window_size + self.condition_size, self.latent_dims, self.network_size, eps=1e-4),
            MultiLinearGaussianStatistic(
                self.latent_dims + self.condition_size, self.window_size, self.network_size, eps=1e-4),
        )
        if self.cuda:
            self._model = self._model.cuda()

        if cuda:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32)).cuda()),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)).cuda())
            )
        else:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32))),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)))
            )

    def fit(self, kpi: KPISeries, valid_kpi: KPISeries = None, callbacks: Optional[List[Callable]] = None):
        bernoulli = torch.distributions.Bernoulli(probs=self.condition_dropout_left_rate)
        self._model.train()
        with Loop(max_epochs=self.max_epoch, use_cuda=self.cuda, disp_epoch_freq=5,
                  print_fn=self.print_fn).with_context() as loop:
            optimizer = Adam(self._model.parameters(), lr=1e-3)
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
            train_timestamp_dataset = TimestampDataset(
                kpi, frame_size=self.window_size)
            train_kpiframe_dataset = KpiFrameDataset(kpi,
                                                     frame_size=self.window_size, missing_injection_rate=0.01)
            train_dataloader = KpiFrameDataLoader(VstackDataset(
                [train_kpiframe_dataset, train_timestamp_dataset]), batch_size=self.batch_size, shuffle=True,
                drop_last=True)
            if valid_kpi is not None:
                valid_timestamp_dataset = TimestampDataset(
                    valid_kpi.label_sampling(0.), frame_size=self.window_size)
                valid_kpiframe_dataset = KpiFrameDataset(valid_kpi,
                                                         frame_size=self.window_size, missing_injection_rate=0.)
                valid_dataloader = KpiFrameDataLoader(VstackDataset(
                    [valid_kpiframe_dataset, valid_timestamp_dataset]), batch_size=256, shuffle=True)
            else:
                valid_dataloader = None

            early_stopping = EarlyStopping(delta=self.early_stopping_delta, patience=self.early_stopping_patience, epochs=self.max_epoch, callbacks=callbacks)
            early_stopping_iter = iter(early_stopping)

            for epoch in loop.iter_epochs():
                next(early_stopping_iter)
                for _, batch_data in loop.iter_steps(train_dataloader):
                    optimizer.zero_grad()
                    observe_x, observe_normal, observe_y = batch_data
                    if self.cuda:
                        mask = bernoulli.sample(sample_shape=observe_y.size()).cuda()
                    else:
                        mask = bernoulli.sample(sample_shape=observe_y.size())

                    observe_y = observe_y * mask
                    p_xz, q_zx, observe_z = self._model(
                        observe_x=observe_x, observe_y=observe_y)
                    loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                  self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                    loss.backward()
                    clip_grad_norm_(self._model.parameters(), max_norm=10.)
                    optimizer.step()
                    loop.submit_metric("train_loss", loss.data)
                lr_scheduler.step()
                if valid_kpi is not None:
                    with torch.no_grad():
                        val_loss = []
                        for _, batch_data in loop.iter_steps(valid_dataloader):
                            observe_x, observe_normal, observe_y = batch_data  # type: Variable, Variable
                            p_xz, q_zx, observe_z = self._model(
                                observe_x=observe_x, observe_y=observe_y)
                            loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                          self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                            loop.submit_metric("valid_loss", loss.data)
                            val_loss.append(loss.data)
                        early_stopping.update(np.mean(val_loss).item())

    def predict(self, kpi: KPISeries, return_statistics=False, indicator_name="indicator"):
        """
        :param kpi:
        :param return_statistics:
        :param indicator_name:
            default "indicator": Reconstructed probability
            "indicator_prior": E_q(z|x)[log p(x|z) * p(z) / q(z|x)]
            "indicator_erf": erf(abs(x - x_mean) / x_std * scale_factor)
        :return:
        """
        with torch.no_grad():
            with TestLoop(use_cuda=self.cuda, print_fn=self.print_fn).with_context() as loop:
                test_timestamp_dataset = TimestampDataset(kpi, frame_size=self.window_size)
                test_kpiframe_dataset = KpiFrameDataset(kpi, frame_size=self.window_size, missing_injection_rate=0.0)
                test_dataloader = KpiFrameDataLoader(VstackDataset(
                    [test_kpiframe_dataset, test_timestamp_dataset]), batch_size=32, shuffle=False, drop_last=False)
                self._model.eval()
                for _, batch_data in loop.iter_steps(test_dataloader):
                    observe_x, observe_normal, observe_y = batch_data  # type: Variable, Variable
                    observe_x = mcmc_missing_imputation(observe_normal=observe_normal,
                                                        vae=self._model,
                                                        n_iteration=10,
                                                        observe_x=observe_x,
                                                        observe_y=observe_y
                                                        )
                    p_xz, q_zx, observe_z = self._model(observe_x=observe_x,
                                                        n_sample=128,
                                                        observe_y=observe_y)
                    loss = m_elbo(observe_x, observe_z, observe_normal,
                                  p_xz, q_zx, self.z_prior_dist)  # type: Variable
                    loop.submit_metric("test_loss", loss.data.cpu())

                    log_p_xz = p_xz.log_prob(observe_x).data.cpu().numpy()

                    log_p_x = log_p_xz * np.sum(
                        torch.exp(self.z_prior_dist.log_prob(observe_z) - q_zx.log_prob(observe_z)).cpu().numpy(),
                        axis=-1, keepdims=True)

                    indicator_erf = erf((torch.abs(observe_x - p_xz.mean) / p_xz.stddev).cpu().numpy() * 0.1589967)

                    loop.submit_data("indicator", -np.mean(log_p_xz[:, :, -1], axis=0))
                    loop.submit_data("indicator_prior", -np.mean(log_p_x[:, :, -1], axis=0))
                    loop.submit_data("indicator_erf", np.mean(indicator_erf[:, :, -1], axis=0))

                    loop.submit_data("x_mean", np.mean(
                        p_xz.mean.data.cpu().numpy()[:, :, -1], axis=0))
                    loop.submit_data("x_std", np.mean(
                        p_xz.stddev.data.cpu().numpy()[:, :, -1], axis=0))

            indicator = np.concatenate(loop.get_data_by_name(indicator_name))
            x_mean = np.concatenate(loop.get_data_by_name("x_mean"))
            x_std = np.concatenate(loop.get_data_by_name("x_std"))

            indicator = np.concatenate([np.ones(shape=self.window_size - 1) * np.min(indicator), indicator])
            if return_statistics:
                return indicator, x_mean, x_std
            else:
                return indicator

    def detect(self, kpi: KPISeries, train_kpi: KPISeries = None, return_threshold=False):
        indicators = self.predict(kpi)
        indicators_ignore_missing, *_ = ignore_missing(indicators, missing=kpi.missing)
        labels_ignore_missing, *_ = ignore_missing(kpi.label, missing=kpi.missing)
        threshold = threshold_ml(indicators_ignore_missing, labels_ignore_missing)

        predict = indicators >= threshold

        if return_threshold:
            return predict, threshold
        else:
            return predict


class Donut:
    def __init__(self, max_epoch: int = 150, batch_size: int = 128, network_size: List[int] = None,
                 latent_dims: int = 8, window_size: int = 120, cuda: bool = True, print_fn=print):
        if network_size is None:
            network_size = [100, 100]

        self.print_fn = print_fn
        self.window_size = window_size
        self.latent_dims = latent_dims
        self.network_size = network_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.cuda = cuda

        self._model = BasicVAE(
            MultiLinearGaussianStatistic(
                self.window_size, self.latent_dims, self.network_size, eps=1e-4),
            MultiLinearGaussianStatistic(
                self.latent_dims, self.window_size, self.network_size, eps=1e-4),
        )
        if self.cuda:
            self._model = self._model.cuda()

        if cuda:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32)).cuda()),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)).cuda())
            )
        else:
            self.z_prior_dist = dist.Normal(
                Variable(torch.from_numpy(np.zeros((self.latent_dims,), np.float32))),
                Variable(torch.from_numpy(np.ones((self.latent_dims,), np.float32)))
            )

    def fit(self, kpi: KPISeries, valid_kpi: KPISeries = None):
        self._model.train()
        with Loop(max_epochs=self.max_epoch, use_cuda=self.cuda, disp_epoch_freq=5,
                  print_fn=self.print_fn).with_context() as loop:
            optimizer = Adam(self._model.parameters(), lr=1e-3)
            lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
            train_kpiframe_dataset = KpiFrameDataset(kpi,
                                                     frame_size=self.window_size, missing_injection_rate=0.01)
            train_dataloader = KpiFrameDataLoader(train_kpiframe_dataset, batch_size=self.batch_size, shuffle=True,
                                                  drop_last=True)
            if valid_kpi is not None:
                valid_kpiframe_dataset = KpiFrameDataset(valid_kpi,
                                                         frame_size=self.window_size, missing_injection_rate=0.)
                valid_dataloader = KpiFrameDataLoader(valid_kpiframe_dataset, batch_size=256, shuffle=True)
            else:
                valid_dataloader = None

            for epoch in loop.iter_epochs():
                lr_scheduler.step()
                for _, batch_data in loop.iter_steps(train_dataloader):
                    optimizer.zero_grad()
                    observe_x, observe_normal = batch_data

                    p_xz, q_zx, observe_z = self._model(observe_x=observe_x)
                    loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                  self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                    loss.backward()
                    clip_grad_norm_(self._model.parameters(), max_norm=10.)
                    optimizer.step()
                    loop.submit_metric("train_loss", loss.data)
                if valid_kpi is not None:
                    with torch.no_grad():
                        for _, batch_data in loop.iter_steps(valid_dataloader):
                            observe_x, observe_normal = batch_data  # type: Variable, Variable
                            p_xz, q_zx, observe_z = self._model(observe_x=observe_x)
                            loss = m_elbo(observe_x, observe_z, observe_normal, p_xz, q_zx,
                                          self.z_prior_dist) + self._model.penalty() * 0.001  # type: Variable
                            loop.submit_metric("valid_loss", loss.data)
            # train_loss_epochs, train_loss = loop.get_metric_by_name("train_loss")
            # valid_loss_epochs, valid_loss = loop.get_metric_by_name("valid_loss")

    def predict(self, kpi: KPISeries, return_statistics=False, indicator_name="indicator"):
        """
        :param kpi:
        :param return_statistics:
        :param indicator_name:
            default "indicator": Reconstructed probability
            "indicator_prior": E_q(z|x)[log p(x|z) * p(z) / q(z|x)]
            "indicator_erf": erf(abs(x - x_mean) / x_std * scale_factor)
        :return:
        """
        with torch.no_grad():
            with TestLoop(use_cuda=True, print_fn=self.print_fn).with_context() as loop:
                test_kpiframe_dataset = KpiFrameDataset(kpi, frame_size=self.window_size, missing_injection_rate=0.0)
                test_dataloader = KpiFrameDataLoader(test_kpiframe_dataset, batch_size=32, shuffle=False,
                                                     drop_last=False)
                self._model.eval()
                for _, batch_data in loop.iter_steps(test_dataloader):
                    observe_x, observe_normal = batch_data  # type: Variable, Variable
                    observe_x = mcmc_missing_imputation(observe_normal=observe_normal,
                                                        vae=self._model,
                                                        n_iteration=10,
                                                        observe_x=observe_x,
                                                        )
                    p_xz, q_zx, observe_z = self._model(observe_x=observe_x,
                                                        n_sample=128,
                                                        )
                    loss = m_elbo(observe_x, observe_z, observe_normal,
                                  p_xz, q_zx, self.z_prior_dist)  # type: Variable
                    loop.submit_metric("test_loss", loss.data.cpu())

                    log_p_xz = p_xz.log_prob(observe_x).data.cpu().numpy()

                    log_p_x = log_p_xz * np.sum(
                        torch.exp(self.z_prior_dist.log_prob(observe_z) - q_zx.log_prob(observe_z)).cpu().numpy(),
                        axis=-1, keepdims=True)

                    indicator_erf = erf((torch.abs(observe_x - p_xz.mean) / p_xz.stddev).cpu().numpy() * 0.1589967)

                    loop.submit_data("indicator", -np.mean(log_p_xz[:, :, -1], axis=0))
                    loop.submit_data("indicator_prior", -np.mean(log_p_x[:, :, -1], axis=0))
                    loop.submit_data("indicator_erf", np.mean(indicator_erf[:, :, -1], axis=0))

                    loop.submit_data("x_mean", np.mean(
                        p_xz.mean.data.cpu().numpy()[:, :, -1], axis=0))
                    loop.submit_data("x_std", np.mean(
                        p_xz.stddev.data.cpu().numpy()[:, :, -1], axis=0))

            indicator = np.concatenate(loop.get_data_by_name(indicator_name))
            x_mean = np.concatenate(loop.get_data_by_name("x_mean"))
            x_std = np.concatenate(loop.get_data_by_name("x_std"))

            indicator = np.concatenate([np.ones(shape=self.window_size - 1) * np.min(indicator), indicator])
            if return_statistics:
                return indicator, x_mean, x_std
            else:
                return indicator

    def detect(self, kpi: KPISeries, train_kpi: KPISeries = None, return_threshold=False):
        indicators = self.predict(kpi)
        indicators_ignore_missing, *_ = ignore_missing(indicators, missing=kpi.missing)
        labels_ignore_missing, *_ = ignore_missing(kpi.label, missing=kpi.missing)
        threshold = threshold_ml(indicators_ignore_missing, labels_ignore_missing)

        predict = indicators >= threshold

        if return_threshold:
            return predict, threshold
        else:
            return predict
