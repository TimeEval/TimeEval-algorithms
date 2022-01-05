import logging
import numpy as np
from torsk.data import detrend
from scipy.fftpack import dctn, idctn

logger = logging.getLogger(__name__)


def split_train_label_pred(sequence, train_length, pred_length):
    train_end = train_length + 1
    train_seq = sequence[:train_end]
    inputs = train_seq[:-1]
    labels = train_seq[1:]
    pred_labels = sequence[train_end:train_end + pred_length]
    return inputs, labels, pred_labels


class NumpyImageDataset:
    """Dataset that contains the raw images and does nothing but providing
    convenient access to inputs/labels/pred_labels
    """
    def __init__(self, images, params, scale_images=True):
        self.params = params
        self.train_length = params.train_length
        self.pred_length = params.pred_length
        # -1 (for input/label shift) and +1 (for index to size conversion) cancel out each other:
        self.nr_sequences = images.shape[0] - self.train_length - self.pred_length   # - 1 + 1
        logger.debug(f"#batches = {self.nr_sequences}")
        self.max = None
        self.min = None

        self.dtype = np.dtype(params.dtype)

        if "cycle_length" in params.dict:
            cycle_length    = params.cycle_length
            cycle_timescale = params.dict.get('cycle_timescale',1)

            logger.info(f"Detrending and removing average length-{cycle_length} cycle")

            Ftkk = dctn(images,norm='ortho',axes=[1,2])

            if cycle_timescale == 1:
                (ftkk,bkk,Ckk) = detrend.separate_trends_unscaled(Ftkk,cycle_length)
            else:
                nT = int((cycle_timescale*Ftkk.shape[0]).round())
                (ftkk,bkk,Ckk) = detrend.separate_trends_scaled(Ftkk,nT,cycle_length)

            ftxx = idctn(ftkk,norm='ortho',axes=[1,2])

            self.quadratic_trend = bkk
            self.mean_cycle      = Ckk
            self.cycle_timescale = cycle_timescale
            self.cycle_lengtth   = cycle_length

            self.detrend_training_data = params.dict.get('detrend_training_data',False)

            if self.detrend_training_data:
                images = ftxx

        if scale_images:
            logger.debug("Scaling input images to (-1, 1)")
            images = self.scale(images)
        self._images = images.astype(self.dtype)
        self.image_shape = images.shape[1:]

    def scale(self, images):
        self.min = images.min()
        self.max = images.max()
        normalized = (images - self.min) / (self.max - self.min)
        scaled = normalized * 2 - 1
        return scaled

    def unscale(self, images):
        if self.max is None or self.min is None:
            raise ValueError("Min/max not set. Call 'scale' first.")
        normalized = (images + 1) * 0.5
        orig = normalized * (self.max - self.min) + self.min
        return orig

    def __getitem__(self, index):
        if (index < 0) or (index >= self.nr_sequences):
            raise IndexError('ImageDataset index out of range.')

        images = self._images[
            index:index + self.train_length + self.pred_length + 1]

        inputs, labels, pred_labels = split_train_label_pred(
            images, self.train_length, self.pred_length)

        logger.debug(f"Generated batch shapes for idx={index}: inputs={inputs.shape}, labels={labels.shape}, pred_labels={pred_labels.shape}")
        return inputs, labels, pred_labels

    def __len__(self):
        return self.nr_sequences
