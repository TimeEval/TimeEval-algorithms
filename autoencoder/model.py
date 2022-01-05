import shutil, os
import tensorflow as tf
from tensorflow import keras
from keras import Input
from keras.layers import Dense

class AutoEn():

    def __init__(self, latent_size = 32, epochs = 10, learning_rate = 0.005, early_stopping_patience: int = 10, early_stopping_delta: float = 1e-2, split: float = 0.8):
        super(AutoEn, self).__init__()
        self.encoder_dim = latent_size
        self.epochs = epochs
        self.lr = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.validation_split = 1 - split

    def get_models(self):
        self.inp = Input(shape= (self.features,))
        self.fc = Dense(self.encoder_dim)(self.inp)
        self.d1 = Dense(self.features)(self.fc)
        self.autoencoder = tf.keras.Model(inputs = self.inp, outputs = self.d1)

    def fit(self, xtr, model_path):
        self.features = xtr.shape[1]
        opt = keras.optimizers.Adam(learning_rate=self.lr)
        self.get_models()
        self.autoencoder.compile(optimizer = opt, loss = 'mse')
        self.autoencoder.fit(xtr, xtr, epochs = self.epochs, validation_split=self.validation_split,
                             callbacks=[
                                 tf.keras.callbacks.EarlyStopping(patience=self.early_stopping_patience, min_delta=self.early_stopping_delta),
                                 tf.keras.callbacks.ModelCheckpoint("check", save_best_only=True),
                                 tf.keras.callbacks.LambdaCallback(
                                     on_epoch_end=lambda x, y: AutoEn._create_archive("check", model_path) if os.path.exists("check") else None
                                 )
                             ])

    def save(self, model_path):
        self.autoencoder.save("tmp_model")
        AutoEn._create_archive("tmp_model", model_path)

    @staticmethod
    def _create_archive(tmp_path, model_path):
        shutil.make_archive(model_path, root_dir=tmp_path, format="zip")
        shutil.move(f"{model_path}.zip", model_path)

    @staticmethod
    def load(model_path):
        shutil.unpack_archive(model_path, extract_dir="tmp_model", format="zip")
        model = keras.models.load_model("tmp_model")
        return model
