import os
import random
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable


class AutonomousForecastModelArchitecture:
    """
    Deterministic convolutional neural network (CNN) architecture for
    multi-horizon time-series forecasting.

    This class implements a feed-forward temporal convolutional model using
    1D convolutional layers to learn predictive representations from fixed-length
    historical windows. The model maps a sequence of past observations to a
    vector of future targets, enabling direct multi-step forecasting.

    The architecture is designed for structured multivariate time-series data,
    where each training sample consists of a fixed lookback window of historical
    features and a corresponding future forecast horizon.

    Key design characteristics
    --------------------------
    - Temporal feature extraction using Conv1D layers
    - GELU nonlinearities for smooth gradient propagation
    - Layer normalization for training stability
    - Dense projection head for multi-step forecasting
    - Direct sequence-to-vector prediction (no autoregressive recursion)
    - Linear output layer suitable for regression objectives

    Forecasting formulation
    -----------------------
    For a reference timestamp T:

        Input  (X): observations from [T - lookback + 1, ..., T]
        Output (y): targets from      [T + 1, ..., T + horizon]

    This corresponds to supervised sequence-to-multi-horizon regression.

    Reproducibility considerations
    ------------------------------
    - Random seeds are set for Python, NumPy, and TensorFlow
    - Deterministic TensorFlow operations are requested where supported
    - CUDA/GPU nondeterminism may still introduce minor numerical variation
    - Full bitwise determinism is not guaranteed due to floating-point and
    optimizer-level behavior

    Parameters
    ----------
    seed : int
        Random seed used to initialize Python, NumPy, and TensorFlow randomness.

    input_horizon_space : int
        Number of historical timesteps provided as model input
        (lookback window length).

    input_feature_space : int
        Number of features observed at each timestep.

    output_horizon_space : int
        Number of future timesteps predicted by the model
        (forecast horizon).

    Attributes
    ----------
    model : tf.keras.Model
        Compiled TensorFlow model mapping historical sequences to
        multi-step forecasts.

    Notes
    -----
    This implementation is intended for multivariate tabular time-series data.
    It is not designed for image, NLP, or autoencoder-based representation
    learning tasks.

    The model performs direct multi-horizon forecasting, meaning all future
    steps are predicted simultaneously rather than recursively, reducing
    error accumulation during inference."""

    def __init__(
        self,
        seed: int,
        input_horizon_space: int,
        input_feature_space: int,
        output_horizon_space: int,
    ):
        self._set_seed(seed)
        self._disable_cuda()

        self.seed = seed
        self.input_horizon_space = input_horizon_space
        self.input_feature_space = input_feature_space
        self.output_horizon_space = output_horizon_space

        self.model = self._build_model()
        self._compile_model()

    def _set_seed(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Enforce deterministic ops where possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

    def _disable_cuda(self):
        # Disable all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    @register_keras_serializable(package="Autonomous")
    def _directional_penalty_loss(y_true, y_pred, sample_weight=None):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        directional_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32)
        )
        directional_penalty = 2 / (1 + directional_accuracy)
        return mse * directional_penalty

    def _build_model(self) -> tf.keras.models.Model:
        inp = tf.keras.layers.Input(
            shape=(
                self.input_horizon_space,
                self.input_feature_space,
            ),
            name="input_space",
        )

        x = tf.keras.layers.Conv1D(
            filters=12, kernel_size=5, activation="gelu", use_bias=True
        )(inp)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Flatten()(x)

        out = tf.keras.layers.Dense(
            self.output_horizon_space,
            activation="tanh",
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            activity_regularizer=tf.keras.regularizers.L1(0.0001),
            name="final_forecasting_space",
        )(x)

        return tf.keras.models.Model(inputs=inp, outputs=out, name="forecast_model")

    def _compile_model(self):
        self.model.compile(
            optimizer="adam",
            loss=AutonomousForecastModelArchitecture._directional_penalty_loss,
            metrics=["mse"],
        )

    def summary(self):
        return self.model.summary()

    def fit(self, x, y, **kwargs):
        return self.model.fit(
            x,
            y,
            callbacks=tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            **kwargs,
        )

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "AutonomousForecastModelArchitecture":
        model = tf.keras.models.load_model(path)

        obj = cls(
            seed=0,
            input_horizon_space=model.input_shape[1],
            input_feature_space=model.input_shape[2],
            output_horizon_space=model.output_shape[1],
        )

        obj.model = model
        return obj
