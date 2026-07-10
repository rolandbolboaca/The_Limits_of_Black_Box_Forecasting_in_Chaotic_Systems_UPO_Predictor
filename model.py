import time

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
)
from sklearn.decomposition import PCA, KernelPCA

from scipy.linalg import hankel
from hankel import HankelTransform

from metrics import Metrics
from data import DataHandler

# tf.compat.v1.enable_eager_execution()
# tf.config.run_functions_eagerly(True)

class SequenceModel:
    def __init__(self, model_parameters=None, logging = False):
        default_params = {
            'model_type': 'LSTM',   # 'RNN', 'GRU'
            'seq_len': 256,
            'hidden_size': [32, 16],
            'LR': 0.01,
            'decay_rate': 0.96,
            'decay_step': 40,
            'batch_size': 16,
            'epochs': 100,
            'train_ratio': 1.0,     # 70% train / 30% test
            'optimizer': 'adam',
            'loss': 'mse',
            'metrics': ['mse', 'mae'],
            
            'seed': 42,
            'TF': False,             # teacher forcing
            'stateful': True,
            'MISO': False,
            'forecast_size': 1,
            'rolling': False,
            'correction': 0

        }

        self.params = {**default_params, **(model_parameters or {})}
        self.scaler = None

        # self.model = self.create_model(self)
        self.metrics = Metrics()

        self.differential_feedback = False
        self.second_differential = False
        self.delay_embedding = False
        self.dimensionality_reduction = False
        self.test = False

        self.embedding_delay = 0

        self.logging = logging
        self.fixed_sl = False
        self.multi_scaler = None
    # ---------------------------------------------------------------------

    def create_model(self, n_inputs=1, n_outputs=1, dataset_parameters=None, test = False, dropout = 0, HED_alpha = 0.5, delay_embedding = False, embedding_delay = 0, att = False):

        def MED(y_true, y_pred):
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)))
        
        def HED_loss(alpha=0.5):

            def loss(y_true, y_pred):
                return HED(y_true, y_pred, alpha=alpha)
            return loss

        def HED(y_true, y_pred, alpha=0.5):
            mse = tf.reduce_mean(tf.square(y_pred - y_true))
            euclid = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)))
            return alpha * mse + (1-alpha) * euclid


        self.delay_embedding = delay_embedding
        self.embedding_delay = embedding_delay

        if dataset_parameters is not None:
            n_inputs = len(dataset_parameters['inputs'])
            n_outputs = len(dataset_parameters['outputs'])

        if self.params['TF']:
            n_inputs += len(dataset_parameters['outputs'])


        original_n_inputs = n_inputs
        if delay_embedding and embedding_delay > 0:
            n_inputs += embedding_delay  * original_n_inputs  # add 5 time-delayed embeddings

        keras.utils.set_random_seed(self.params['seed'])
        model = keras.Sequential()

        self.early_stopping = EarlyStopping(
            # monitor='val_loss',     
            patience=50,             
            restore_best_weights=True 
        )
        
        # input
        if test:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(1,
                                    self.params['seq_len'],
                                    n_inputs)
                )
            )
        else:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(self.params['batch_size'],
                                    self.params['seq_len'],
                                    n_inputs)
                )
            )

        # model.add(keras.layers.GaussianNoise(stddev=0.1))
        # recurrent layers
        for i in range(len(self.params['hidden_size'])):

            # return_sequences = (i < len(self.params['hidden_size']) - 1)
            layer_args = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'return_state': False,
                'stateful': self.params['stateful'],
            }

            if self.params.get('dropout', 0) > 0:
                layer_args['dropout'] = self.params['dropout']

            if self.params.get('recurrent_dropout', 0) > 0:
                layer_args['recurrent_dropout'] = self.params['recurrent_dropout']

            l2_value = self.params.get("l2_reg", 0)

            if l2_value and l2_value > 0:
                reg = regularizers.l2(l2_value)
                layer_args['kernel_regularizer'] = reg
                layer_args['recurrent_regularizer'] = reg

            # layer_args['return_state'] = True if i == len(self.params['hidden_size']) - 1 else False

            # if i > 0:
            #     model.add(keras.layers.Dense(self.params['hidden_size'][i], activation='relu'))
            # else:
            #     if self.params['model_type'] == 'RNN':
            #         model.add(keras.layers.SimpleRNN(**layer_args))
            #     elif self.params['model_type'] == 'LSTM':
            #         model.add(keras.layers.LSTM(**layer_args))
            #     elif self.params['model_type'] == 'GRU':
            #         model.add(keras.layers.GRU(**layer_args))

            if self.params['model_type'] == 'RNN':
                model.add(keras.layers.SimpleRNN(**layer_args))

            elif self.params['model_type'] == 'LSTM':
                model.add(keras.layers.LSTM(**layer_args))
            elif self.params['model_type'] == 'GRU':
                model.add(keras.layers.GRU(**layer_args))

        # output
        model.add(keras.layers.Dense(n_outputs))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.params['LR'],
            decay_steps=self.params['decay_step'],
            decay_rate=self.params['decay_rate']) 
        


        # compile
        if self.params['loss'] == "LDA":
            model.compile(
                    optimizer=self.params['optimizer'],
                    loss=self.polar_loss_factory(),
                    metrics=self.params['metrics'],
                    run_eagerly=False
                )
        elif self.params['loss'] == "var_loss":
            model.compile(
                    optimizer=self.params['optimizer'],
                    loss=self.var_loss,
                    metrics=self.params['metrics'],
                    run_eagerly=False
                )
        elif self.params['loss'] == "bsc_loss":
            model.compile(
                    optimizer=self.params['optimizer'],
                    loss=BSP1DLoss(num_bins=20),
                    metrics=self.params['metrics'],
                    run_eagerly=False
                )
        elif self.params['loss'] == "MED":
            model.compile(
                    optimizer=self.params['optimizer'],
                    loss=MED,
                    metrics=self.params['metrics'],
                    run_eagerly=False
                )
        elif self.params['loss'] == "HED":
            model.compile(
                    optimizer=self.params['optimizer'],
                    loss=HED_loss(alpha=HED_alpha),
                    metrics=self.params['metrics'],
                    run_eagerly=False
                )
        else:
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.params['loss'],
                metrics=self.params['metrics'],

                run_eagerly=False
            )
        
        model.optimizer.lr=lr_schedule
        model.optimizer.learning_rate=self.params['LR']
        model.optimizer.global_clipnorm = self.params.get('clipnorm', None)
       

        # Create models. When testing, only create self.model_test and transfer parameters and states
        if test:
            self.model_test = model
            self.model_test.set_weights(self.model.get_weights())

            if self.logging:
                self.model_test.summary()
        else:
            self.model = model
            if self.logging:
                self.model.summary()

        return model
    
    def create_model_attention(self,
                            n_inputs=1,
                            n_outputs=1,
                            dataset_parameters=None,
                            test=False,
                            dropout=0,
                            HED_alpha=0.5,
                            delay_embedding=False,
                            embedding_delay=0,
                            num_heads=4,
                            key_dim=16):

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import EarlyStopping

        def MED(y_true, y_pred):
            return tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
            )

        def HED_loss(alpha=0.5):
            def loss(y_true, y_pred):
                return HED(y_true, y_pred, alpha=alpha)
            return loss

        def HED(y_true, y_pred, alpha=0.5):
            mse = tf.reduce_mean(tf.square(y_pred - y_true))
            euclid = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
            )
            return alpha * mse + (1 - alpha) * euclid

        self.delay_embedding = delay_embedding
        self.embedding_delay = embedding_delay

        if dataset_parameters is not None:
            n_inputs = len(dataset_parameters['inputs'])
            n_outputs = len(dataset_parameters['outputs'])

        if self.params['TF']:
            n_inputs += len(dataset_parameters['outputs'])

        original_n_inputs = n_inputs
        if delay_embedding and embedding_delay > 0:
            n_inputs += embedding_delay * original_n_inputs

        keras.utils.set_random_seed(self.params['seed'])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # ----- INPUT -----
        if test:
            inputs = keras.Input(
                batch_shape=(1, self.params['seq_len'], n_inputs)
            )
        else:
            inputs = keras.Input(
                batch_shape=(self.params['batch_size'],
                            self.params['seq_len'],
                            n_inputs)
            )

        x = inputs

        # ----- STACKED RECURRENT LAYERS -----
        for i in range(len(self.params['hidden_size'])):

            layer_args = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'stateful': self.params['stateful'],
                'dropout': dropout,
                'recurrent_dropout': dropout / 2
            }

            if self.params['model_type'] == 'RNN':
                x = layers.SimpleRNN(**layer_args)(x)

            elif self.params['model_type'] == 'LSTM':
                x = layers.LSTM(**layer_args)(x)

            elif self.params['model_type'] == 'GRU':
                x = layers.GRU(**layer_args)(x)

        # ----- SINGLE MULTIHEAD ATTENTION AT THE TOP -----
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )(x, x)

        # Residual connection + normalization (recommended)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)

        # ----- OUTPUT -----
        outputs = layers.Dense(n_outputs)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.params['LR'],
            decay_steps=self.params['decay_step'],
            decay_rate=self.params['decay_rate']
        )

        # ----- COMPILE -----
        if self.params['loss'] == "LDA":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.polar_loss_factory(),
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        elif self.params['loss'] == "var_loss":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.var_loss,
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        elif self.params['loss'] == "bsc_loss":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=BSP1DLoss(num_bins=20),
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        elif self.params['loss'] == "MED":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=MED,
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        elif self.params['loss'] == "HED":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=HED_loss(alpha=HED_alpha),
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        else:
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.params['loss'],
                metrics=self.params['metrics'],
                run_eagerly=False
            )

        model.optimizer.lr = lr_schedule
        model.optimizer.learning_rate = self.params['LR']

        # ----- TRAIN / TEST HANDLING -----
        if test:
            self.model_test = model
            self.model_test.set_weights(self.model.get_weights())

            if self.logging:
                self.model_test.summary()
        else:
            self.model = model
            if self.logging:
                self.model.summary()

        return model
    # ---------------------------------------------------------------------
    def create_model_feedback(self, n_inputs=1, n_outputs=1, 
                              dataset_parameters=None, 
                              test = False, 
                              dropout = 0, 
                              differential_feedback = False,
                              second_differential = False,
                              delay_embedding = False,
                              embedding_delay = 0,
                              dimensionality_reduction = False,
                              reduced_dimensions = 3):


        self.differential_feedback = differential_feedback
        self.second_differential = second_differential
        self.delay_embedding = delay_embedding
        self.embedding_delay = embedding_delay
        self.dimensionality_reduction = dimensionality_reduction
        self.reduced_dimensions = reduced_dimensions
        self.test = test

        original_n_inputs = len(dataset_parameters['inputs'])

        if dataset_parameters is not None:
            if differential_feedback:
                if second_differential:
                    n_inputs = len(dataset_parameters['inputs']) * 3 + 1 # + 2  # additional input(s) for feedback difference
                else:
                    n_inputs = len(dataset_parameters['inputs']) * 2 + 1 # + 2  # additional input(s) for feedback difference
            else:
                n_inputs = len(dataset_parameters['inputs']) + 1 # additional input for error feedback

            n_outputs = len(dataset_parameters['outputs'])

        if self.params['TF']:
            n_inputs += len(dataset_parameters['outputs'])


        if delay_embedding and embedding_delay > 0:
            n_inputs += embedding_delay  * original_n_inputs  # add 5 time-delayed embeddings

        if dimensionality_reduction:
            n_inputs = reduced_dimensions + 1  # reduced dimensions + error feedback
        # n_inputs = 1021
        # reg = regularizers.l1_l2(l1=1e-5, l2=1e-4)
        # reg = regularizers.l2(1e-5)

        keras.utils.set_random_seed(self.params['seed'])

        model = keras.Sequential()
        
        # input
        if test:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(self.params['batch_size'],
                                    self.params['seq_len'],
                                    n_inputs)
                )
            )
        else:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(self.params['batch_size'],
                                    self.params['seq_len'],
                                    n_inputs)
                )
            )

        # model.add(keras.layers.Dense(1))
        # Add CNN 1 D layer to preprocess error feedback
        # model.add(keras.layers.Conv1D(filters=3, kernel_size=5, padding='same'))

        # recurrent layers
        for i in range(len(self.params['hidden_size'])):

            # return_sequences = (i < len(self.params['hidden_size']) - 1)
            layer_args = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'return_state': False,
                'stateful': self.params['stateful'],
                # 'dropout': dropout,
                # 'recurrent_dropout': dropout/2,
                # 'go_backwards': True,
                # 'kernel_regularizer': reg,
                #'recurrent_regularizer': reg,
                # 'bias_regularizer': reg
            }

            if self.params['model_type'] == 'RNN':
                model.add(keras.layers.SimpleRNN(**layer_args))
            elif self.params['model_type'] == 'LSTM':
                model.add(keras.layers.LSTM(**layer_args))
            elif self.params['model_type'] == 'GRU':
                model.add(keras.layers.GRU(**layer_args))

        # output
        # model.add(keras.layers.Dense(1, activation='tanh'))
        model.add(keras.layers.Dense(n_outputs))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.params['LR'],
            decay_steps=self.params['decay_step'],
            decay_rate=self.params['decay_rate']) 
        
        # compile
        model.compile(
            optimizer=self.params['optimizer'],
            loss=self.params['loss'],
            metrics=self.params['metrics'],
            run_eagerly=False

        )
        
        model.optimizer.lr=lr_schedule
        model.optimizer.learning_rate=self.params['LR']
        # model.optimizer.global_clipnorm = 0.5
        # model.optimizer.clipvalue=0.2

        # Create models. When testing, only create self.model_test and transfer parameters and states
        if test:
            self.model_test = model
            self.model_test.set_weights(self.model.get_weights())

            if self.logging:
                self.model_test.summary()
        else:
            self.model = model
            if self.logging:
                self.model.summary()

        return model
    # ---------------------------------------------------------------------
    def create_model_API(self, n_inputs=1, n_outputs=1, dataset_parameters=None, test = False, dropout = 0):

        if dataset_parameters is not None:
            n_inputs = len(dataset_parameters['inputs'])
            n_outputs = len(dataset_parameters['outputs'])

        if self.params['TF']:
            n_inputs += len(dataset_parameters['outputs'])

        keras.utils.set_random_seed(self.params['seed'])

        # Input definition
        if test:
            inp = Input(
                batch_shape=(1, self.params['seq_len'], n_inputs),
                name="input"
            )
        else:
            inp = Input(
                batch_shape=(self.params['batch_size'],
                            self.params['seq_len'],
                            n_inputs),
                name="input"
            )


        x = inp

        # recurrent layers
        h = []
        c = []
        
        for i in range(len(self.params['hidden_size'])):


            layer_args = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'stateful': self.params['stateful'],
                'return_state': False
            }
            if test:
                layer_args['return_state'] = True

            if i == 0:
                hidden_input = inp
            else:
                hidden_input = x

            if test:
                if self.params['model_type'] == 'RNN':
                    x, ht = keras.layers.SimpleRNN(**layer_args)(hidden_input)
                    h.append(ht)
                elif self.params['model_type'] == 'LSTM':
                    x, ht, ct = keras.layers.LSTM(**layer_args)(hidden_input)
                    h.append(ht)
                    c.append(ct)
                elif self.params['model_type'] == 'GRU':
                    x, ht = keras.layers.GRU(**layer_args)(hidden_input)
                    h.append(ht)

            else:
                if self.params['model_type'] == 'RNN':
                    x = keras.layers.SimpleRNN(**layer_args)(hidden_input)
                elif self.params['model_type'] == 'LSTM':
                    x = keras.layers.LSTM(**layer_args)(hidden_input)
                elif self.params['model_type'] == 'GRU':
                    x = keras.layers.GRU(**layer_args)(hidden_input)


        # out = custom_layer(units=n_outputs)([x, inp, external_input])

        out = keras.layers.Dense(units=n_outputs)(x)

        #model = Model(inputs=inp, outputs=out)
        if test:
            model = Model(inputs=inp, outputs=[out, h, c])
        else:
            model = Model(inputs=inp, outputs=[out])


        # out = keras.layers.Activation("leaky_relu")(raw_out)         
          
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.params['LR'],
            decay_steps=self.params['decay_step'],
            decay_rate=self.params['decay_rate']) 

        # compile
        if self.params['loss'] == "var_loss":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.var_loss,
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        elif self.params['loss'] == "bsc_loss":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=BSP1DLoss(num_bins=20),
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        else:
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.params['loss'],
                metrics=self.params['metrics'],
                run_eagerly=False
            )

        model.optimizer.lr=lr_schedule

        # Create models. When testing, only create self.model_test and transfer parameters and states
        if test:
            self.model_test = model
            self.model_test.set_weights(self.model.get_weights())

            if self.logging:
                self.model_test.summary()
        else:
            self.model = model
            if self.logging:
                self.model.summary()

        return model
    # ---------------------------------------------------------------------
    def create_model_API_jump(self, n_inputs=1, n_outputs=1, dataset_parameters=None, test=False, dropout=0):
        if dataset_parameters is not None:
            n_inputs = len(dataset_parameters['inputs'])
            n_outputs = len(dataset_parameters['outputs'])

        if self.params['TF']:
            n_inputs += len(dataset_parameters['outputs'])

        keras.utils.set_random_seed(self.params['seed'])

        # Input definition
        if test:
            inp = Input(
                batch_shape=(1, self.params['seq_len'], n_inputs),
                name="input"
            )
        else:
            inp = Input(
                batch_shape=(self.params['batch_size'], self.params['seq_len'], n_inputs),
                name="input"
            )

        x = inp

        h1_out = None  # store first hidden output

        # recurrent layers
        for i in range(len(self.params['hidden_size'])):
            layer_args = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'stateful': self.params['stateful'],
                'return_state': False
            }

            if test:
                layer_args['return_state'] = False

            if i == 0:
                hidden_input = inp
            else:
                hidden_input = x

            if self.params['model_type'] == 'RNN':
                x = keras.layers.SimpleRNN(**layer_args)(hidden_input)
            elif self.params['model_type'] == 'LSTM':
                x = keras.layers.LSTM(**layer_args)(hidden_input)
            elif self.params['model_type'] == 'GRU':
                x = keras.layers.GRU(**layer_args)(hidden_input)

            if i == 0:
                h1_out = x  # capture output of first hidden layer

        # concatenate jump connection: h1 + last hidden layer
        if h1_out is not None:
            x = keras.layers.Concatenate(axis=-1)([x, h1_out])

        # Output layer
        out = keras.layers.Dense(units=n_outputs)(x)

        # Create model
        if test:
            model = Model(inputs=inp, outputs=[out])
        else:
            model = Model(inputs=inp, outputs=[out])

        # Learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.params['LR'],
            decay_steps=self.params['decay_step'],
            decay_rate=self.params['decay_rate']
        )

        # Compile with appropriate loss
        if self.params['loss'] == "var_loss":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.var_loss,
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        elif self.params['loss'] == "bsc_loss":
            model.compile(
                optimizer=self.params['optimizer'],
                loss=BSP1DLoss(num_bins=20),
                metrics=self.params['metrics'],
                run_eagerly=False
            )
        else:
            model.compile(
                optimizer=self.params['optimizer'],
                loss=self.params['loss'],
                metrics=self.params['metrics'],
                run_eagerly=False
            )

        model.optimizer.lr = lr_schedule

        # Create train/test version
        if test:
            self.model_test = model
            self.model_test.set_weights(self.model.get_weights())
            if self.logging:
                self.model_test.summary()
        else:
            self.model = model
            if self.logging:
                self.model.summary()

        return model
    # ---------------------------------------------------------------------
    def create_model_classification(self, n_inputs=1, n_outputs=1, dataset_parameters=None, test = False, dropout = 0):
            
        if dataset_parameters is not None:
            n_inputs = len(dataset_parameters['inputs'])
            n_outputs = len(dataset_parameters['outputs'])

        if self.params['TF']:
            n_inputs += len(dataset_parameters['outputs'])

        keras.utils.set_random_seed(self.params['seed'])
        model = keras.Sequential()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',     
            patience=10,             
            restore_best_weights=True 
        )
        
        # input
        if test:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(1,
                                    self.params['seq_len'],
                                    n_inputs)
                )
            )
        else:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(self.params['batch_size'],
                                    self.params['seq_len'],
                                    n_inputs)
                )
            )

        # recurrent layers
        for i in range(len(self.params['hidden_size'])):

            # return_sequences = (i < len(self.params['hidden_size']) - 1)
            layer_args = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'return_state': False,
                'stateful': self.params['stateful'],
                'dropout': dropout,
                'recurrent_dropout': dropout/2
            }

            # layer_args['return_state'] = True if i == len(self.params['hidden_size']) - 1 else False
            
            # if i > 0:
            #     model.add(keras.layers.Dense(self.params['hidden_size'][i], activation='relu'))
            # else:
            #     if self.params['model_type'] == 'RNN':
            #         model.add(keras.layers.SimpleRNN(**layer_args))
            #     elif self.params['model_type'] == 'LSTM':
            #         model.add(keras.layers.LSTM(**layer_args))
            #     elif self.params['model_type'] == 'GRU':
            #         model.add(keras.layers.GRU(**layer_args))

            if self.params['model_type'] == 'RNN':
                model.add(keras.layers.SimpleRNN(**layer_args))
            elif self.params['model_type'] == 'LSTM':
                model.add(keras.layers.LSTM(**layer_args))
            elif self.params['model_type'] == 'GRU':
                model.add(keras.layers.GRU(**layer_args))

        # output
        model.add(keras.layers.Dense(6, activation='softmax'))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.params['LR'],
            decay_steps=self.params['decay_step'],
            decay_rate=self.params['decay_rate']) 
        


        model.compile(
                optimizer=self.params['optimizer'],
                loss='categorical_crossentropy',
                metrics=self.params['metrics'],
                run_eagerly=False
        )

        model.optimizer.lr=lr_schedule
        model.optimizer.learning_rate=self.params['LR']
       

        # Create models. When testing, only create self.model_test and transfer parameters and states
        if test:
            self.model_test = model
            self.model_test.set_weights(self.model.get_weights())

            if self.logging:
                self.model_test.summary()
        else:
            self.model = model
            if self.logging:
                self.model.summary()

        return model
    # ---------------------------------------------------------------------    
    def create_transfer_model(self, n_inputs=1, n_outputs=1, dataset_parameters=None, test = False):

        if self.model is None:
            raise Exception("No model exists. Please create a model first.")
        
        if dataset_parameters is not None:
            n_inputs = len(dataset_parameters['inputs'])
            n_outputs = len(dataset_parameters['outputs'])

        if self.params['TF']:
            n_inputs += len(dataset_parameters['outputs'])

        keras.utils.set_random_seed(self.params['seed'])
        model = keras.Sequential()
        
        # input
        if test:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(1,
                                    self.params['seq_len'],
                                    n_inputs)
                )
            )
        else:
            model.add(
                keras.layers.InputLayer(
                    batch_input_shape=(self.params['batch_size'],
                                    self.params['seq_len'],
                                    n_inputs), trainable=False
                )
            )

        # recurrent layers
        for i in range(len(self.params['hidden_size'])):

            # return_sequences = (i < len(self.params['hidden_size']) - 1)
            layer_args = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'stateful': self.params['stateful'],
                'trainable': False
            }

            if self.params['model_type'] == 'RNN':
                model.add(keras.layers.SimpleRNN(**layer_args))
            elif self.params['model_type'] == 'LSTM':
                model.add(keras.layers.LSTM(**layer_args))
            elif self.params['model_type'] == 'GRU':
                model.add(keras.layers.GRU(**layer_args))

        layer_args_2 = {
                'units': self.params['hidden_size'][i],
                'return_sequences': True,
                'stateful': self.params['stateful'],
                'trainable': True
            }

        if self.params['model_type'] == 'RNN':
            model.add(keras.layers.SimpleRNN(**layer_args_2))
        elif self.params['model_type'] == 'LSTM':
            model.add(keras.layers.LSTM(**layer_args_2))
        elif self.params['model_type'] == 'GRU':
            model.add(keras.layers.GRU(**layer_args_2))
        
        # output
        model.add(keras.layers.Dense(n_outputs, trainable=True))

        for i in range(len(self.params['hidden_size'])):
            model.layers[i].set_weights(self.model.layers[i].get_weights())

        model.layers[-1].set_weights(self.model.layers[-1].get_weights())

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.params['LR'],
            decay_steps=self.params['decay_step'],
            decay_rate=self.params['decay_rate']) 

        # compile
        model.compile(
            optimizer=self.params['optimizer'],
            loss=self.params['loss'],
            metrics=self.params['metrics'],
            run_eagerly=False
        )

        model.optimizer.lr=lr_schedule

        # Create models. When testing, only create self.model_test and transfer parameters and states
        if test:
            self.model_transfer_test = model
            self.model_transfer_test.set_weights(self.model_transfer.get_weights())

            if self.logging:
                self.model_test.summary()
        else:
            self.model_transfer = model
            if self.logging:
                self.model_transfer.summary()
        
        return model
    # ---------------------------------------------------------------------
    def _prepare_data(self, data, dataset_parameters, normalize=True):
        """
        Normalize data, construct X/y pairs, and apply optional
        teacher forcing and feature augmentations.
        """

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------
        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')

        if data is None:
            raise ValueError('data must be provided')

        if len(dataset_parameters['inputs']) == 0:
            raise ValueError('inputs must be provided')

        if len(dataset_parameters['outputs']) > 1 and self.params['MISO']:
            raise ValueError('MISO supports only single output')

        data = np.asarray(data)

        # --------------------------------------------------
        # NORMALIZATION
        # --------------------------------------------------
        if normalize:
            if self.multi_scaler is None:
                data = self.scaler.transform(data)
            else:
                for rho in np.unique(data[:, 6]):
                    mask = data[:, 6] == rho
                    data[mask] = self.multi_scaler[rho].transform(data[mask])

        # --------------------------------------------------
        # INPUT / OUTPUT SELECTION
        # --------------------------------------------------
        X = data[:, dataset_parameters['inputs']]
        y = data[:, dataset_parameters['outputs']]

        # --------------------------------------------------
        # TEACHER FORCING (feature feedback)
        # --------------------------------------------------
        if self.params['TF']:
            X = np.hstack((X[1:], y[:-1]))
            y = y[1:]

        # --------------------------------------------------
        # TEMPORAL SHIFTING
        # --------------------------------------------------
        if not self.params['MISO']:
            X = X[:-1]
            y = y[1:]

        fs = self.params['forecast_size']

        if fs > 1:
            X = X[:-(fs + 1)]
            y = y[fs - 1:]
        else:
            # disable rolling for one-step forecast
            self.params['rolling'] = False
            X = X[:-1]
            y = y[1:]

        # --------------------------------------------------
        # ROLLING WINDOW CONSTRUCTION
        # --------------------------------------------------
        if self.params['rolling']:
            new_X, new_y = [], []

            seq_len = self.params['seq_len']
            for i in range(0, X.shape[0] - seq_len + 1, fs):
                new_X.extend(X[i:i + seq_len])
                new_y.extend(y[i:i + seq_len])

            X = np.asarray(new_X)
            y = np.asarray(new_y)

        # --------------------------------------------------
        # FEATURE DIMENSIONS
        # --------------------------------------------------
        num_feats = (
            len(dataset_parameters['inputs']) +
            (len(dataset_parameters['outputs']) if self.params['TF'] else 0)
        )
        num_outputs = len(dataset_parameters['outputs'])

        base_feat_dim = num_feats

        # --------------------------------------------------
        # DIFFERENTIAL FEEDBACK (1st derivative)
        # --------------------------------------------------
        if self.differential_feedback:
            num_feats += base_feat_dim

            diff = X[1:] - X[:-1]
            diff = np.vstack((np.zeros((1, X.shape[1])), diff))

            X = np.hstack((X, diff))

        # --------------------------------------------------
        # SECOND DIFFERENTIAL FEEDBACK
        # --------------------------------------------------
        if self.second_differential:
            num_feats += base_feat_dim

            first_diff = X[1:, 0:3] - X[:-1, 0:3]
            second_diff = first_diff[1:] - first_diff[:-1]
            second_diff = np.vstack(
                (np.zeros((2, second_diff.shape[1])), second_diff)
            )

            X = np.hstack((X, second_diff))

        # --------------------------------------------------
        # DELAY EMBEDDING
        # --------------------------------------------------
        if self.delay_embedding:
            tau_max = self.embedding_delay

            X_base = X[:-tau_max]
            X_delay = X[:-tau_max, 0:base_feat_dim]

            for tau in range(1, tau_max + 1):
                num_feats += base_feat_dim
                delayed = X[tau_max - tau:-tau, 0:base_feat_dim]
                X_delay = np.hstack((X_delay, delayed))

            X_delay = X_delay[:, base_feat_dim:]
            X = np.hstack((X_base, X_delay))

            if self.test:
                y = y[tau_max:]
                print(f"Test mode: Adjusted y for delay embedding {tau_max}")

        # --------------------------------------------------
        # DIMENSIONALITY REDUCTION
        # --------------------------------------------------
        if self.dimensionality_reduction:
            pca = PCA(n_components=self.reduced_dimensions)

            X = pca.fit_transform(X)
            y = pca.fit_transform(y)

            num_feats = self.reduced_dimensions + 1

        # --------------------------------------------------
        # STORE META INFORMATION
        # --------------------------------------------------
        self.num_feats = num_feats
        self.num_outputs = num_outputs

        # --------------------------------------------------
        # SEQUENCE RESHAPING
        # --------------------------------------------------
        return self._reshape_sequences(X, y, num_feats, num_outputs)
    # ---------------------------------------------------------------------

    def _prepare_data_tf(self, data, dataset_parameters, normalize=True):
        """
        Teacher Forcing data preparation.
        Trains model for ONE-STEP prediction:
            [x(t-seq_len+1) ... x(t)] -> x(t+1)

        Ground-truth previous values are always used during training.
        """

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')

        if data is None:
            raise ValueError('data must be provided')

        if len(dataset_parameters['inputs']) == 0:
            raise ValueError('inputs must be provided')

        data = np.array(data)

        # --------------------------------------------------
        # NORMALIZATION (identical behaviour)
        # --------------------------------------------------
        if self.multi_scaler is None:
            if normalize:
                data = self.scaler.transform(data)
        else:
            if normalize:
                for rho in np.unique(data[:, 6]):
                    mask = data[:, 6] == rho
                    data[mask, :] = self.multi_scaler[rho].transform(data[mask, :])

        # --------------------------------------------------
        # SELECT INPUTS / OUTPUTS
        # --------------------------------------------------
        X_raw = data[:, dataset_parameters['inputs']]
        y_raw = data[:, dataset_parameters['outputs']]

        # --------------------------------------------------
        # TEACHER FORCING SHIFT (ONE STEP)
        # --------------------------------------------------
        # input at t predicts output at t+1
        X_raw = X_raw[:-1]
        y_raw = y_raw[1:]

        # optional: append previous true output as input
        # (classic teacher forcing feedback)
        if self.params.get("TF", True):
            prev_y = y_raw[:-1]
            X_raw = np.hstack((X_raw[1:], prev_y))
            y_raw = y_raw[1:]

        X = X_raw
        y = y_raw

        # --------------------------------------------------
        # FEATURE ENGINEERING (REUSED EXACTLY)
        # --------------------------------------------------
        num_feats = X.shape[1]
        num_outputs = y.shape[1]

        temp_feats = num_feats
        X_original = X.copy()

        # ---------- differential feedback ----------
        if self.differential_feedback:
            num_feats += temp_feats
            diff_vectors = X[1:] - X[:-1]
            diff_vectors = np.vstack((np.zeros((1, X.shape[1])), diff_vectors))
            X = np.hstack((X, diff_vectors))

        # ---------- second differential ----------
        if self.second_differential:
            num_feats += temp_feats
            first_diff = X[1:, 0:3] - X[:-1, 0:3]
            second_diff = first_diff[1:] - first_diff[:-1]
            second_diff = np.vstack((np.zeros((2, second_diff.shape[1])), second_diff))
            X = np.hstack((X, second_diff))

        # ---------- delay embedding ----------
        if self.delay_embedding:
            tau_max = self.embedding_delay

            X_save = X[:-tau_max]
            X_temp = X[:-tau_max, 0:temp_feats]

            for tau in range(1, tau_max + 1):
                num_feats += temp_feats
                X_temp = np.hstack(
                    (X_temp, X[tau_max - tau:-tau, 0:temp_feats])
                )

            X_temp = X_temp[:, temp_feats:]
            X = np.hstack((X_save, X_temp))
            y = y[tau_max:]

        # ---------- dimensionality reduction ----------
        if self.dimensionality_reduction:
            pca = PCA(n_components=self.reduced_dimensions)
            X = pca.fit_transform(X)
            y = pca.fit_transform(y)
            num_feats = self.reduced_dimensions + 1

        self.num_feats = num_feats
        self.num_outputs = num_outputs

        # --------------------------------------------------
        # BUILD SEQUENCES
        # --------------------------------------------------
        X_seq, y_seq = self._reshape_sequences(
            X, y,
            num_feats=num_feats,
            num_outputs=num_outputs
        )


        return X_seq, y_seq
    # ---------------------------------------------------------------------
    def _reshape_sequences(self, X, y, num_feats = 1, num_outputs = 1):
        """Helper: reshape into sequences and trim for batch_size if needed"""

        if X is None or y is None:
            raise ValueError('X and y must contain data')
        
        if not self.fixed_sl:
            seq_len = self.params['seq_len']

            if X.shape[0] < seq_len:
                raise ValueError(f"Not enough samples ({X.shape[0]}) for seq_len={seq_len}")

            valid_rows = (X.shape[0] // seq_len) * seq_len
            X, y = X[:valid_rows], y[:valid_rows]

            num_seq = valid_rows // seq_len
            num_feats = X.shape[-1]
            num_outputs = y.shape[-1]

            X = X.reshape(num_seq, seq_len, num_feats)
            y = y.reshape(num_seq, seq_len, num_outputs)

        if self.params['stateful']:
            batch_size = self.params['batch_size']
            if X.shape[0] < batch_size:
                raise ValueError("Not enough sequences for stateful training.")

            max_seq = X.shape[0] - (X.shape[0] % batch_size)
            X, y = X[:max_seq], y[:max_seq]

        return X, y
    # ---------------------------------------------------------------------

    def hankel_matrix_multivariate(self, X, L):
        """
        X: (N, D) array, e.g., Lorenz with D=3 variables (X,Y,Z)
        L: window length
        Returns: Hankel matrix H of shape (L*D, N-L+1)
        """
        N, D = X.shape
        K = N - L + 1                   # number of sliding windows
        H = np.empty((L * D, K))

        row = 0
        for d in range(D):
            xd = X[:, d]
            for i in range(L):
                H[row] = xd[i:i+K]
                row += 1
                
        return H
     # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    def hankel_svd_features(self, X, L, r):
        """
        X: (N, D) time series
        L: window length
        r: number of dominant singular values/features
        Returns: feature matrix F with shape (K, r)
        - Each sample is based on Hankel windowed structure
        - Ready to feed into ML models
        """
        H = self.hankel_matrix_multivariate(X, L)
        U, S, Vt = np.linalg.svd(H, full_matrices=False)

        # Keep r strongest components
        Vt_r = Vt[:r, :]             # shape (r, K)

        # Transpose => samples × features
        F = Vt_r.T                   # shape (K, r)
        return F
    # ---------------------------------------------------------------------
    def _prepare_data_class(self, data, dataset_parameters, normalize = True):

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        if data is None:
            raise ValueError('data must be provided')
        
        if len(dataset_parameters['inputs']) == 0:
            raise ValueError('inputs must be provided')
        
        if len(dataset_parameters['outputs']) > 1 and self.params['MISO']:
            raise ValueError('MISO is only supported for single output')

        # Convert data to numpy
        data = np.array(data)

        # Normalize
        if normalize:
            data = self.scaler.transform(data)

        X, y = data[:, dataset_parameters['inputs']], data[:, dataset_parameters['outputs']]

        # # forecast, output is shifted by forecast steps
        # if self.params['forecast_size'] > 1:
        #     X = X[:-self.params['forecast_size'], :]
        #     y = y[self.params['forecast_size'] - 1:, :]

        # NEW_X = []
        # NEW_y = []

        # if self.params['rolling']:
        #     for i in range(0, np.size(X, axis = 0) - self.params['seq_len'] + 1, self.params['forecast_size']):
        #         NEW_X.extend(X[i: i + self.params['seq_len'], : ])
        #         NEW_y.extend(y[i: i + self.params['seq_len'], : ])

        #     X = np.array(NEW_X)
        #     y = np.array(NEW_y)

        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        y = to_categorical(y)

        num_feats = len(dataset_parameters['inputs']) + (len(dataset_parameters['outputs']) if self.params['TF'] else 0)
        num_outputs = y.shape[1]

        return self._reshape_sequences(X, y, num_feats, num_outputs)
    # ---------------------------------------------------------------------
    def measure_model_time(self, clean_data, fault_data, dataset_parameters, LENGTH=0):
        """Train on clean data, measure train & test time"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if clean_data is None:
            raise ValueError('clean_data must be provided')
        if fault_data is None:
            raise ValueError('fault_data must be provided')

        results = {}

        # fit scaler on clean data
        self.scaler.fit(clean_data)

        # prepare clean data
        X_clean, y_clean = self._prepare_data(clean_data, dataset_parameters)

        # measure training
        start = time.perf_counter()
        self.model.fit(
            X_clean, y_clean,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=0, shuffle=False
        )
        results['train'] = time.perf_counter() - start

        # prepare fault data
        X_fault, _ = self._prepare_data(fault_data, dataset_parameters)

        # measure prediction
        start = time.perf_counter()
        self.model.predict(X_fault, batch_size=self.params['batch_size'], verbose=0)
        results['test'] = time.perf_counter() - start

        return results
    # ---------------------------------------------------------------------
    def train_model(
            self,
            data,
            dataset_parameters=None,
            metric='mse',
            test=False,
            fixed_sl=False,
            all_data=None,
            retrain=False,
            fit_scaler=True,
            validation=False,
            early_stop=False,
            scale_per_rho=False,
            scaler_mode=2,
            verbose=0):
        """
        Train model and optionally compute residuals.
        """

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------
        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')

        if data is None:
            raise ValueError('data must be provided')

        if self.model is None:
            self.create_model()
            if retrain:
                raise ValueError('Model must be provided when retrain is True')

        self.fixed_sl = fixed_sl

        # --------------------------------------------------
        # HANDLE DROPPED COLUMNS
        # --------------------------------------------------
        dropped_cols = dataset_parameters.get('dropped_columns')

        if dropped_cols is not None:
            for col in dropped_cols:
                data[:, col] = 1.0
                if all_data is not None:
                    all_data[:, col] = 1.0

        # --------------------------------------------------
        # SCALER INITIALIZATION
        # --------------------------------------------------
        if self.scaler is None:
            if scaler_mode == 1:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            elif scaler_mode == 2:
                self.scaler = StandardScaler()
            else:
                raise ValueError("Invalid scaler_mode")

        # --------------------------------------------------
        # SCALER FITTING
        # --------------------------------------------------
        if fit_scaler:

            if scale_per_rho:
                if all_data is None:
                    raise ValueError(
                        'all_data must be provided when scale_per_rho is True'
                    )

                scalers = {}
                for rho in np.unique(all_data[:, 6]):

                    scaler = (
                        MinMaxScaler(feature_range=(0, 1))
                        if scaler_mode == 1
                        else StandardScaler()
                    )

                    scaler.fit(all_data[all_data[:, 6] == rho])
                    scalers[rho] = scaler

                self.multi_scaler = scalers

            else:
                self.scaler.fit(all_data if all_data is not None else data)

        # --------------------------------------------------
        # DATA PREPARATION
        # --------------------------------------------------
        X, y = self._prepare_data(
            data=data,
            dataset_parameters=dataset_parameters
        )

        self.X_train, self.y_train = X, y

        # --------------------------------------------------
        # STATEFUL TRIMMING
        # --------------------------------------------------
        if self.params['stateful']:
            batch_size = self.params['batch_size']
            max_seq = self.X_train.shape[0] - (
                self.X_train.shape[0] % batch_size
            )

            self.X_train = self.X_train[:max_seq]
            self.y_train = self.y_train[:max_seq]

        # --------------------------------------------------
        # VALIDATION SPLIT (LAST 10%)
        # --------------------------------------------------
        if validation:
            # split_idx = int(self.X_train.shape[0] * self.params['batch_size'])
            split_idx = self.params['batch_size']


            self.X_valid = self.X_train[-split_idx:]
            self.y_valid = self.y_train[-split_idx:]

            self.X_train = self.X_train[:-split_idx]
            self.y_train = self.y_train[:-split_idx]

        # --------------------------------------------------
        # TRAINING
        # --------------------------------------------------
        if self.logging:
            print('Training model...')

        callbacks = []

        if early_stop:
            # callbacks.append(
            #     keras.callbacks.EarlyStopping(
            #         monitor=self.params.get('monitor', 'val_loss'),
            #         patience=self.params.get('patience', 20),
            #         restore_best_weights=True
            #     )
            # )
            callbacks.append(self.early_stopping)

        fit_kwargs = dict(
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=verbose,
            shuffle=False,
            callbacks=callbacks
        )

        if validation:
            fit_kwargs["validation_data"] = (
                self.X_valid,
                self.y_valid
            )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            **fit_kwargs
        )

        if self.logging:
            print('Model trained.')

        # --------------------------------------------------
        # OPTIONAL TESTING
        # --------------------------------------------------
        if not test:
            return None

        self.y_hat, self.h_state, self.c_state = self.model.predict(
            self.X_test,
            batch_size=self.params['batch_size'],
            verbose=0
        )

        self.residuals = self.metrics.compute_residuals(
            self.y_test,
            self.y_hat,
            metric
        )

        return self.residuals
        # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    def train_model_tf(self, 
                    data, 
                    dataset_parameters = None, 
                    metric = 'mse', 
                    test = False, 
                    fixed_sl = False, 
                    all_data = None, 
                    retrain = False,
                    fit_scaler = True,
                    validation = False,
                    early_stop = False,
                    scale_per_rho = False,
                    scaler_mode = 2,
                    verbose = 0,
                    tf_train = True,
                    mode = "sequence"):

        """
        Train model and return residuals
        mode = seq or point
            seq = sequence model, train with seq to seq forecast
            point = point model, train with point to point forecast
        tf_train = teacher forcing train
            train with real values and teacher forcing
        during testing the model can be tested with point to point or sequence to sequence
            Take a point and predict n points or take a sequence and predict n points with continuously previous predicted values as inputs.
        """

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        if data is None:
            raise ValueError('data must be provided')
        
        if self.model is None:
            self.create_model()
            if retrain:
                raise ValueError('Model must be provided when retrain is True')
        
        self.fixed_sl = fixed_sl

        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
                if all_data is not None:
                    all_data[:, col] = 1.0

        if self.scaler is None:
            if scaler_mode == 1:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            elif scaler_mode == 2:
                self.scaler = StandardScaler()

        # normalize
        if scale_per_rho and fit_scaler is True:
            if all_data is None:
                raise ValueError('all_data must be provided when scale_per_rho is True')
            
            scalers = {}
            for rho in np.unique(all_data[:, 6]):
                if scaler_mode == 1:
                    scalers[rho] = MinMaxScaler(feature_range=(0, 1))
                elif scaler_mode == 2:
                    scalers[rho] = StandardScaler()
                scalers[rho].fit(all_data[all_data[:, 6] == rho])

            self.multi_scaler = scalers

        else:
            if fit_scaler is True and all_data is not None:
                self.scaler.fit(all_data)
            elif fit_scaler is True and all_data is None:
                self.scaler.fit(data)

        # --------------------------------------------------
        # DATA PREPARATION MODE SELECTION
        # --------------------------------------------------
        if tf_train or mode == "point":
            X, y = self._prepare_data_tf(
                data=data,
                dataset_parameters=dataset_parameters
            )
        else:
            X, y = self._prepare_data(
                data=data,
                dataset_parameters=dataset_parameters
            )

        self.X_train, self.y_train = X, y

        # If the model is statefull, trim to the maximum multiples of batch size
        if self.params['stateful']:
            maximum_sequences = self.X_train.shape[0] - self.X_train.shape[0] % \
                                     self.params['batch_size']

            
            self.X_train = self.X_train[:maximum_sequences]
            self.y_train = self.y_train[:maximum_sequences]

        # Create validation during training set from 10% of training data
        if validation:
            self.X_valid = self.X_train[-int(self.X_train.shape[0] * 0.1):]
            self.y_valid = self.y_train[-int(self.y_train.shape[0] * 0.1):]

            self.X_train = self.X_train[:-int(self.X_train.shape[0] * 0.1)]
            self.y_train = self.y_train[:-int(self.y_train.shape[0] * 0.1)]


        if self.logging:
            print('Training model...')
        
        callbacks = []
        if early_stop:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=self.params.get('monitor', 'val_loss'),
                    patience=self.params.get('patience', 20),
                    restore_best_weights=True
                )
            )
    
        if validation is False:
            self.history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                verbose=verbose, shuffle=False,
                callbacks = callbacks,
            )
        else:
            self.history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                verbose=verbose, shuffle=False,
                validation_data = (self.X_valid, self.y_valid),
                callbacks = callbacks
            )


        if self.logging:
            print('Model trained.')

        if test == False:
            return None
        else:
            self.y_hat, self.h_state, self.c_state = self.model.predict(self.X_test, batch_size=self.params['batch_size'], verbose=0)

            self.residuals = self.metrics.compute_residuals(self.y_test, self.y_hat, metric)

            return self.residuals
    # ---------------------------------------------------------------------       
    def train_model_feedback(self, 
                        data, 
                        dataset_parameters=None, 
                        metric='mse', 
                        test=False, 
                        fixed_sl=False, 
                        all_data=None, 
                        retrain=False,
                        fit_scaler=True,
                        validation=False,
                        early_stop=False,
                        error_mode='mean',
                        error_scale='all',
                        error_mode_metric='standard',
                        first_only=False,
                        all_variables_individual_input=False,):
        
        """Train model with error-feedback loop (batch-safe, fast, graph-only)"""


        scaler_temp = StandardScaler()
        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')

        if self.model is None:
            self.create_model()
            print("Created new model for feedback training.")
            if retrain:
                raise ValueError('Model must be provided when retrain=True')

        self.fixed_sl = fixed_sl

        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
                if all_data is not None:
                    all_data[:, col] = 1.0

        if self.scaler is None:
            self.scaler = MinMaxScaler()
            # self.scaler = StandardScaler()

        if fit_scaler and all_data is not None:
            self.scaler.fit(all_data)
        elif fit_scaler:
            self.scaler.fit(data)

        X, y = self._prepare_data(data=data, dataset_parameters=dataset_parameters)
        self.X_train, self.y_train = X, y

        # Handle stateful trimming
        if self.params['stateful']:
            max_seq = self.X_train.shape[0] - self.X_train.shape[0] % self.params['batch_size']
            self.X_train = self.X_train[:max_seq]
            self.y_train = self.y_train[:max_seq]

        # Train/valid split
        if validation:
            split = self.params['batch_size']
            self.X_valid = self.X_train[-split:]
            self.y_valid = self.y_train[-split:]
            self.X_train = self.X_train[:-split]
            self.y_train = self.y_train[:-split]

        if self.logging:
            print("Training model with error feedback...")



        batch_size = self.params['batch_size']

        train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)) \
                                .batch(batch_size) \
                                .prefetch(tf.data.AUTOTUNE) \
                                .cache()
                                # .shuffle(len(self.X_train), reshuffle_each_iteration=True)

        loss_fn = self.model.compiled_loss
        optimizer = self.model.optimizer

        seq_len = self.params['seq_len']


        @tf.function
        def train_step(x, y, error_vector):

            # Concatenate feedback error to the input
            error_vector = tf.cast(error_vector, x.dtype)
            x = tf.concat([x, error_vector], axis=2)   # shape: (B, seq_len, features+1)

            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True, batch_size=batch_size)  # shape: (B, seq_len, outputs)
                loss = loss_fn(y, predictions)

            if error_mode != 'zero':

                # compute error over whole seq or only forecast horizon
                if error_scale == 'all':
                    err = self.error_computation(
                        tf.cast(predictions, tf.float32), tf.cast(y, tf.float32), error_mode_metric)
                elif error_scale == 'forecast':
                    err = self.error_computation(
                        tf.cast(predictions[:, :self.params['forecast_size'], :], tf.float32),
                        tf.cast(y[:, :self.params['forecast_size'], :], tf.float32),
                        error_mode_metric
                    )

                if all_variables_individual_input == False:
                    err = tf.reduce_mean(err, axis=-1, keepdims=True)
                else:
                    pass  # keep all variable errors

                # err = scaler_temp.fit_transform(err)

                # Reduce to one scalar PER BATCH SAMPLE
                # stat shape: (B, 1, 1)
                if error_mode == 'max':
                    stat = tf.reduce_max(err, axis=[1, 2], keepdims=True)
                elif error_mode == 'sum':
                    stat = tf.reduce_sum(err, axis=[1, 2], keepdims=True)
                elif error_mode == 'min':
                    stat = tf.reduce_min(err, axis=[1, 2], keepdims=True)
                elif error_mode == 'std':
                    stat = tf.math.reduce_std(err, axis=[1, 2], keepdims=True)
                elif error_mode == 'var':
                    stat = tf.math.reduce_variance(err, axis=[1, 2], keepdims=True)
                elif error_mode == 'mean':
                    stat = tf.reduce_mean(err, axis=[1, 2], keepdims=True)
                else:
                    stat = tf.zeros_like(err[:, :1, :])    # fallback

                # Broadcast back to full sequence: (B, seq_len, 1)
                if error_mode == "all_errors":
                    err_new = err
                else:
                    # If first only, repeat only for first time step, rest zero!
                    if first_only:
                        bs = tf.shape(err)[0]
                        sl    = tf.shape(err)[1]
                        feat_dim   = tf.shape(err)[2]

                        mask = tf.concat(
                            [
                                tf.ones((bs, 1, feat_dim), dtype=err.dtype),
                                tf.zeros((bs, sl - 1, feat_dim), dtype=err.dtype)
                            ],
                            axis=1
                        )
                        err_new = stat * mask  
                    else:
                        err_new = tf.ones_like(err) * stat  

            else:
                # zero error feedback
                err_new = tf.zeros_like(error_vector)

            # print err_new
            #tf.print(err_new, summarize=-1)

            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            return loss, err_new

        error_vector = tf.zeros((batch_size, seq_len, 1), dtype=tf.float32)

        history = {"loss": [], "val_loss": []}

        for epoch in range(self.params['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")

            epoch_loss = 0.0
            steps = 0
            val_loss = 0.0

            for batch_x, batch_y in train_ds:
                loss_value, error_vector = train_step(batch_x, batch_y, error_vector)
                epoch_loss += loss_value
                steps += 1
                # self.model.layers[0].reset_states()  # reset states after each batch if stateful
                # self.model.layers[1].reset_states()


            train_loss = float(epoch_loss / steps)
            history["loss"].append(train_loss)

            print(f"Train Loss: {train_loss:.4f}")

            if validation:
                error_vector = tf.cast(error_vector, self.X_valid.dtype)
                xvalid = tf.concat([self.X_valid, error_vector], axis=2)
                res = self.model(xvalid, training=False, batch_size=batch_size)
                res = loss_fn(self.y_valid, res)
                history["val_loss"].append(res)
                print(f"Validation Loss: {res:.4f}")

        self.history = history

        if self.logging:
            print("Model trained.")
    # --------------------------------------------------------------------- 
    def train_model_classification(self, 
                    data, 
                    dataset_parameters = None, 
                    metric = 'mse', 
                    test = False, 
                    fixed_sl = False, 
                    all_data = None, 
                    retrain = False,
                    fit_scaler = True,
                    validation = False,
                    early_stop = False):
        
        """Train model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        if data is None:
            raise ValueError('data must be provided')
        
        if self.model is None:
            self.create_model()
            if retrain:
                raise ValueError('Model must be provided when retrain is True')
        
        self.fixed_sl = fixed_sl

        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
                if all_data is not None:
                    all_data[:, col] = 1.0

        if self.scaler is None:
            self.scaler = MinMaxScaler()

        # normalize
        if fit_scaler is True and all_data is not None:
            self.scaler.fit(all_data)
        elif fit_scaler is True and all_data is None:
            self.scaler.fit(data)

        X, y = self._prepare_data_class(data = data, dataset_parameters = dataset_parameters)

        self.X_train, self.y_train = X, y

        # If the model is statefull, trim to the maximum multiples of batch size
        if self.params['stateful']:
            maximum_sequences = self.X_train.shape[0] - self.X_train.shape[0] % \
                                     self.params['batch_size']

            
            self.X_train = self.X_train[:maximum_sequences]
            self.y_train = self.y_train[:maximum_sequences]

        # Create validation during training set from 10% of training data
        if validation:
            self.X_valid = self.X_train[-int(self.X_train.shape[0] * 0.1):]
            self.y_valid = self.y_train[-int(self.y_train.shape[0] * 0.1):]

            self.X_train = self.X_train[:-int(self.X_train.shape[0] * 0.1)]
            self.y_train = self.y_train[:-int(self.y_train.shape[0] * 0.1)]


        if self.logging:
            print('Training model...')
        
        callbacks = []
        if early_stop:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=self.params.get('monitor', 'val_loss'),
                    patience=self.params.get('patience', 20),
                    restore_best_weights=True
                )
            )
    
        if validation is False:
            self.history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                verbose=0, shuffle=False,
                callbacks = callbacks,
            )
        else:
            self.history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                verbose=0, shuffle=False,
                validation_data = (self.X_valid, self.y_valid),
                callbacks = callbacks
            )


        if self.logging:
            print('Model trained.')

        if test == False:
            return None
        else:
            self.y_hat = self.model.predict(self.X_test, batch_size=self.params['batch_size'], verbose=0)

            self.residuals = self.metrics.compute_residuals(self.y_test, self.y_hat, metric)

            return self.residuals
    # --------------------------------------------------------------------- 
    def train_model_skip(self, 
                    data, 
                    dataset_parameters = None, 
                    metric = 'mse', 
                    test = False, 
                    fixed_sl = False, 
                    all_data = None, 
                    retrain = False,
                    fit_scaler = True):
        
        """Train model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        if data is None:
            raise ValueError('data must be provided')
        
        if self.model is None:
            self.create_model()
            if retrain:
                raise ValueError('Model must be provided when retrain is True')
        
        self.fixed_sl = fixed_sl

        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
                if all_data is not None:
                    all_data[:, col] = 1.0

        if self.scaler is None:
            self.scaler = MinMaxScaler()

        # normalize
        if fit_scaler is True and all_data is not None:
            self.scaler.fit(all_data)
        elif fit_scaler is True and all_data is None:
            self.scaler.fit(data)

        X, y = self._prepare_data(data = data, dataset_parameters = dataset_parameters)

        # Skip every second sequence in the training data
        X = X[::2]
        y = y[::2]

        self.X_train, self.y_train = X, y

        # If the model is statefull, trim to the maximum multiples of batch size
        if self.params['stateful']:
            maximum_sequences = self.X_train.shape[0] - self.X_train.shape[0] % \
                                     self.params['batch_size']

            
            self.X_train = self.X_train[:maximum_sequences]
            self.y_train = self.y_train[:maximum_sequences]


        if self.logging:
            print('Training model...')

        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=0, shuffle=False
        )
        if self.logging:
            print('Model trained.')

        if test == False:
            return None
        else:
            self.y_hat, self.h_state, self.c_state  = self.model.predict(self.X_test, batch_size=self.params['batch_size'], verbose=0)

            self.residuals = self.metrics.compute_residuals(self.y_test, self.y_hat, metric)

            return self.residuals
    # --------------------------------------------------------------------- 
    def train_retrain_model(self, features, targets, dataset_parameters = None, metric = 'mse', test = False,  all_data = None, fit_scaler = True):
        """Train model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        if targets is None or features is None:
            raise ValueError('features and targets must be provided')
        
        if self.model is None:
            self.create_model()


        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                features[:, col] = 1.0
                targets[:, col] = 1.0
                if all_data is not None:
                    all_data[:, col] = 1.0

        # normalize
        if fit_scaler is True and all_data is not None:
            self.scaler.fit(all_data)
        elif fit_scaler is True and all_data is None:
            self.scaler.fit(targets)


        # The features are the predictions from the first model, the targets are the real values
        X, _= self._prepare_data(data = features, dataset_parameters = dataset_parameters)
        _, y = self._prepare_data(data = targets, dataset_parameters = dataset_parameters)

        self.X_train, self.y_train = X, y

        # If the model is statefull, trim to the maximum multiples of batch size
        if self.params['stateful']:
            maximum_sequences = self.X_train.shape[0] - self.X_train.shape[0] % \
                                     self.params['batch_size']

            
            self.X_train = self.X_train[:maximum_sequences]
            self.y_train = self.y_train[:maximum_sequences]


        if self.logging:
            print('Training model...')

        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=0, shuffle=False,
            
        )
        if self.logging:
            print('Model trained.')

        if test == False:
            return None
        else:
            self.y_hat, self.h_state, self.c_state  = self.model.predict(self.X_test, batch_size=self.params['batch_size'], verbose=0)

            self.residuals = self.metrics.compute_residuals(self.y_test, self.y_hat, metric)

            return self.residuals
     # ---------------------------------------------------------------------
    def transfer_learning(self, 
                    data, 
                    dataset_parameters = None, 
                    all_data = None):
        
        """Train model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        if data is None:
            raise ValueError('data must be provided')
        
        if self.model is None:
            raise ValueError('Model must be provided, load model first (use load_model())')
        
        if self.scaler is None:
            raise ValueError('Scaler must be provided (use load_scaler())')
        
        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
                if all_data is not None:
                    all_data[:, col] = 1.0

        # Preparee Data
        X, y = self._prepare_data(data = data, dataset_parameters = dataset_parameters)

        self.X_train, self.y_train = X, y

        # If the model is statefull, trim to the maximum multiples of batch size
        if self.params['stateful']:
            maximum_sequences = self.X_train.shape[0] - self.X_train.shape[0] % \
                                     self.params['batch_size']
           
            self.X_train = self.X_train[:maximum_sequences]
            self.y_train = self.y_train[:maximum_sequences]

        if self.logging:
            print('Training model...')


        self.model_transfer = self.create_transfer_model(dataset_parameters=dataset_parameters, test = False)

        self.model_transfer.summary()

        # Create transfer learning model and freeze layers
        self.model_transfer.fit(
            self.X_train, self.y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=0, shuffle=False
        )

        if self.logging:
            print('Model trained.')

        self.model_transfer_test = self.create_transfer_model(dataset_parameters=dataset_parameters, test = True)

        self.model_transfer_test.summary()
     # ---------------------------------------------------------------------
    def test_model(self, data, dataset_parameters, metric = 'mse', fixed_sl = False, scale_per_rho = False):
        """Test model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')
        
        self.fixed_sl = fixed_sl

        if scale_per_rho == True and self.multi_scaler is None:
            raise ValueError('multi_scaler must be provided')
        
        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
        
        X, self.y = self._prepare_data(data, dataset_parameters)

        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % \
                                        self.params['batch_size']
            
            X = X[:maximum_sequences]
            self.y  = self.y[:maximum_sequences]
        
        if self.logging:
            print('Testing model...')

        self.y_hat  = self.model_test.predict(X, 
                                        batch_size=1, 
                                        verbose=0,
                                        )

        # Compile forward pass once
        # @tf.function
        # def forward_pass(x):
        #     return self.model_test(x, training=False)

        # # Call it and convert to numpy
        # self.y_hat = forward_pass(tf.convert_to_tensor(X, dtype=tf.float32)).numpy()

        if self.logging:
            print('Model tested.')

        # self.y_hat = self.scaler.inverse_transform(self.y_hat)
        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            self.y = self.y.flatten()
        else:
            self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[2])
            self.y = self.y.reshape(-1, self.y.shape[2])

        resid_vector, residuals = self.metrics.compute_residuals(self.y, self.y_hat, metric)

        results = {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': self.y,
            'y_hat': self.y_hat
        }
        
        return  results
     # ---------------------------------------------------------------------
    def test_model_tf(self,
                    data,
                    dataset_parameters,
                    metric='mse',
                    fixed_sl=False,
                    scale_per_rho=False):
        
        """Test model using teacher-forcing rollout (autoregressive prediction)"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')

        self.fixed_sl = fixed_sl

        if scale_per_rho and self.multi_scaler is None:
            raise ValueError('multi_scaler must be provided')

        # --------------------------------------------------
        # dropped columns handling (same as original)
        # --------------------------------------------------
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0

        # --------------------------------------------------
        # TF data preparation
        # --------------------------------------------------
        temp_forecast_size = self.params['forecast_size']
        self.params['forecast_size'] = 1  # for teacher forcing, we predict one step at a time and feed it back into the input
        
        X, self.y = self._prepare_data(data, dataset_parameters)
        self.params['forecast_size'] = temp_forecast_size

        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % self.params['batch_size']
            X = X[:maximum_sequences]
            self.y = self.y[:maximum_sequences]

        if self.logging:
            print('Testing model (teacher forcing rollout)...')

        horizon = self.params['forecast_size']
        num_outputs = self.num_outputs

        all_predictions = []

        # --------------------------------------------------
        # AUTOREGRESSIVE ROLLOUT
        # --------------------------------------------------
        for seq_id in range(X.shape[0]):

            window = X[seq_id].copy()
            seq_preds = []
            print(f"Predicting sequence {seq_id + 1}/{X.shape[0]}...")
            for _ in range(horizon):

                pred = self.model_test.predict(
                    window[np.newaxis, :, :],
                    batch_size=1,
                    verbose=0
                )[0][0]

                seq_preds.append(pred)

                # slide window
                window = np.roll(window, -1, axis=0)

                # insert prediction as newest timestep
                window[-1, :num_outputs] = pred

            all_predictions.append(np.array(seq_preds))

        self.y_hat = np.array(all_predictions)

        if self.logging:
            print('Model tested.')

        # --------------------------------------------------
        # MATCH ORIGINAL OUTPUT FORMAT
        # --------------------------------------------------
        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            self.y = self.y.flatten()
        else:
            self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[-1])
            self.y = self.y.reshape(-1, self.y.shape[-1])

        # resid_vector, residuals = self.metrics.compute_residuals(
        #     self.y,
        #     self.y_hat,
        #     metric
        # )

        results = {
            # 'resid_vector': resid_vector,
            # 'residuals': residuals,
            'y': self.y,
            'y_hat': self.y_hat
        }

        return results
    def test_model_tf2(self,
                    data,
                    dataset_parameters,
                    metric='mse',
                    fixed_sl=False,
                    scale_per_rho=False):

        """
        Teacher-forcing evaluation:
        SL real conditioning and FH autonomous rollout
        """

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')

        self.fixed_sl = fixed_sl

        if scale_per_rho and self.multi_scaler is None:
            raise ValueError('multi_scaler must be provided')

        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0

        original_fh = self.params['forecast_size']
        self.params['forecast_size'] = 1

        original_bs = self.params['batch_size']
        self.params['batch_size'] = 1

        X, self.y = self._prepare_data(data, dataset_parameters)

        X = X.reshape(-1, X.shape[-1])

        self.params['forecast_size'] = original_fh
        self.params['batch_size'] = original_bs
        
        if self.logging:
            print('Testing model (teacher forcing rollout)...')

        horizon = original_fh
        seq_len = self.params['seq_len']
        num_outputs = self.num_outputs

        all_predictions = []

        seq_id = 0

        while seq_id < X.shape[0] - horizon - seq_len + 1:

            start = seq_id
            end = seq_id + seq_len

            # Teacher forcing (state adaptation)
            window = X[start:end].copy()
            # window = X[seq_id].copy()

            # run SL real steps to adapt hidden state
            _ = self.model_test.predict(
                window[np.newaxis, :, :],
                batch_size=1,
                verbose=0
            )

            # Autonomous rollout (TRUE FORECAST)
            print(f"Predicting sequence {seq_id + 1}/{X.shape[0]}...")
            for _ in range(horizon):

                pred = self.model_test.predict(
                    window[np.newaxis, :, :],
                    batch_size=1,
                    verbose=0
                )[0][0]

                all_predictions.append(pred)

                # slide window
                window = np.roll(window, -1, axis=0)

                # feedback prediction
                window[-1, :num_outputs] = pred

            seq_id += horizon

        self.y_hat = np.array(all_predictions)

        if self.logging:
            print('Model tested.')

        y_true = self.y.reshape(-1, self.y.shape[-1])
        y_true = y_true[:len(self.y_hat)]

        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            y_true = y_true.flatten()

        resid_vector, residuals = self.metrics.compute_residuals(
            y_true,
            self.y_hat,
            metric
        )

        return {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': y_true,
            'y_hat': self.y_hat
        }

    def test_model_tf3(self,
                    data,
                    dataset_parameters,
                    metric='mse',
                    fixed_sl=False,
                    scale_per_rho=False):

        """
        Teacher-forcing evaluation:
        SL real conditioning and FH autonomous rollout
        """

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')

        self.fixed_sl = fixed_sl

        if scale_per_rho and self.multi_scaler is None:
            raise ValueError('multi_scaler must be provided')

        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0

        original_fh = self.params['forecast_size']
        self.params['forecast_size'] = 1

        original_bs = self.params['batch_size']
        self.params['batch_size'] = 1

        X, self.y = self._prepare_data(data, dataset_parameters)

        X = X.reshape(-1, X.shape[-1])

        self.params['forecast_size'] = original_fh
        self.params['batch_size'] = original_bs

        if self.logging:
            print('Testing model (teacher forcing rollout)...')

        horizon = original_fh
        seq_len = self.params['seq_len']
        num_outputs = self.num_outputs

        # compiled fast forward pass
        @tf.function(reduce_retracing=True)
        def forward_step(x):
            return self.model_test(x, training=False)

        all_predictions = []

        seq_id = seq_len  # start at the end of the first sequence to have a full window for teacher forcing


        while seq_id < X.shape[0] - horizon - seq_len + 1:

            start = seq_id - seq_len
            end = seq_id 

            # Teacher forcing (state adaptation)
            window = X[start:end].copy()

            for layer in self.model_test.layers:
                if hasattr(layer, "reset_states"):
                    # print(f"Resetting states of layer: {layer.name}")
                    layer.reset_states()

            _ = forward_step(
                tf.convert_to_tensor(window[np.newaxis, :, :], dtype=tf.float32)
            )

            # Autonomous rollout (TRUE FORECAST)
            input_data_window = window.copy()[-1:, :]
            # print(f"Predicting sequence {seq_id + 1}/{X.shape[0]}...")
            for _ in range(horizon):

                pred = forward_step(
                    tf.convert_to_tensor(input_data_window[np.newaxis, :, :], dtype=tf.float32)
                )[0].numpy()

                all_predictions.append(pred[0])

                # slide window
                # window[:-1] = window[1:]

                # feedback prediction
                # window[-1, :num_outputs] = pred
                input_data_window = pred

            seq_id += horizon

        self.y_hat = np.array(all_predictions)

        if self.logging:
            print('Model tested.')

        y_true = self.y.reshape(-1, self.y.shape[-1])
        y_true = y_true[seq_len:len(self.y_hat) + seq_len]

        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            y_true = y_true.flatten()

        resid_vector, residuals = self.metrics.compute_residuals(
            y_true,
            self.y_hat,
            metric
        )

        return {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': y_true,
            'y_hat': self.y_hat
        }

    def error_computation(self, y, y_hat, error_metric = 'standard'):
        """Compute error vector based on selected error mode"""

        if error_metric == 'standard':
            error_vector = y - y_hat
        elif error_metric == 'absolute':
            error_vector = tf.math.abs(y - y_hat)
        elif error_metric == 'squared':
            error_vector = (y_hat - y) ** 2
        elif error_metric == 'constant':
            error_vector = tf.ones_like(y_hat) - 1.5
        else:
            raise ValueError('Invalid error_metric specified')

        return error_vector
    # ---------------------------------------------------------------------
    def test_model_feedback(
            self, 
            data, 
            dataset_parameters, 
            metric='mse', 
            fixed_sl=False, 
            error_mode='mean',
            error_scale='all',
            error_mode_metric='standard',
            first_only=False,
            differential_feedback=False
        ):
        scaler_temp = StandardScaler()
        """Batch-capable test loop with feedback error vector."""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')

        self.fixed_sl = fixed_sl

        # Replace dropped columns with constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0

        X, self.y = self._prepare_data(data, dataset_parameters)
        
        # Ensure divisible by batch_size for stateful LSTM
        if self.params['stateful']:
            b = self.params['batch_size']
            max_seq = X.shape[0] - X.shape[0] % b
            X = X[:max_seq]
            self.y = self.y[:max_seq]

        if self.logging:
            print(f'Testing model on {X.shape[0]} samples...')

        # Batching
        batch_size = self.params['batch_size']
        num_batches = int(np.ceil(X.shape[0] / batch_size))

        y_hats = []

        error_vector = np.zeros((batch_size, self.params['seq_len'], 1))

        for b in range(num_batches):

            start = b * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = self.y[start:end]

            # Attach error vector
            #X_batch = np.concatenate((X_batch[:, :, :-1], error_vector), axis=2)
            X_batch = np.concatenate((X_batch, error_vector), axis=2)
           
            # Predict entire batch
            y_hat_batch = self.model_test.predict(
                X_batch,
                batch_size=batch_size,
                verbose=0
            )

            if error_scale == 'all':
                ev = self.error_computation(
                    tf.cast(y_hat_batch, tf.float32), 
                    tf.cast(y_batch, tf.float32), 
                    error_mode_metric
                )
            else:
                fs = self.params['forecast_size']
                ev = self.error_computation(
                    tf.cast(y_hat_batch, tf.float32)[:, :fs, :],
                    tf.cast(y_batch[:, :fs, :], tf.float32),
                    error_mode_metric
                )

            ev = np.mean(ev, axis=-1, keepdims=True)



            #ev = scaler_temp.fit_transform(ev)
            
            # Reduce to one scalar PER BATCH SAMPLE
            # stat shape: (B, 1, 1)
            if error_mode == 'max':
                stat = tf.reduce_max(ev, axis=[1, 2], keepdims=True)
            elif error_mode == 'sum':
                stat = tf.reduce_sum(ev, axis=[1, 2], keepdims=True)
            elif error_mode == 'min':
                stat = tf.reduce_min(ev, axis=[1, 2], keepdims=True)
            elif error_mode == 'std':
                stat = tf.math.reduce_std(ev, axis=[1, 2], keepdims=True)
            elif error_mode == 'var':
                stat = tf.math.reduce_variance(ev, axis=[1, 2], keepdims=True)
            elif error_mode == 'mean':
                stat = tf.reduce_mean(ev, axis=[1, 2], keepdims=True)
            else:
                stat = tf.zeros_like(ev[:, :1, :])  


            if error_mode == "all_errors":
                error_vector = ev
            else:
                # If first only, repeat only for first time step, rest zero!
                if first_only:
                    batch_size = tf.shape(ev)[0]
                    seq_len    = tf.shape(ev)[1]
                    feat_dim   = tf.shape(ev)[2]

                    mask = tf.concat(
                        [
                            tf.ones((batch_size, 1, feat_dim), dtype=ev.dtype),
                            tf.zeros((batch_size, seq_len - 1, feat_dim), dtype=ev.dtype)
                        ],
                        axis=1
                    )
                    error_vector = stat * mask  
                else:
                    error_vector = np.ones_like(ev) * stat  

            # print error_vector
            # tf.print(error_vector, summarize=-1)

            y_hats.append(y_hat_batch)

        self.y_hat = np.vstack(y_hats)

        if self.logging:
            print('Model tested.')

        # Format outputs
        if not self.params['MISO']:
            self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[2])
            self.y = self.y.reshape(-1, self.y.shape[2])
        else:
            self.y_hat = self.y_hat.flatten()
            self.y = self.y.flatten()

        resid_vector, residuals = self.metrics.compute_residuals(
            self.y, self.y_hat, metric
        )

        return {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': self.y,
            'y_hat': self.y_hat
        }
     # ---------------------------------------------------------------------
    def test_model_API(self, data, dataset_parameters, metric = 'mse', fixed_sl = False):
        """Test model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')
        
        self.fixed_sl = fixed_sl

        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
        
        X, self.y = self._prepare_data(data, dataset_parameters)

        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % \
                                        self.params['batch_size']
            
            X = X[:maximum_sequences]
            self.y  = self.y[:maximum_sequences]
        
        if self.logging:
            print('Testing model...')

        if self.params['model_type'] == 'LSTM':
            self.y_hat, self.h_state, self.c_state  = self.model_test.predict(X, 
                                            batch_size=1, 
                                            verbose=0,
                                            )
        else:
            self.y_hat, self.h_state,  self.c_state   = self.model_test.predict(X, 
                                batch_size=1, 
                                verbose=0,
                                )


        if self.logging:
            print('Model tested.')

        # self.y_hat = self.scaler.inverse_transform(self.y_hat)
        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            self.y = self.y.flatten()
        else:
            self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[2])
            self.y = self.y.reshape(-1, self.y.shape[2])

        resid_vector, residuals = self.metrics.compute_residuals(self.y, self.y_hat, metric)

        results = {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': self.y,
            'y_hat': self.y_hat,
            'h_state': self.h_state,
            'c_state': self.c_state
        }
        
        return  results
    
    def test_model_classification(self, data, dataset_parameters, metric = 'mse', fixed_sl = False):
        """Test model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')
        
        self.fixed_sl = fixed_sl

        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
        
        X, self.y = self._prepare_data_class(data, dataset_parameters)

        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % \
                                        self.params['batch_size']
            
            X = X[:maximum_sequences]
            self.y  = self.y[:maximum_sequences]
        
        if self.logging:
            print('Testing model...')

        self.y_hat = self.model_test.predict(X, 
                                        batch_size=1, 
                                        verbose=0,
                                        )
        if self.logging:
            print('Model tested.')

        self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[2])
        self.y = self.y.reshape(-1, self.y.shape[2])

        results = {
            'y': self.y,
            'y_hat': self.y_hat
        }
        
        return  results
    def test_model_skip(self, data, dataset_parameters, metric = 'mse', fixed_sl = False):
        """Test model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')
        
        self.fixed_sl = fixed_sl

        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
        
        X, self.y = self._prepare_data(data, dataset_parameters)

        # Skip every second sequence
        X = X[::2]
        self.y = self.y[::2]
        
        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % \
                                        self.params['batch_size']
            
            X = X[:maximum_sequences]
            self.y  = self.y[:maximum_sequences]
        
        if self.logging:
            print('Testing model...')

        self.y_hat, self.h_state, self.c_state  = self.model_test.predict(X, 
                                        batch_size=1, 
                                        verbose=0,
                                        )
        if self.logging:
            print('Model tested.')

        # self.y_hat = self.scaler.inverse_transform(self.y_hat)
        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            self.y = self.y.flatten()
        else:
            self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[2])
            self.y = self.y.reshape(-1, self.y.shape[2])

        resid_vector, residuals = self.metrics.compute_residuals(self.y, self.y_hat, metric)

        results = {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': self.y,
            'y_hat': self.y_hat,
            'h_state': self.h_state,
            'c_state': self.c_state
        }
        
        return  results
    
    def test_transfer_model(self, data, dataset_parameters, metric = 'mse'):
        """Test model and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_transfer_test is None:
            raise ValueError('Model must be provided')
        
        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
        
        X, self.y = self._prepare_data(data, dataset_parameters)

        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % \
                                        self.params['batch_size']
            
            X = X[:maximum_sequences]
            self.y  = self.y[:maximum_sequences]
        
        if self.logging:
            print('Testing model...')

        self.y_hat = self.model_transfer_test.predict(X, 
                                        batch_size=1, 
                                        verbose=0,
                                        )
        if self.logging:
            print('Model tested.')

        # self.y_hat = self.scaler.inverse_transform(self.y_hat)
        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            self.y = self.y.flatten()
        else:
            self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[2])
            self.y = self.y.reshape(-1, self.y.shape[2])

        resid_vector, residuals = self.metrics.compute_residuals(self.y, self.y_hat, metric)

        results = {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': self.y,
            'y_hat': self.y_hat
        }
        
        return  results
    
    # ---------------------------------------------------------------------
    def test_model_with_corrections(self, data, dataset_parameters, metric = 'mse'):
        """Test model with corrections and return residuals"""

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        if data is None:
            raise ValueError('data must be provided')
        if self.model_test is None:
            raise ValueError('Model must be provided')

        # If the columns are marked as dropped (e.g. time column, strings, etc), for now convert to numeric constant
        if dataset_parameters['dropped_columns'] is not None:
            for col in dataset_parameters['dropped_columns']:
                data[:, col] = 1.0
        
        # The features are the predictions from the first model, the targets are the real values
        X, self.y = self._prepare_data(data, dataset_parameters)

        # If the model is statefull, trim to the maximum multiples of batch size
        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % \
                                        self.params['batch_size']
            
            X = X[:maximum_sequences]
            self.y  = self.y[:maximum_sequences]
        
        if self.logging:
            print('Testing model...')


        # Prediction logic where the prediction is corrected at the end of the sequence with the real values
        # The rest of the prediction are made with the previous predictions
        # Example: [o1, o2, o3, o4, o5] -> [p6, p7, p8, p9, p10] 
        #          -> [p6, p7, p8, p9, p10] -> [p11, p12, p13, p14, p15]
        #          -> [o16, o17, o18, o19, o20] -> [p21, p22, p23, p24, p25] -> ...
        # where o are the real values and p are the predicted values
        # This is done by iterating over the sequences and predicting one sequence at a time, until the prediction horizon is reached
        # Then the next sequence is predicted, using the last sequence's real values as input
        # forecast_size gives the prediction horizon
        
        self.y_hat = np.empty((0, len(dataset_parameters['outputs'])))
        counter = 0
        X_r = []

        for i in range(X.shape[0]):
            X_r.append(X[i].reshape(1,self.params['seq_len'], len(dataset_parameters['inputs'])))

        for i in range(0, len(X_r)):
            if counter <= self.params['correction']:
                pred = self.model_test.predict(X_r[i], 
                                        batch_size=1, 
                                        verbose=0,
                                        ).reshape(-1, len(dataset_parameters['outputs']))
                
                self.y_hat = np.vstack((self.y_hat, pred))

            else:
                pred = pred.reshape(1, self.params['seq_len'], len(dataset_parameters['inputs']))
                pred = self.model_test.predict(pred, 
                                        batch_size=1, 
                                        verbose=0,
                                        ).reshape(-1, len(dataset_parameters['outputs']))
                
                self.y_hat = np.vstack((self.y_hat, pred))
            
            if self.params['forecast_size'] > 1:
                counter += self.params['seq_len']
                if counter >= self.params['forecast_size'] - (self.params['seq_len'] * self.params['correction']):
                    counter = 0
                


        if self.logging:
            print('Model tested.')

        # self.y_hat = self.scaler.inverse_transform(self.y_hat)
        if self.params['MISO']:
            self.y_hat = self.y_hat.flatten()
            self.y = self.y.flatten()
        else:
            # self.y_hat = self.y_hat.reshape(-1, self.y_hat.shape[2])
            self.y = self.y.reshape(-1, self.y.shape[2])

        resid_vector, residuals = self.metrics.compute_residuals(self.y, self.y_hat, metric)

        results = {
            'resid_vector': resid_vector,
            'residuals': residuals,
            'y': self.y,
            'y_hat': self.y_hat
        }
        
        return  results

    def predict(self, data, dataset_parameters):
        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        X, _ = self._prepare_data(data, dataset_parameters)

        if self.params['stateful']:
            maximum_sequences = X.shape[0] - X.shape[0] % \
                                        self.params['batch_size']
            
            X = X[:maximum_sequences]

        return self.model.predict(X, batch_size=self.params['batch_size'], verbose=0)
    # ---------------------------------------------------------------------    
    def save_model(self, model_path):
        if model_path is None:
            raise ValueError('model_path must be provided')
        
        self.model.save(model_path)
    # ---------------------------------------------------------------------
    def load_model(self, model_path):
        if model_path is None:
            raise ValueError('model_path must be provided')
        
        self.model = keras.models.load_model(model_path)
    # ---------------------------------------------------------------------
    def parameter_tuning(self, data, data_test, dataset_parameters, metric = 'mse'):

        if dataset_parameters is None:
            raise ValueError('dataset_parameters must be provided')
        
        if data is None:
            raise ValueError('data must be provided')
        
        # normalize
        self.scaler.fit(data)

        resid_vector = pd.DataFrame(columns = ['seq_len', 'hidden_size', 'batch_size', 'epochs', 'LR', 'resid'])

        for batch_size in [32, 64, 128]:
            X, y = self._prepare_data(data = data, dataset_parameters = dataset_parameters)

            size = int(X.shape[0])

            X_train, y_train = X[:size], y[:size]

            maximum_sequences = X_train.shape[0] - X_train.shape[0] % \
                                    batch_size
            
            X_train = X_train[:maximum_sequences]
            y_train = y_train[:maximum_sequences]

            X, y = self._prepare_data(data = data_test, dataset_parameters = dataset_parameters)

            X_test, y_test = X[:size], y[:size]

            X_test = X_test[:maximum_sequences]
            y_test = y_test[:maximum_sequences]

            if self.params['MISO']:
                y_test = y_test.flatten()
            else:
                y_test = y_test.reshape(-1, y_test.shape[2])

            for seq_len in [40, 50, 80, 100]:
                for hidden_size in [16, 32, 64, 128]:
                    for epochs in [50, 100, 150, 200]:
                        for LR in [0.01, 0.05, 0.1]:
                            models_parameters = {
                                    'model_type': 'LSTM',
                                    'seq_len': seq_len,
                                    'hidden_size': [hidden_size],
                                    'LR': LR,
                                    'batch_size': batch_size,
                                    'epochs': epochs,
                                    'train_ratio': 0.5,
                                    'optimizer': 'Adam',
                                    'loss': 'MSE',
                                    'metrics': ['accuracy'],
                                    'seed': 42,
                                    'TF': False,
                                    'stateful': True,
                                    'MISO': False,
                                    'forecast_size': 1,
                                    'rolling': False}
                            
                            self.params = {**self.params, **models_parameters}

                            model = self.create_model(dataset_parameters=dataset_parameters)

                            model.fit(
                                X_train, y_train,
                                epochs=self.params['epochs'],
                                batch_size=self.params['batch_size'],
                                verbose=0, 
                                shuffle=False
                            )
                            self.model = model

                            model_test = self.create_model(dataset_parameters=dataset_parameters, test = True)

                            y_hat = model_test.predict(X_test, batch_size=1, verbose=0)

                            if self.params['MISO']:
                                y_hat = y_hat.flatten()
                            else:
                                y_hat = y_hat.reshape(-1, y_hat.shape[2])

                            _, residuals = self.metrics.compute_residuals(y_test, y_hat, metric)
                            
                            row = {
                                'seq_len': seq_len,
                                'hidden_size': hidden_size,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'LR': LR,
                                'resid': residuals
                            }
                            row_df = pd.DataFrame([row])

                            resid_vector = pd.concat([resid_vector, row_df], ignore_index=True)

                            print(f'seq_len: {seq_len},hidden_size: {hidden_size},batch_size: {batch_size}, epochs: {epochs}, LR: {LR}, resid: {residuals}')
                            
        self.opt_results = resid_vector

        return resid_vector
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def save_scaler(self, name):
        if self.scaler is not None:
            DataHandler.save_pickle(self.scaler, name)

    def load_scaler(self, name):
        self.scaler = DataHandler.load_pickle(name)

    def create_scaler(self, data, all_data, scale_per_rho = False, mode = 2):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        if self.scaler is None:
            if mode == 1:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            elif mode == 2:
                self.scaler = StandardScaler()
            #self.scaler = RobustScaler()
            #self.scaler = RobustScaler()

        if scale_per_rho:
            
            if all_data is None:
                raise ValueError('all_data must be provided when SCALE_PER_RHO is True')
            
            scalers = {}
            for rho in np.unique(all_data[:, 6]):
                if mode == 1:
                    scalers[rho] = MinMaxScaler(feature_range=(0, 1))
                elif mode == 2:
                    scalers[rho] = StandardScaler()
                scalers[rho].fit(all_data[all_data[:, 6] == rho])

            self.multi_scaler = scalers
            
        else:
            # normalize
            if all_data is not None:
                self.scaler.fit(all_data)
            else:
                self.scaler.fit(data)

    def var_loss(self, a, b):
        return tf.math.reduce_variance(a - b)
    
    def bsc_loss(self, a, b, small_constant=1e-8):
        # Balanced Spectral Power Loss
        # N-D Spatial Fourier Transform
        fa = tf.signal.fft3d(tf.cast(a.flatten(), tf.complex64))
        fb = tf.signal.fft3d(tf.cast(b.flatten(), tf.complex64))

        # Energy per mode(c,k)
        Ea = tf.math.reduce_mean(tf.math.square(tf.abs(fa)), axis=-1)
        Eb = tf.math.reduce_mean(tf.math.square(tf.abs(fb)), axis=-1)

        # Wavenumber magnitude
        k = tf.sqrt(tf.cast(tf.range(tf.shape(Ea)[-1]), tf.float32))

        # Binning, 
        def binned_energy(E, k, num_bins=10):
            bin_edges = tf.linspace(tf.reduce_min(k), tf.reduce_max(k), num_bins + 1)
            binned_E = []
            for i in range(num_bins):
                mask = (k >= bin_edges[i]) & (k < bin_edges[i + 1])
                bin_mean = tf.reduce_mean(tf.boolean_mask(E, mask))
                binned_E.append(bin_mean)
            return tf.stack(binned_E)
        
        Ea_binned = binned_energy(Ea, k)    
        Eb_binned = binned_energy(Eb, k)    

        loss = tf.reduce_mean(tf.square(1 - (Ea_binned + small_constant) / (Eb_binned + small_constant)))

        return loss

    def normed_ld(self, a, b):
        return tf.abs(tf.norm(a, axis=-1) - tf.norm(b, axis=-1)) / (tf.norm(a, axis=-1) + tf.norm(b, axis=-1) + 1e-8)

    def cosine_similarity(self, a, b):
        a_norm = tf.norm(a, axis=-1)
        b_norm = tf.norm(b, axis=-1)
        return tf.reduce_sum(a * b, axis=-1) / (a_norm * b_norm + 1e-8)

    def cosine_distance(self, a, b):
        return (1 - self.cosine_similarity(a, b)) / 2

    def polar_loss(self, w=[0.5, 0.5]):
        def loss(y_true, y_pred):
            return w[0] * tf.reduce_mean(self.normed_ld(y_true, y_pred)) + \
                w[1] * tf.reduce_mean(self.cosine_distance(y_true, y_pred))
        return loss
    
    def normed_ld2(self, a, b):
        norm_a = K.l2_norm(a)
        norm_b = K.l2_norm(b)
        
        # Calculate abs(|a| - |b|) / (|a| + |b|)
        numerator = K.abs(norm_a - norm_b)
        denominator = norm_a + norm_b
        
        return numerator / (denominator + K.epsilon())

    def cosine_similarity2(self, a, b):
        a_normalized = K.l2_normalize(a, axis=-1)
        b_normalized = K.l2_normalize(b, axis=-1)

        return K.sum(a_normalized * b_normalized, axis=-1)

    def cosine_distance2(self, a, b):

        similarity = self.cosine_similarity2(a, b)

        return (1.0 - similarity) / 2.0

    def polar_loss_factory(self, w = [0.5, 0.5]):
        w1 = w[0]
        w2 = w[1]
        
        def polar_loss(y_true, y_pred):
            loss_ld_per_sample = self.normed_ld2(y_true, y_pred)
            loss_cos_per_sample = self.cosine_distance2(y_true, y_pred)
            combined_loss_per_sample = w1 * loss_ld_per_sample + w2 * loss_cos_per_sample
            final_loss = K.mean(combined_loss_per_sample) 
            
             # final_loss = K.sum(combined_loss_per_sample)
            return final_loss

        return polar_loss

class BSP1DLoss(tf.keras.losses.Loss):
    def __init__(self, num_bins=10, eps=1e-8, linear_bins=True, name="bsp_1d_loss"):
        super().__init__(name=name)
        self.num_bins = num_bins
        self.eps = eps
        self.linear_bins = linear_bins

    def call(self, y_true, y_pred):
        # Expected shape: (batch, timesteps, channels)
        fft_true = tf.signal.fft(tf.cast(y_true, tf.complex64))
        fft_pred = tf.signal.fft(tf.cast(y_pred, tf.complex64))
        E_true = tf.abs(fft_true) ** 2
        E_pred = tf.abs(fft_pred) ** 2

        # Compute wavenumber magnitudes
        N = tf.shape(y_true)[1]
        k = tf.range(-N // 2, N // 2)
        k = tf.signal.fftshift(tf.cast(tf.abs(k), tf.float32))
        k_max = tf.reduce_max(k)

        # Define bins (linear or log)
        if self.linear_bins:
            bins = tf.linspace(0.0, k_max, self.num_bins + 1)
        else:
            bins = tf.exp(tf.linspace(0.0, tf.math.log(k_max + 1.0), self.num_bins + 1)) - 1.0

        # Bin-wise energy ratios
        losses = []
        for i in range(self.num_bins):
            mask = (k >= bins[i]) & (k < bins[i + 1])
            mask = tf.cast(mask, tf.float32)
            n_bin = tf.reduce_sum(mask) + self.eps

            E_true_bin = tf.reduce_sum(E_true * mask[None, :, None]) / n_bin
            E_pred_bin = tf.reduce_sum(E_pred * mask[None, :, None]) / n_bin

            loss_bin = (1.0 - E_true_bin / (E_pred_bin + self.eps)) ** 2
            losses.append(loss_bin)

        L_spec = tf.reduce_mean(tf.stack(losses))
        return L_spec
