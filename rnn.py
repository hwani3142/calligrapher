from __future__ import print_function
import os

import numpy as np
import tensorflow as tf

import drawing
from data_frame import DataFrame
# LSTMAttentionCellState 도 가져오면 편함 (필수는 아님)
from rnn_cell import LSTMAttentionCell, LSTMAttentionCellState
from rnn_ops import rnn_free_run
from tf_base_model import TFBaseModel
from tf_utils import time_distributed_dense_layer


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = ['x', 'x_len', 'c', 'c_len']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95, random_state=2018)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            mode='test'
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, mode='train'):
        gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )
        for batch in gen:
            batch['x_len'] = batch['x_len'] - 1
            max_x_len = np.max(batch['x_len'])
            max_c_len = np.max(batch['c_len'])
            batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
            batch['x'] = batch['x'][:, :max_x_len, :]
            batch['c'] = batch['c'][:, :max_c_len]
            yield batch


class rnn(TFBaseModel):

    def __init__(
        self,
        lstm_size,
        output_mixture_components,
        attention_mixture_components,
        **kwargs
    ):
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.output_units = self.output_mixture_components * 6 + 1
        self.attention_mixture_components = attention_mixture_components
        super(rnn, self).__init__(**kwargs)

    def parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            z,
            [
                1 * self.output_mixture_components,
                2 * self.output_mixture_components,
                1 * self.output_mixture_components,
                2 * self.output_mixture_components,
                1
            ],
            axis=-1
        )
        pis = tf.nn.softmax(pis, axis=-1)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        return pis, mus, sigmas, rhos, es

    def NLL(self, y, lengths, pis, mus, sigmas, rho, es, eps=1e-8):
        sigma_1, sigma_2 = tf.split(sigmas, 2, axis=2)
        y_1, y_2, y_3 = tf.split(y, 3, axis=2)
        mu_1, mu_2 = tf.split(mus, 2, axis=2)

        norm = 1.0 / (2 * np.pi * sigma_1 * sigma_2 * tf.sqrt(1 - tf.square(rho)))
        Z = tf.square((y_1 - mu_1) / (sigma_1)) + \
            tf.square((y_2 - mu_2) / (sigma_2)) - \
            2 * rho * (y_1 - mu_1) * (y_2 - mu_2) / (sigma_1 * sigma_2)

        exp = -1.0 * Z / (2 * (1 - tf.square(rho)))
        gaussian_likelihoods = tf.exp(exp) * norm
        gmm_likelihood = tf.reduce_sum(pis * gaussian_likelihoods, 2)
        gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

        bernoulli_likelihood = tf.squeeze(tf.where(tf.equal(tf.ones_like(y_3), y_3), es, 1 - es))

        nll = -(tf.math.log(gmm_likelihood) + tf.math.log(bernoulli_likelihood))
        sequence_mask = tf.logical_and(
            tf.sequence_mask(lengths, maxlen=tf.shape(y)[1]),
            tf.logical_not(tf.math.is_nan(nll)),
        )
        nll = tf.where(sequence_mask, nll, tf.zeros_like(nll))
        num_valid = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1)

        sequence_loss = tf.reduce_sum(nll, axis=1) / tf.maximum(num_valid, 1.0)
        element_loss = tf.reduce_sum(nll) / tf.maximum(tf.reduce_sum(num_valid), 1.0)
        return sequence_loss, element_loss

    def sample(self, cell):
        # rnn_free_run 이 cell.zero_state 를 기대하므로 그대로 사용
        initial_state = cell.zero_state(self.num_samples, dtype=tf.float32)
        initial_input = tf.concat([
            tf.zeros([self.num_samples, 2]),
            tf.ones([self.num_samples, 1]),
        ], axis=1)
        return rnn_free_run(
            cell=cell,
            sequence_length=self.sample_tsteps,
            initial_state=initial_state,
            initial_input=initial_input,
            scope='rnn'
        )[1]

    def primed_sample(self, cell):
        """
        기존에는 tf.nn.dynamic_rnn 으로 prime 시퀀스를 흘려보냈는데,
        이제는 tf.keras.layers.RNN(cell, ...) 을 사용.
        """
        # 샘플 개수 기준으로 initial state 생성
        initial_state = cell.zero_state(self.num_samples, dtype=tf.float32)
        initial_state_list = list(initial_state)

        # prime 입력에 대한 mask (길이: x_prime_len)
        prime_mask = tf.sequence_mask(self.x_prime_len, maxlen=tf.shape(self.x_prime)[1])

        # return_state=True 로 마지막 state 를 받아서 사용
        prime_rnn_layer = tf.keras.layers.RNN(
            cell,
            return_sequences=False,
            return_state=True,
            name="rnn_prime"
        )
        prime_outputs_and_states = prime_rnn_layer(
            self.x_prime,
            mask=prime_mask,
            initial_state=initial_state_list
        )
        # prime_outputs_and_states[0] 은 마지막 output, 이후는 state 리스트
        primed_state_list = prime_outputs_and_states[1:]
        primed_state = LSTMAttentionCellState(*primed_state_list)

        return rnn_free_run(
            cell=cell,
            sequence_length=self.sample_tsteps,
            initial_state=primed_state,
            scope='rnn'
        )[1]

    def calculate_loss(self):
        # 여기 부분은 여전히 TF1 스타일 placeholder,
        # TFBaseModel 쪽에서 tf.compat.v1.disable_eager_execution() 을 해준다고 가정.
        # self.x = tf.placeholder(tf.float32, [None, None, 3])
        # self.y = tf.placeholder(tf.float32, [None, None, 3])
        # self.x_len = tf.placeholder(tf.int32, [None])
        # self.c = tf.placeholder(tf.int32, [None, None])
        # self.c_len = tf.placeholder(tf.int32, [None])

        # self.sample_tsteps = tf.placeholder(tf.int32, [])
        # self.num_samples = tf.placeholder(tf.int32, [])
        # self.prime = tf.placeholder(tf.bool, [])
        # self.x_prime = tf.placeholder(tf.float32, [None, None, 3])
        # self.x_prime_len = tf.placeholder(tf.int32, [None])
        # self.bias = tf.placeholder_with_default(
        #     tf.zeros([self.num_samples], dtype=tf.float32), [None])
        
        self.x = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
        self.x_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.c = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.c_len = tf.compat.v1.placeholder(tf.int32, [None])

        self.sample_tsteps = tf.compat.v1.placeholder(tf.int32, [])
        self.num_samples = tf.compat.v1.placeholder(tf.int32, [])
        self.prime = tf.compat.v1.placeholder(tf.bool, [])
        self.x_prime = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
        self.x_prime_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.bias = tf.compat.v1.placeholder_with_default(
            tf.zeros([self.num_samples], dtype=tf.float32), [None]
        )

        # Keras 3 호환 LSTMAttentionCell 사용
        cell = LSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            attention_values=tf.one_hot(self.c, len(drawing.alphabet)),
            attention_values_lengths=self.c_len,
            num_output_mixture_components=self.output_mixture_components,
            bias=self.bias
        )

        # initial_state: batch_size x state_dims
        batch_size = tf.shape(self.x)[0]
        self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        initial_state_list = list(self.initial_state)

        # 길이 정보로 mask 생성 (True = 유효 time-step)
        rnn_mask = tf.sequence_mask(self.x_len, maxlen=tf.shape(self.x)[1])

        # tf.nn.dynamic_rnn 대신 Keras RNN
        rnn_layer = tf.keras.layers.RNN(
            cell,
            return_sequences=True,
            return_state=True,
            name="rnn"
        )
        outputs_and_states = rnn_layer(
            self.x,
            mask=rnn_mask,
            initial_state=initial_state_list
        )
        outputs = outputs_and_states[0]
        final_state_list = outputs_and_states[1:]
        self.final_state = LSTMAttentionCellState(*final_state_list)

        # GMM 파라미터 추출
        params = time_distributed_dense_layer(outputs, self.output_units, scope='rnn/gmm')
        pis, mus, sigmas, rhos, es = self.parse_parameters(params)
        sequence_loss, self.loss = self.NLL(self.y, self.x_len, pis, mus, sigmas, rhos, es)

        # 샘플링: prime 여부에 따라 분기
        self.sampled_sequence = tf.cond(
            self.prime,
            lambda: self.primed_sample(cell),
            lambda: self.sample(cell)
        )
        return self.loss


if __name__ == '__main__':
    dr = DataReader(data_dir='data/processed/')

    nn = rnn(
        reader=dr,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions',
        learning_rates=[.0001, .00005, .00002],
        batch_sizes=[32, 64, 64],
        patiences=[1500, 1000, 500],
        beta1_decays=[.9, .9, .9],
        validation_batch_size=32,
        optimizer='rms',
        num_training_steps=100000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=2000,
        log_interval=20,
        grad_clip=10,
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10
    )
    nn.fit()
