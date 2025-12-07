from collections import namedtuple

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tf_utils import shape  # dense_layer 대신 Keras Dense를 직접 사용하므로 shape만 사용

tfd = tfp.distributions


LSTMAttentionCellState = namedtuple(
    'LSTMAttentionCellState',
    ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi']
)


class LSTMAttentionCell(tf.keras.layers.Layer):
    """
    Keras 3 / TF 2.16+ 호환 버전의 LSTMAttentionCell
    - tf.compat.v1.nn.rnn_cell.RNNCell 상속 → tf.keras.layers.AbstractRNNCell 상속
    - tf.compat.v1.nn.rnn_cell.LSTMCell → tf.keras.layers.LSTMCell (Keras)
    """

    def __init__(
        self,
        lstm_size,
        num_attn_mixture_components,
        attention_values,
        attention_values_lengths,
        num_output_mixture_components,
        bias,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values = attention_values  # [B, T_char, window_size]
        self.attention_values_lengths = attention_values_lengths  # [B]
        # window_size 는 static shape를 사용하는 게 Keras state_size에 더 안전함
        self.window_size = shape(self.attention_values, 2)  # int
        self.char_len = shape(self.attention_values, 1)     # int (문자 길이)
        self.num_output_mixture_components = num_output_mixture_components
        self.output_units = 6 * self.num_output_mixture_components + 1
        self.bias = bias

        # 하위 LSTM 셀과 Dense 레이어는 build() 에서 생성
        self.lstm_cell1 = None
        self.lstm_cell2 = None
        self.lstm_cell3 = None
        self.attention_dense = None
        self.gmm_dense = None

    # ----- 필수 property -----

    @property
    def state_size(self):
        """
        Keras RNN이 state 크기를 알 수 있게 숫자/shape 리스트를 리턴해야 함.
        여기서는 각 상태가 [batch, dim] 형태이므로 dim만 넣어줌.
        순서는 LSTMAttentionCellState 와 같게 맞춰둠.
        """
        return [
            self.lstm_size,  # h1
            self.lstm_size,  # c1
            self.lstm_size,  # h2
            self.lstm_size,  # c2
            self.lstm_size,  # h3
            self.lstm_size,  # c3
            self.num_attn_mixture_components,  # alpha
            self.num_attn_mixture_components,  # beta
            self.num_attn_mixture_components,  # kappa
            self.window_size,                  # w
            self.char_len,                     # phi
        ]

    @property
    def output_size(self):
        # 셀의 출력은 s3_out (크기 = lstm_size)
        return self.lstm_size

    # ----- build: 가중치/레이어 정의 -----

    def build(self, input_shape):
        # 세 개의 LSTMCell (Keras 버전)
        self.lstm_cell1 = tf.keras.layers.LSTMCell(self.lstm_size)
        self.lstm_cell2 = tf.keras.layers.LSTMCell(self.lstm_size)
        self.lstm_cell3 = tf.keras.layers.LSTMCell(self.lstm_size)

        # Attention 파라미터용 Dense: alpha, beta, kappa
        self.attention_dense = tf.keras.layers.Dense(
            3 * self.num_attn_mixture_components,
            name="attention_dense"
        )

        # GMM 파라미터용 Dense
        self.gmm_dense = tf.keras.layers.Dense(
            self.output_units,
            name="gmm_dense"
        )

        super().build(input_shape)

    # ----- core: 한 time-step 연산 -----

    def call(self, inputs, states, training=None):
        """
        inputs: [batch, input_dim]
        states: 리스트 형태의 previous states (state_size 순서대로 들어옴)
        return: (output, new_states_list)
        """
        # Keras는 states를 리스트로 넘겨줌 → namedtuple로 변환해서 기존 코드 재사용
        state = LSTMAttentionCellState(*states)

        # 배치 크기는 동적으로 계산
        batch_size = tf.shape(inputs)[0]

        # ----- LSTM 1 -----
        s1_in = tf.concat([state.w, inputs], axis=1)
        # Keras LSTMCell 은 (outputs, [h, c]) 를 리턴
        s1_out, [h1_new, c1_new] = self.lstm_cell1(s1_in, [state.h1, state.c1])

        # ----- Attention -----
        attention_inputs = tf.concat([state.w, inputs, s1_out], axis=1)
        attention_params = self.attention_dense(attention_inputs)  # [B, 3*M]
        alpha, beta, kappa = tf.split(
            tf.nn.softplus(attention_params),
            3,
            axis=1
        )  # 각 [B, M]

        kappa = state.kappa + kappa / 25.0
        beta = tf.clip_by_value(beta, 0.01, np.inf)

        kappa_flat, alpha_flat, beta_flat = kappa, alpha, beta

        # [B, M, 1]
        kappa_exp = tf.expand_dims(kappa, 2)
        alpha_exp = tf.expand_dims(alpha, 2)
        beta_exp = tf.expand_dims(beta, 2)

        # u: [1, 1, char_len] → [B, M, char_len]
        enum = tf.reshape(tf.range(self.char_len), (1, 1, self.char_len))
        u = tf.cast(tf.tile(enum, (batch_size, self.num_attn_mixture_components, 1)), tf.float32)

        phi_flat = tf.reduce_sum(alpha_exp * tf.exp(-tf.square(kappa_exp - u) / beta_exp), axis=1)  # [B, char_len]

        phi = tf.expand_dims(phi_flat, 2)  # [B, char_len, 1]

        # sequence mask: [B, char_len, 1]
        sequence_mask = tf.cast(
            tf.sequence_mask(self.attention_values_lengths, maxlen=self.char_len),
            tf.float32
        )
        sequence_mask = tf.expand_dims(sequence_mask, 2)

        # attention_values: [B, char_len, window_size]
        # w: [B, window_size]
        w = tf.reduce_sum(phi * self.attention_values * sequence_mask, axis=1)

        # ----- LSTM 2 -----
        s2_in = tf.concat([inputs, s1_out, w], axis=1)
        s2_out, [h2_new, c2_new] = self.lstm_cell2(s2_in, [state.h2, state.c2])

        # ----- LSTM 3 -----
        s3_in = tf.concat([inputs, s2_out, w], axis=1)
        s3_out, [h3_new, c3_new] = self.lstm_cell3(s3_in, [state.h3, state.c3])

        # 새 state(namedtuple)
        new_state = LSTMAttentionCellState(
            h1_new,
            c1_new,
            h2_new,
            c2_new,
            h3_new,
            c3_new,
            alpha_flat,
            beta_flat,
            kappa_flat,
            w,
            phi_flat,
        )

        # Keras는 리스트 형태의 states를 기대함
        return s3_out, list(new_state)

    # ----- zero_state: 필요하면 직접 사용 (Keras RNN의 initial_state로도 사용 가능) -----

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Keras RNN에서 initial_state를 자동으로 만들 때 사용.
        """
        if batch_size is None:
            # inputs가 있을 경우 거기서 추론
            if inputs is not None:
                batch_size = tf.shape(inputs)[0]
            else:
                raise ValueError("batch_size나 inputs 중 하나는 제공되어야 합니다.")

        if dtype is None:
            dtype = tf.float32

        return [
            tf.zeros([batch_size, self.lstm_size], dtype=dtype),  # h1
            tf.zeros([batch_size, self.lstm_size], dtype=dtype),  # c1
            tf.zeros([batch_size, self.lstm_size], dtype=dtype),  # h2
            tf.zeros([batch_size, self.lstm_size], dtype=dtype),  # c2
            tf.zeros([batch_size, self.lstm_size], dtype=dtype),  # h3
            tf.zeros([batch_size, self.lstm_size], dtype=dtype),  # c3
            tf.zeros([batch_size, self.num_attn_mixture_components], dtype=dtype),  # alpha
            tf.zeros([batch_size, self.num_attn_mixture_components], dtype=dtype),  # beta
            tf.zeros([batch_size, self.num_attn_mixture_components], dtype=dtype),  # kappa
            tf.zeros([batch_size, self.window_size], dtype=dtype),  # w
            tf.zeros([batch_size, self.char_len], dtype=dtype),     # phi
        ]

    # ----- 이하 함수들은 로직 그대로 사용 (batch_size 부분만 동적으로 수정) -----

    def output_function(self, state: LSTMAttentionCellState):
        """
        한 state에서 실제 (Δx, Δy, e) 를 샘플링하는 부분.
        Keras와 무관한 순수 TF 로직이므로 거의 그대로 사용 가능.
        """
        # state.h3: [B, lstm_size]
        params = self.gmm_dense(state.h3)  # [B, output_units]
        pis, mus, sigmas, rhos, es = self._parse_parameters(params)

        mu1, mu2 = tf.split(mus, 2, axis=1)
        mus = tf.stack([mu1, mu2], axis=2)  # [B, M, 2]
        sigma1, sigma2 = tf.split(sigmas, 2, axis=1)

        covar_matrix = [
            tf.square(sigma1),          # [B, M]
            rhos * sigma1 * sigma2,     # [B, M]
            rhos * sigma1 * sigma2,     # [B, M]
            tf.square(sigma2)           # [B, M]
        ]
        covar_matrix = tf.stack(covar_matrix, axis=2)  # [B, M, 4]

        batch_size = tf.shape(mus)[0]

        covar_matrix = tf.reshape(
            covar_matrix,
            (batch_size, self.num_output_mixture_components, 2, 2)
        )

        mvn = tfd.MultivariateNormalFullCovariance(
            loc=mus,
            covariance_matrix=covar_matrix
        )
        b = tfd.Bernoulli(probs=es)
        c = tfd.Categorical(probs=pis)

        sampled_e = b.sample()          # [B, 1]
        sampled_coords = mvn.sample()   # [B, M, 2]
        sampled_idx = c.sample()        # [B]

        idx = tf.stack([tf.range(batch_size), sampled_idx], axis=1)
        coords = tf.gather_nd(sampled_coords, idx)  # [B, 2]

        return tf.concat([coords, tf.cast(sampled_e, tf.float32)], axis=1)  # [B, 3]

    def termination_condition(self, state: LSTMAttentionCellState):
        # state.phi: [B, char_len]
        char_idx = tf.cast(tf.argmax(state.phi, axis=1), tf.int32)  # [B]
        final_char = char_idx >= self.attention_values_lengths - 1
        past_final_char = char_idx >= self.attention_values_lengths

        output = self.output_function(state)
        es = tf.cast(output[:, 2], tf.int32)
        is_eos = tf.equal(es, np.ones_like(es))

        return tf.logical_or(tf.logical_and(final_char, is_eos), past_final_char)

    def _parse_parameters(self, gmm_params, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            gmm_params,
            [
                1 * self.num_output_mixture_components,  # pis
                2 * self.num_output_mixture_components,  # sigmas
                1 * self.num_output_mixture_components,  # rhos
                2 * self.num_output_mixture_components,  # mus
                1                                       # es
            ],
            axis=-1
        )

        pis = pis * (1 + tf.expand_dims(self.bias, 1))
        sigmas = sigmas - tf.expand_dims(self.bias, 1)

        pis = tf.nn.softmax(pis, axis=-1)
        pis = tf.where(pis < 0.01, tf.zeros_like(pis), pis)

        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        es = tf.where(es < 0.01, tf.zeros_like(es), es)

        return pis, mus, sigmas, rhos, es
