import tensorflow as tf

# 그래프 모드 사용 (이미 tf_base_model 에서도 disable_eager_execution 호출됨)
tf.compat.v1.disable_eager_execution()

# 우리가 정의한 state 튜플
from rnn_cell import LSTMAttentionCellState


def rnn_free_run(cell,
                 initial_state,
                 sequence_length,
                 initial_input=None,
                 scope='dynamic-rnn-free-run'):
    """
    Keras 3 / TF2.16+ 호환 rnn_free_run 구현.

    - cell: LSTMAttentionCell (tf.keras.layers.AbstractRNNCell 상속)
    - initial_state: LSTMAttentionCellState (namedtuple)
    - sequence_length: int scalar (최대 타임스텝 수)
    - initial_input: [batch_size, input_dim] or None
        None 이면 cell.output_function(initial_state) 로 한 스텝 생성

    반환값: (states, outputs, final_state)
        - states: 현재 구현에서는 사용되지 않으므로 None
        - outputs: [batch_size, sequence_length, 3] (각 step의 (Δx, Δy, e))
        - final_state: 마지막 time-step 의 LSTMAttentionCellState
    """

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # 초기 input이 없으면 state 에서 한 번 뽑아서 시작
        if initial_input is None:
            initial_input = cell.output_function(initial_state)  # [B, 3]

        # batch size 추론
        batch_size = tf.shape(initial_input)[0]

        # while_loop 초기값들
        time0 = tf.constant(0, dtype=tf.int32)
        finished0 = tf.zeros([batch_size], dtype=tf.bool)  # [B]
        state0_list = list(initial_state)                  # while_loop에서는 list 로 다룸
        input0 = initial_input                             # [B, 3]
        ta0 = tf.TensorArray(dtype=initial_input.dtype,
                             size=sequence_length,
                             clear_after_read=False)

        zero_input = tf.zeros_like(initial_input)

        def cond(t, state_list, inp, finished, ta):
            # 모든 시퀀스가 종료되면 loop 중단
            return tf.logical_not(tf.reduce_all(finished))

        def body(t, state_list, inp, finished, ta):
            """
            한 타임스텝:
              1) cell(inp, state_list) 로 새로운 state 계산
              2) termination_condition 으로 각 샘플별 finished 여부 판단
              3) output_function(state_new) 으로 다음 입력 생성
              4) outputs TensorArray 에 기록
            """
            # Keras RNNCell 호출: (output_t, new_state_list)
            output_t, new_state_list = cell(inp, state_list)
            state_new = LSTMAttentionCellState(*new_state_list)

            # elements_finished: [B] (이 타임스텝 기준 종료 여부)
            elements_finished = tf.logical_or(
                t >= sequence_length,
                cell.termination_condition(state_new)
            )
            # 누적 finished
            finished_new = tf.logical_or(finished, elements_finished)

            # 다음 입력: 전부 끝났으면 0, 아니면 새로 샘플링
            next_inp = tf.cond(
                tf.reduce_all(finished_new),
                lambda: zero_input,
                lambda: cell.output_function(state_new)
            )

            # 현재 step 의 출력은 next_inp 로 기록 (샘플링된 stroke)
            ta = ta.write(t, next_inp)

            return t + 1, new_state_list, next_inp, finished_new, ta

        # while_loop 실행
        _, final_state_list, _, _, ta_final = tf.compat.v1.while_loop(
            cond,
            body,
            loop_vars=[time0, state0_list, input0, finished0, ta0],
            parallel_iterations=32,
            swap_memory=False
        )

        # outputs: [time, batch, 3] → [batch, time, 3]
        outputs_time_major = ta_final.stack()
        outputs = tf.transpose(outputs_time_major, [1, 0, 2])

        final_state = LSTMAttentionCellState(*final_state_list)

        # 이전 raw_rnn 과 인터페이스를 맞추기 위해 (states, outputs, final_state) 3-tuple 반환
        # 여기서는 states 전체 시퀀스는 사용하지 않으므로 None 으로 둔다.
        states = None
        return states, outputs, final_state


# rnn_teacher_force / raw_rnn 는 현재 코드 경로에서 사용되지 않으므로 제거했음.
# 필요해지면 Keras RNN 또는 위 패턴(tf.while_loop + cell 호출)을 이용해 별도로 다시 구현하는 것이 안전함.
