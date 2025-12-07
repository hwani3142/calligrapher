import tensorflow as tf
from rnn_cell import LSTMAttentionCellState

tf.compat.v1.disable_eager_execution()


def rnn_free_run(cell,
                 initial_state,
                 sequence_length,
                 initial_input=None,
                 scope='dynamic-rnn-free-run'):
    """
    Keras 3 / TF2.16+ 호환 rnn_free_run 구현.
    """
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        if initial_input is None:
            initial_input = cell.output_function(initial_state)  # [B, 3]

        batch_size = tf.shape(initial_input)[0]

        time0 = tf.constant(0, dtype=tf.int32)
        finished0 = tf.zeros([batch_size], dtype=tf.bool)
        state0_list = list(initial_state)
        input0 = initial_input

        # ★ size=sequence_length, 인덱스는 0..sequence_length-1 만 유효
        ta0 = tf.TensorArray(
            dtype=initial_input.dtype,
            size=sequence_length,
            clear_after_read=False
        )

        zero_input = tf.zeros_like(initial_input)

        def cond(t, state_list, inp, finished, ta):
            # ★ 길이 제한 + 아직 전부 안 끝났을 때만 계속
            return tf.logical_and(
                tf.less(t, sequence_length),
                tf.logical_not(tf.reduce_all(finished))
            )

        def body(t, state_list, inp, finished, ta):
            output_t, new_state_list = cell(inp, state_list)
            state_new = LSTMAttentionCellState(*new_state_list)

            # 종료 조건: 셀에서 정의한 termination_condition
            elements_finished = cell.termination_condition(state_new)
            finished_new = tf.logical_or(finished, elements_finished)

            # 모두 끝났으면 0, 아니면 다음 step 입력 생성
            next_inp = tf.cond(
                tf.reduce_all(finished_new),
                lambda: zero_input,
                lambda: cell.output_function(state_new)
            )

            # 현재 step 의 샘플을 기록 (index = t, t < sequence_length 보장됨)
            ta = ta.write(t, next_inp)

            return t + 1, new_state_list, next_inp, finished_new, ta

        _, final_state_list, _, _, ta_final = tf.compat.v1.while_loop(
            cond,
            body,
            loop_vars=[time0, state0_list, input0, finished0, ta0],
            parallel_iterations=32,
            swap_memory=False
        )

        outputs_time_major = ta_final.stack()          # [T, B, 3] (T <= sequence_length)
        outputs = tf.transpose(outputs_time_major, [1, 0, 2])  # [B, T, 3]

        final_state = LSTMAttentionCellState(*final_state_list)

        # states 전체는 지금 안 쓰니까 None
        states = None
        return states, outputs, final_state
