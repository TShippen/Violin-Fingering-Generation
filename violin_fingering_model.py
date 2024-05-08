import os
import numpy as np
import random
import itertools
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import pretty_midi
import time
from tensorflow import LSTMBlockCell

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class violin_fingering_model(object):
    def __init__(self):
        self.pitch_for_invalid_note = 101
        self.lowest_pitch = 55
        self.n_p_classes = (
            46 + 1
        )  # number of pitch classes, pitch range =  55 to 100, pitch_for_invalid_note = 101
        self.n_b_classes = 7  # number of beat_type classes, {'', '1th', '2th', '4th',  '8th',  '16th', '32th'}

        # Existing position and fingering representations:
        # n_str_classes: number of string classes, {'': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4}
        # n_pos_classes: number of position classes, {'': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12}
        # n_fin_classes: number of finger classes, {'': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        # The model currently considers positions up to the 12th position and fingers up to the 4th finger (pinky).

        self.n_str_classes = (
            5  # number of string classes, {'': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4}
        )
        self.n_pos_classes = 15  # number of position classes, {'': 0, '1': 1, '2': 2, ..., '13': 13, '14': 14}
        self.n_fin_classes = 6  # number of finger classes, {'': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'T': 5}

        self.n_steps = 32  # number of note events for an input sequence
        self.embedding_size = 100
        self.hidden_size = 100
        self.batch_size = 26
        self.INFIN = 1e12  # infinite number
        self.n_in_succession = 10
        self.initial_learning_rate = 1e-3
        self.n_epochs = 100
        self.drop = 0.3
        self.dataset_dir = "./TNUA_violin_fingering_dataset"
        self.save_dir = "./model"
        self.model_dir = "./model/violin_fingering_estimation.ckpt"
        self.context_window = 5

    def load_data(self):
        print("Load training data...")
        files = [x for x in os.listdir(self.dataset_dir) if x.endswith("csv")]
        corpus = {}
        for file in files:
            with open(self.dataset_dir + "/" + file) as f:
                data = np.genfromtxt(
                    f,
                    delimiter=",",
                    names=True,
                    dtype=[
                        ("int"),
                        ("float"),
                        ("float"),
                        ("int"),
                        ("int"),
                        ("int"),
                        ("int"),
                    ],
                )
                grouped_data = []
                current_group = []
                current_end_time = 0

                for row in data:
                    start_time = row["start"]
                    end_time = start_time + row["duration"]

                    if not current_group or start_time < current_end_time:
                        current_group.append(row)
                        current_end_time = max(current_end_time, end_time)
                    else:
                        grouped_data.append(current_group)
                        current_group = [row]
                        current_end_time = end_time

                if current_group:
                    grouped_data.append(current_group)

                corpus[file] = grouped_data
        return corpus

    def segment_corpus(self, corpus, context_window=5):
        def _segment_sequence(sequence, context_measures):
            # Calculate the number of measures in the sequence
            num_measures = int(np.ceil(sequence[-1][-1]["start"] / 4))  # Assuming 4/4 time signature

            # Pad the sequence with empty measures at the beginning and end to handle context measures
            padding_start = [[] for _ in range(context_measures)]
            padding_end = [[] for _ in range(num_measures + context_measures - len(sequence))]
            padded_sequence = padding_start + sequence + padding_end

            segments = []
            valid_lens = []
            for i in range(context_measures, len(padded_sequence) - context_measures):
                segment_measures = padded_sequence[i - context_measures : i + context_measures + 1]
                segment = [note for measure in segment_measures for note in measure]
                segments.append(segment)
                valid_lens.append(len(padded_sequence[i]))

            # Pad the segments to a fixed length
            max_segment_len = max(len(segment) for segment in segments)
            padded_segments = [
                segment + [(self.pitch_for_invalid_note, -1, 0, 0, 0, 0, 0)] * (max_segment_len - len(segment))
                for segment in segments
            ]

            return np.array(padded_segments), np.array(valid_lens)

        corpus_seg = {}  # {key: {segments: 3d_array, lens: len_list}}
        for key, sequence in corpus.items():
            corpus_seg[key] = {}
            segments, valid_lens = self._segment_sequence(sequence, context_measures)
            corpus_seg[key]["segments"] = segments
            corpus_seg[key]["lens"] = valid_lens

        print(
            "total number of segments =",
            sum([v["segments"].shape[0] for v in corpus_seg.values()]),
        )
        return corpus_seg

    def create_training_and_testing_sets(self, corpus):
        corpus_vio1 = {k: v for k, v in corpus.items() if "vio1_" in k}  # only use vio1

        training_key_list = [
            key
            for key in corpus_vio1.keys()
            if any(
                x in key for x in ["bach", "mozart", "beeth", "mend", "flower", "wind"]
            )
        ]
        training_data = [v for k, v in corpus_vio1.items() if k in training_key_list]
        testing_data = [v for k, v in corpus_vio1.items() if k not in training_key_list]

        training_segments = np.concatenate(
            [x["segments"] for x in training_data], axis=0
        )
        training_lens = np.array(
            list(itertools.chain.from_iterable([x["lens"] for x in training_data]))
        )
        testing_segments = np.concatenate([x["segments"] for x in testing_data], axis=0)
        testing_lens = np.array(
            list(itertools.chain.from_iterable([x["lens"] for x in testing_data]))
        )

        print("shape of training data =", training_segments.shape)
        print("shape of testing data =", testing_segments.shape)

        X = {
            "train": {
                "pitch": training_segments[:, :, "pitch"],
                "start": training_segments[:, :, "start"],
                "duration": training_segments[:, :, "duration"],
                "beat_type": training_segments[:, :, "beat_type"],
                "lens": training_lens,
            },
            "test": {
                "pitch": testing_segments[:, :, "pitch"],
                "start": testing_segments[:, :, "start"],
                "duration": testing_segments[:, :, "duration"],
                "beat_type": testing_segments[:, :, "beat_type"],
                "lens": testing_lens,
            },
        }

        Y = {
            "train": {
                "string": training_segments[
                    :, self.context_window : -self.context_window, "string"
                ],
                "position": training_segments[
                    :, self.context_window : -self.context_window, "position"
                ],
                "finger": training_segments[
                    :, self.context_window : -self.context_window, "finger"
                ],
            },
            "test": {
                "string": testing_segments[
                    :, self.context_window : -self.context_window, "string"
                ],
                "position": testing_segments[
                    :, self.context_window : -self.context_window, "position"
                ],
                "finger": testing_segments[
                    :, self.context_window : -self.context_window, "finger"
                ],
            },
        }

        return X, Y

    def normalize(self, inputs, epsilon=1e-6, scope="ln", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean = tf.reduce_mean(inputs, axis=[-1], keepdims=True)
            variance = tf.reduce_mean(
                tf.squared_difference(inputs, mean), axis=[-1], keepdims=True
            )
            normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)

            beta = tf.get_variable(
                "beta_bias", params_shape, initializer=tf.zeros_initializer()
            )
            gamma = tf.get_variable(
                "gamma", params_shape, initializer=tf.ones_initializer()
            )
            outputs = gamma * normalized + beta
        return outputs

    def BLSTM(self, x_p, x_s, x_d, x_b, x_len, dropout, activation='tanh'):
        """p=pitch, s=start, d=duration, b=beat_type"""
        with tf.name_scope("Input_embedding"):
            x_p_onehot = tf.one_hot(x_p - self.lowest_pitch, depth=self.n_p_classes)
            x_b_onehot = tf.one_hot(x_b, depth=self.n_b_classes)
            input = tf.concat(
                [x_p_onehot, x_s[:, :, None], x_d[:, :, None], x_b_onehot], axis=2
            )
            input_embedded = tf.layers.dense(input, self.embedding_size)
            input_embedded = self.normalize(input_embedded, scope="input_ln")
            input_embedded = activation(input_embedded)
            input_embedded = tf.nn.dropout(input_embedded, keep_prob=1 - dropout)

        with tf.name_scope("BLSTM_cells"):
            cell_fw = LSTMBlockCell(num_units=self.hidden_size, name="cell_fw")
            cell_bw = LSTMBlockCell(num_units=self.hidden_size, name="cell_bw")

        with tf.name_scope("RNN"):
            # Reshape the input to have shape [batch_size, context_window * 2 + 1, embedding_size]
            input_reshaped = tf.reshape(
                input_embedded, [-1, self.context_window * 2 + 1, self.embedding_size]
            )

            # bi-LSTM
            (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=input_reshaped,
                sequence_length=x_len,
                dtype=tf.float32,
                time_major=False,
            )
            hidden_states = tf.concat((output_fw, output_bw), axis=-1)

            hidden_states = self.normalize(hidden_states, scope="hidden_ln")
            hidden_states = activation(hidden_states)
            hidden_states = tf.nn.dropout(hidden_states, keep_prob=1 - dropout)

        with tf.name_scope("Output"):
            s_logits = tf.layers.dense(
                hidden_states, self.n_str_classes, name="string_out"
            )
            p_logits = tf.layers.dense(
                hidden_states, self.n_pos_classes, name="position_out"
            )

        return s_logits, p_logits

    def create_str_mask(self, pitches):
        mask = np.zeros(
            (pitches.shape[0], pitches.shape[1], self.n_str_classes), dtype=np.float32
        )
        for i in range(pitches.shape[0]):
            for j in range(pitches.shape[1]):
                group = pitches[i, j]
                for pitch in group:
                    if 55 <= pitch <= 103:
                        if pitch >= 76:
                            mask[i, j, 4] = 1  # E string
                        elif pitch >= 69:
                            mask[i, j, 3] = 1  # A string
                        elif pitch >= 62:
                            mask[i, j, 2] = 1  # D string
                        else:
                            mask[i, j, 1] = 1  # G string
        return mask

    def create_pos_mask(self, pitch, str_pred):
        # Define a dictionary to map pitch ranges to their corresponding position masks
        pitch_masks = {
            (55, 55): [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            (56, 58): [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            (59, 59): [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            (60, 61): [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            (62, 64): {
                1: [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                2: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (65, 68): {
                1: [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                2: [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (69, 70): {
                1: [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                2: [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                3: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (71, 73): {
                1: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                3: [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (74, 78): {
                1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                2: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                3: [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                4: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (79, 83): {
                2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                3: [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                4: [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (84, 88): {
                3: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (89, 92): {
                3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                4: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (93, 95): {
                4: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (96, 97): {
                4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (98, 99): {
                4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (100, 101): {
                4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            (102, 103): {
                4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            "default": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }

        mask = []
        for i in range(pitch.shape[0]):
            for j in range(pitch.shape[1]):
                p = pitch[i][j]
                s = str_pred[i][j]

                # Find the pitch range that the current pitch belongs to
                pitch_range = next(
                    (r for r in pitch_masks if r[0] <= p <= r[1]), "default"
                )

                if isinstance(pitch_masks[pitch_range], dict):
                    # If the pitch range has string-specific masks, use the corresponding mask
                    mask.append(
                        pitch_masks[pitch_range].get(
                            s, pitch_masks[pitch_range]["default"]
                        )
                    )
                else:
                    # Otherwise, use the default mask for the pitch range
                    mask.append(pitch_masks[pitch_range])

        return np.reshape(
            mask, newshape=[pitch.shape[0], self.n_steps, self.n_pos_classes]
        )

    def predict_finger(self, pitch, string, pos, prev_fin):
        # Define a dictionary to map string and position to finger patterns
        # finger_patterns is a nested dictionary that maps string numbers (1-4) to positions (0-14),
        # and each position is associated with a tuple of possible finger numbers (0-5).
        # The finger numbers represent:
        # 0: open string, 1: index finger, 2: middle finger, 3: ring finger, 4: pinky finger, 5: thumb position
        # Example: finger_patterns[1][0] = (0, 1, 2, 3, 4) means that for string 1 at position 0,
        # the possible fingers are open string, index, middle, ring, and pinky.
        finger_patterns = {
            1: {
                0: (0, 1, 2, 3, 4),
                1: (0, 1, 2, 3, 4),
                2: (0, 1, 2, 3, 4),
                3: (0, 1, 2, 3, 4),
                4: (0, 1, 2, 3, 4),
                5: (0, 1, 2, 3, 4),
                6: (0, 1, 2, 3, 4),
                7: (1, 2, 3, 4),
                8: (1, 2, 3, 4),
                9: (1, 2, 3, 4),
                10: (1, 2, 3, 4),
                11: (1, 2, 3, 4),
                12: (1, 2, 3, 4),
                13: (1, 2, 3, 4),
                14: (1, 2, 3, 4, 5),
            },
            2: {
                0: (0, 1, 2, 3, 4),
                1: (0, 1, 2, 3, 4),
                2: (0, 1, 2, 3, 4),
                3: (0, 1, 2, 3, 4),
                4: (1, 2, 3, 4),
                5: (1, 2, 3, 4),
                6: (1, 2, 3, 4),
                7: (1, 2, 3, 4),
                8: (1, 2, 3, 4),
                9: (1, 2, 3, 4),
                10: (1, 2, 3, 4),
                11: (1, 2, 3, 4),
                12: (1, 2, 3, 4),
                13: (1, 2, 3, 4),
                14: (1, 2, 3, 4, 5),
            },
            3: {
                0: (0, 1, 2, 3, 4),
                1: (0, 1, 2, 3, 4),
                2: (1, 2, 3, 4),
                3: (1, 2, 3, 4),
                4: (1, 2, 3, 4),
                5: (1, 2, 3, 4),
                6: (1, 2, 3, 4),
                7: (1, 2, 3, 4),
                8: (1, 2, 3, 4),
                9: (1, 2, 3, 4),
                10: (1, 2, 3, 4),
                11: (1, 2, 3, 4),
                12: (1, 2, 3, 4),
                13: (1, 2, 3, 4),
                14: (1, 2, 3, 4, 5),
            },
            4: {
                0: (0, 1, 2, 3, 4),
                1: (1, 2, 3, 4),
                2: (1, 2, 3, 4),
                3: (1, 2, 3, 4),
                4: (1, 2, 3, 4),
                5: (1, 2, 3, 4),
                6: (1, 2, 3, 4),
                7: (1, 2, 3, 4),
                8: (1, 2, 3, 4),
                9: (1, 2, 3, 4),
                10: (1, 2, 3, 4),
                11: (1, 2, 3, 4),
                12: (1, 2, 3, 4),
                13: (1, 2, 3, 4),
                14: (1, 2, 3, 4, 5),
            },
        }

        # Define a dictionary to map special cases to their finger probabilities
        # special_cases is a dictionary that maps specific combinations of pitch, string, and position
        # to a tuple containing two elements:
        # - A tuple of possible finger numbers for that specific case
        # - A tuple of probabilities associated with each finger number
        # This dictionary is used to handle special cases where certain pitch, string, and position
        # combinations have different probabilities for finger assignments.
        # Example: special_cases[(82, 4, 1)] = ((3, 4), (0.2, 0.8)) means that for pitch 82 on string 4 at position 1,
        # the possible fingers are 3 (ring) and 4 (pinky), with probabilities 0.2 and 0.8, respectively.
        special_cases = {
            (82, 4, 1): ((3, 4), (0.2, 0.8)),
            (82, 4, 0): ((3, 4), (0.2, 0.8)),
            (75, 3, 1): ((3, 4), (0.2, 0.8)),
            (75, 3, 0): ((3, 4), (0.2, 0.8)),
            (68, 2, 1): ((3, 4), (0.2, 0.8)),
            (68, 2, 0): ((3, 4), (0.2, 0.8)),
            (61, 1, 1): ((3, 4), (0.2, 0.8)),
            (61, 1, 0): ((3, 4), (0.2, 0.8)),
            (103, 4, 14): ((4, 5), (0.8, 0.2)),
            (102, 4, 14): ((4, 5), (0.8, 0.2)),
            (101, 4, 14): ((4, 5), (0.8, 0.2)),
            (100, 4, 13): ((4, 5), (0.8, 0.2)),
            (99, 4, 13): ((4, 5), (0.8, 0.2)),
            (98, 4, 13): ((4, 5), (0.8, 0.2)),
            (97, 4, 12): ((4, 5), (0.8, 0.2)),
            (96, 4, 12): ((4, 5), (0.8, 0.2)),
            (95, 4, 11): ((4, 5), (0.8, 0.2)),
            (94, 4, 11): ((4, 5), (0.8, 0.2)),
            (93, 4, 10): ((4, 5), (0.8, 0.2)),
            (92, 4, 10): ((4, 5), (0.8, 0.2)),
            (91, 4, 9): ((4, 5), (0.8, 0.2)),
            (90, 4, 9): ((4, 5), (0.8, 0.2)),
            (89, 4, 8): ((4, 5), (0.8, 0.2)),
            (88, 4, 8): ((4, 5), (0.8, 0.2)),
            (87, 4, 7): ((4, 5), (0.8, 0.2)),
            (86, 4, 7): ((4, 5), (0.8, 0.2)),
            (85, 4, 6): ((4, 5), (0.8, 0.2)),
            (84, 4, 6): ((4, 5), (0.8, 0.2)),
            (83, 4, 5): ((4, 5), (0.8, 0.2)),
            (82, 4, 5): ((4, 5), (0.8, 0.2)),
            (81, 4, 4): ((4, 5), (0.8, 0.2)),
            (80, 4, 4): ((4, 5), (0.8, 0.2)),
            (79, 4, 3): ((4, 5), (0.8, 0.2)),
            (78, 4, 3): ((4, 5), (0.8, 0.2)),
            (77, 4, 2): ((4, 5), (0.8, 0.2)),
            (76, 4, 2): ((4, 5), (0.8, 0.2)),
            (76, 3, 14): ((4, 5), (0.8, 0.2)),
            (75, 3, 14): ((4, 5), (0.8, 0.2)),
            (74, 3, 13): ((4, 5), (0.8, 0.2)),
            (73, 3, 13): ((4, 5), (0.8, 0.2)),
            (72, 3, 12): ((4, 5), (0.8, 0.2)),
            (71, 3, 12): ((4, 5), (0.8, 0.2)),
            (70, 3, 11): ((4, 5), (0.8, 0.2)),
            (69, 3, 11): ((4, 5), (0.8, 0.2)),
            (69, 2, 14): ((4, 5), (0.8, 0.2)),
            (68, 2, 14): ((4, 5), (0.8, 0.2)),
            (67, 2, 13): ((4, 5), (0.8, 0.2)),
            (66, 2, 13): ((4, 5), (0.8, 0.2)),
            (65, 2, 12): ((4, 5), (0.8, 0.2)),
            (64, 2, 12): ((4, 5), (0.8, 0.2)),
            (63, 2, 11): ((4, 5), (0.8, 0.2)),
            (62, 2, 11): ((4, 5), (0.8, 0.2)),
            (62, 1, 14): ((4, 5), (0.8, 0.2)),
            (61, 1, 14): ((4, 5), (0.8, 0.2)),
            (60, 1, 13): ((4, 5), (0.8, 0.2)),
            (59, 1, 13): ((4, 5), (0.8, 0.2)),
            (58, 1, 12): ((4, 5), (0.8, 0.2)),
            (57, 1, 12): ((4, 5), (0.8, 0.2)),
            (56, 1, 11): ((4, 5), (0.8, 0.2)),
            (55, 1, 11): ((4, 5), (0.8, 0.2)),
        }

        if string not in finger_patterns:
            return 0  # Invalid string

        if pos not in finger_patterns[string]:
            return 0  # Invalid position

        # Get the finger pattern for the given string and position
        pattern = finger_patterns[string][pos]

        # Calculate the index finger based on the string and position
        index_fin = pattern[0]

        # Calculate the distance between the pitch and the index finger
        distance = pitch - index_fin

        # Determine the finger based on the distance
        if distance < 0:
            finger = 5  # Thumb position
        elif distance == 0:
            if pos == 0:
                finger = 0  # Open string
            else:
                finger = 1  # Index finger
        elif distance <= 2:
            finger = 2  # Middle finger
        elif distance <= 4:
            finger = 3  # Ring finger
        else:
            finger = 4  # Pinky finger

        # Check if the current case matches any special case
        if (pitch, string, pos) in special_cases:
            fingers, probs = special_cases[(pitch, string, pos)]
            if prev_fin == finger:
                if finger == fingers[0]:
                    finger = np.random.choice(fingers, p=probs)
                else:
                    finger = np.random.choice(fingers, p=probs[::-1])
            else:
                finger = np.random.choice(fingers, p=probs[::-1])

        return finger

    def count_fin(self, prev_fin, prev_pitch, pitch, now_fin, now_str, next_str):
        # Define finger patterns for each previous finger
        finger_patterns = {
            1: [
                (2, 0, 2),
                (4, 2, 4),
                (6, 4, 6),
                (8, 6, 14, 2),
                (10, 8, 14, 3),
                (12, 10, 14, 4),
                (14, 12, 14, 5),
            ],
            2: [
                (2, 0, 2),
                (4, 2, 4),
                (6, 4, 14, 1),
                (8, 6, 14, 3),
                (10, 8, 14, 4),
                (14, 10, 14, 5),
            ],
            3: [(2, 0, 2), (4, 2, 14, 1), (6, 4, 14, 2), (8, 6, 14, 4), (14, 8, 14, 5)],
            4: [(2, 0, 14, 1), (4, 2, 14, 2), (6, 4, 14, 3), (14, 6, 14, 5)],
        }

        # Define string change patterns for each previous finger
        string_change_patterns = {
            1: [(4, 3), (3, 2), (2, 1)],
            2: [(1, 2), (2, 3), (3, 4)],
            3: [(1, 2), (2, 3), (3, 4)],
            4: [(4, 3), (3, 2), (2, 1)],
        }

        fin = now_fin
        str = now_str

        if prev_fin != 0:
            pitch_diff = pitch - prev_pitch

            for pattern in finger_patterns[prev_fin]:
                if pattern[0] <= abs(pitch_diff) <= pattern[1]:
                    fin = pattern[2]
                    break

            if pitch_diff < 0 and next_str != now_str and now_str != 1:
                for pattern in string_change_patterns[prev_fin]:
                    if now_str == pattern[0]:
                        str = pattern[1]
                        break

        return fin, str

    def avoid_same_finger(self, str, pos, fin, pitch):
        # change finger if there are >=3 same finger
        new_fin = []
        new_str = []
        prev_fin = 0
        prev_pitch = pitch[0][0]
        prev_pos = 0
        flag = 0

        for i in range(pitch.shape[0]):
            for j in range(self.n_steps):
                # count repeat finger
                if prev_fin == fin[i][j]:
                    flag += 1
                    # if same pitch then don't count
                    if prev_pitch == pitch[i][j]:
                        flag -= 1
                    # repeat three time
                    if flag == 2:
                        flag = 0
                        if j == self.n_steps - 1 and i != pitch.shape[0] - 1:
                            next_str = str[i + 1][0]
                            next_pos = pos[i + 1][0]
                        elif j != self.n_steps - 1:
                            next_str = str[i][j + 1]
                            next_pos = pos[i][j + 1]
                        else:
                            next_str = str[i][j]
                            next_pos = pos[i][j]

                        if fin[i][j] == 5:  # thumb position
                            if next_pos > pos[i][j]:
                                fin[i][j] = 1  # change to index finger
                            else:
                                fin[i][j] = 4  # change to pinky finger
                        else:
                            fin[i][j], str[i][j] = self.count_fin(
                                prev_fin,
                                prev_pitch,
                                pitch[i][j],
                                fin[i][j],
                                str[i][j],
                                next_str,
                            )
                else:
                    flag = 0

                new_fin.append(fin[i][j])
                new_str.append(str[i][j])
                prev_fin = fin[i][j]
                prev_pitch = pitch[i][j]
                prev_pos = pos[i][j]

        new_str = np.reshape(new_str, newshape=pitch.shape)
        new_fin = np.reshape(new_fin, newshape=pitch.shape)

        return new_str, pos, new_fin

    def decode_pos(self, pos_scores, mode="basic", k=3):
        # get the nearest position w.r.t. the previous position from the top-k positions
        top_k_pos = (np.argsort(pos_scores, axis=2)[:, :, ::-1])[
            :, :, :k
        ]  # [batch, n_steps, 3]

        if mode == "basic":
            out_pos = np.argmax(pos_scores, axis=2)
        elif mode == "nearest":
            out_pos = np.zeros_like(
                pos_scores[:, :, 0], dtype=np.int32
            )  # [batch, n_steps]
            out_pos[:, 0] = np.argmax(pos_scores[:, 0, :], axis=1)  # [batch]
            for j in range(1, pos_scores.shape[1]):
                prev_pos = out_pos[:, j - 1]  # [batch]
                dist = top_k_pos[:, j, :] - prev_pos[:, None]  # [batch, 3]
                minarg = np.argmin(np.abs(dist), axis=1)  # [batch]
                minarg = (np.arange(len(minarg)), minarg)
                out_pos[:, j] = top_k_pos[:, j, :][minarg]  # [batch]
        elif mode == "lowest":
            # out_pos = np.min(top_k_pos, axis=2)
            best_pos = np.argmax(pos_scores, axis=2)
            if_best_in_top_k = np.any(top_k_pos == best_pos[:, :, None], axis=2)
            out_pos = np.min(top_k_pos, axis=2)
            out_pos[if_best_in_top_k] = best_pos[if_best_in_top_k]
        else:
            print("Error: invalid mode.")
            exit(1)
        return out_pos

    def create_placeholders(self):
        x_p = tf.placeholder(
            tf.int32, [None, self.n_steps + self.context_window * 2], name="pitch"
        )
        x_s = tf.placeholder(
            tf.float32, [None, self.n_steps + self.context_window * 2], name="start"
        )
        x_d = tf.placeholder(
            tf.float32, [None, self.n_steps + self.context_window * 2], name="duration"
        )
        x_b = tf.placeholder(
            tf.int32, [None, self.n_steps + self.context_window * 2], name="beat_type"
        )
        x_len = tf.placeholder(tf.int32, [None], name="valid_lens")
        y_s = tf.placeholder(tf.int32, [None, self.n_steps], name="string")
        y_p = tf.placeholder(tf.int32, [None, self.n_steps], name="position")
        y_f = tf.placeholder(tf.int32, [None, self.n_steps], name="finger")
        f_s = tf.placeholder(tf.float32, name="f_score_string")
        f_p = tf.placeholder(tf.float32, name="f_score_position")
        f_f = tf.placeholder(tf.float32, name="f_score_finger")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        global_step = tf.placeholder(dtype=tf.int32, name="global_step")
        return (
            x_p,
            x_s,
            x_d,
            x_b,
            x_len,
            y_s,
            y_p,
            y_f,
            f_s,
            f_p,
            f_f,
            dropout,
            global_step,
        )

    def build_model(self, x_p, x_s, x_d, x_b, x_len, dropout):
        with tf.variable_scope("model"):
            logits_s, logits_p = self.BLSTM(x_p, x_s, x_d, x_b, x_len, dropout)
        return logits_s, logits_p

    def calculate_loss(self, logits_s, logits_p, y_s, y_p, x_len):
        with tf.name_scope("loss"):
            seq_mask = tf.sequence_mask(
                lengths=x_len,
                maxlen=self.n_steps + self.context_window * 2,
                dtype=tf.float32,
            )  # [batch, n_steps + context_window * 2]
            central_mask = seq_mask[
                :, self.context_window : -self.context_window
            ]  # [batch, n_steps]
            n_valid = tf.reduce_sum(central_mask)
            loss_s = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(y_s, self.n_str_classes),
                logits=logits_s,
                weights=central_mask,
            )
            loss_p = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(y_p, self.n_pos_classes),
                logits=logits_p,
                weights=central_mask,
            )
            loss = loss_s + loss_p
        summary_loss = tf.Variable(
            [0.0 for _ in range(3)], trainable=False, dtype=tf.float32
        )
        summary_valid = tf.Variable(0, trainable=False, dtype=tf.float32)
        update_loss = tf.assign(
            summary_loss, summary_loss + n_valid * [loss, loss_s, loss_p]
        )
        update_valid = tf.assign(summary_valid, summary_valid + n_valid)
        mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
        clr_summary_loss = summary_loss.initializer
        clr_summary_valid = summary_valid.initializer
        tf.summary.scalar("Loss_total", summary_loss[0])
        tf.summary.scalar("Loss_string", summary_loss[1])
        tf.summary.scalar("Loss_position", summary_loss[2])
        return (
            loss,
            update_loss,
            update_valid,
            mean_loss,
            clr_summary_loss,
            clr_summary_valid,
            summary_loss,
        )

    def optimize(self, loss, global_step, n_iterations_per_epoch):
        with tf.name_scope("Optimization"):
            # apply learning rate decay
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.initial_learning_rate,
                global_step=global_step,
                decay_steps=n_iterations_per_epoch,
                decay_rate=0.96,
                staircase=True,
            )
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9
            )
            train_op = optimizer.minimize(loss)
        return train_op

    def evaluate(self, x_p, logits_s, f_s, f_p, f_f):
        with tf.name_scope("Evaluation"):
            str_mask = self.create_str_mask(
                x_p[:, self.context_window : -self.context_window]
            )
            logits_s = tf.where(
                tf.equal(str_mask, 1), logits_s, tf.ones_like(logits_s) * -self.INFIN
            )  # masking
            pred_s = tf.argmax(logits_s, axis=2, output_type=tf.int32)
        tf.summary.scalar("F1_s", f_s)
        tf.summary.scalar("F1_p", f_p)
        tf.summary.scalar("F1_f", f_f)
        return pred_s


def run_training(
    self,
    X,
    Y,
    n_train_samples,
    n_test_samples,
    n_iterations_per_epoch,
    x_p,
    x_s,
    x_d,
    x_b,
    x_len,
    y_s,
    y_p,
    y_f,
    f_s,
    f_p,
    f_f,
    dropout,
    global_step,
    loss,
    update_loss,
    update_valid,
    mean_loss,
    clr_summary_loss,
    clr_summary_valid,
    train_op,
    pred_s,
    logits_p,
):
    print("Saving model to: %s" % self.save_dir)
    train_writer = tf.summary.FileWriter(self.save_dir + "/train")
    test_writer = tf.summary.FileWriter(self.save_dir + "/test")
    merged = tf.summary.merge_all()
    train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time()  # start time of training
        best_test_score = [0.0 for _ in range(3)]
        in_succession = 0
        best_epoch = 0
        # Batched indices
        indices = np.arange(n_train_samples)
        batch_indices = [
            indices[x : x + self.batch_size]
            for x in range(0, len(indices), self.batch_size)
        ]
        step = 0
        for epoch in range(self.n_epochs):
            if epoch > 0:
                # Shuffle training data
                indices = np.array(
                    random.sample(range(n_train_samples), n_train_samples)
                )
                batch_indices = np.array(
                    [
                        indices[x : x + self.batch_size]
                        for x in range(0, len(indices), self.batch_size)
                    ]
                )

            # Batched training
            train_str, train_pos_scores = [], []
            for idx in batch_indices:
                batch = (
                    X["train"]["pitch"][idx],
                    X["train"]["start"][idx],
                    X["train"]["duration"][idx],
                    X["train"]["beat_type"][idx],
                    X["train"]["lens"][idx],
                    Y["train"]["string"][idx],
                    Y["train"]["position"][idx],
                    Y["train"]["finger"][idx],
                )

                train_run_list = [
                    train_op,
                    update_valid,
                    update_loss,
                    loss,
                    loss_s,
                    loss_p,
                    pred_s,
                    logits_p,
                ]
                train_feed_dict = {
                    x_p: batch[0],
                    x_s: batch[1],
                    x_d: batch[2],
                    x_b: batch[3],
                    x_len: batch[4],
                    y_s: batch[5][:, self.context_window : -self.context_window],
                    y_p: batch[6][:, self.context_window : -self.context_window],
                    y_f: batch[7][:, self.context_window : -self.context_window],
                    dropout: self.drop,
                    global_step: step + 1,
                }
                (
                    _,
                    _,
                    _,
                    train_loss,
                    train_loss_s,
                    train_loss_p,
                    train_pred_s,
                    train_logits_p,
                ) = sess.run(train_run_list, feed_dict=train_feed_dict)
                train_pos_scores.append(train_logits_p)
                if step == 0:
                    print(
                        "*~ loss_s %.4f, loss_p %.4f, ~*" % (train_loss_s, train_loss_p)
                    )
                step += 1
            train_str = np.concatenate(train_str, axis=0)
            train_pos_scores = np.concatenate(train_pos_scores, axis=0)
            # Recovery ordering
            gather_id = [
                np.where(indices == ord)[0][0] for ord in range(n_train_samples)
            ]
            train_str = train_str[gather_id, :]
            train_pos_scores = train_pos_scores[gather_id, :]
            # Get position mask
            train_pos_mask = self.create_pos_mask(
                X["train"]["pitch"][:, self.context_window : -self.context_window],
                train_str,
            )
            # Get masked positions
            train_pos_scores[train_pos_mask == 0] = -self.INFIN
            train_pos = np.argmax(train_pos_scores, axis=2)
            # Decode fingers
            train_fin = []
            prev_train_fin = 0
            for i in range(n_train_samples):
                for j in range(self.n_steps):
                    prev_train_fin = self.predict_finger(
                        X["train"]["pitch"][i, j + self.context_window],
                        train_str[i, j],
                        train_pos[i, j],
                        prev_train_fin,
                    )
                    train_fin.append(prev_train_fin)
            train_fin = np.reshape(train_fin, newshape=[n_train_samples, self.n_steps])
            # Change str and fin to avoid using the same finger
            train_str, train_pos, train_fin = self.avoid_same_finger(
                train_str,
                train_pos,
                train_fin,
                X["train"]["pitch"][:, self.context_window : -self.context_window],
            )
            # Calculate performance
            boolean_mask = (
                np.arange(self.n_steps)[None, :]
                < (X["train"]["lens"] - self.context_window * 2)[:, None]
            )
            train_P_s, train_R_s, train_F_s, _ = precision_recall_fscore_support(
                y_true=Y["train"]["string"][boolean_mask],
                y_pred=train_str[boolean_mask],
                average="micro",
            )
            train_P_p, train_R_p, train_F_p, _ = precision_recall_fscore_support(
                y_true=Y["train"]["position"][boolean_mask],
                y_pred=train_pos[boolean_mask],
                average="micro",
            )
            train_P_f, train_R_f, train_F_f, _ = precision_recall_fscore_support(
                y_true=Y["train"]["finger"][boolean_mask],
                y_pred=train_fin[boolean_mask],
                average="micro",
            )

            # Display training log
            _, train_losses, train_summary = sess.run(
                [mean_loss, summary_loss, merged],
                feed_dict={f_s: train_F_s, f_p: train_F_p, f_f: train_F_f},
            )
            sess.run([clr_summary_valid, clr_summary_loss])  # clear summaries
            train_writer.add_summary(train_summary, epoch)
            print(
                "==== epoch %d: train_loss %.4f (s %.4f, p %.4f), train_F1: s %.4f, p %.4f, f %.4f ===="
                % (
                    epoch,
                    train_losses[0],
                    train_losses[1],
                    train_losses[2],
                    train_F_s,
                    train_F_p,
                    train_F_f,
                )
            )
            sample_id = random.randint(0, n_train_samples - 1)
            print("len".ljust(6, " "), X["train"]["lens"][sample_id])
            print(
                "x_p".ljust(4, " "),
                "".join(
                    [
                        (
                            pretty_midi.note_number_to_name(p).rjust(4, " ")
                            if p < self.pitch_for_invalid_note
                            else "X".rjust(4, " ")
                        )
                        for p in X["train"]["pitch"][
                            sample_id, self.context_window : -self.context_window
                        ]
                    ]
                ),
            )
            print(
                "y_s".ljust(4, " "),
                "".join(
                    [str(s).rjust(4, " ") for s in Y["train"]["string"][sample_id]]
                ),
            )
            print(
                "o_s".ljust(4, " "),
                "".join([str(s).rjust(4, " ") for s in train_str[sample_id]]),
            )
            print(
                "y_p".ljust(4, " "),
                "".join(
                    [str(p).rjust(4, " ") for p in Y["train"]["position"][sample_id]]
                ),
            )
            print(
                "o_p".ljust(4, " "),
                "".join([str(p).rjust(4, " ") for p in train_pos[sample_id]]),
            )
            print(
                "y_f".ljust(4, " "),
                "".join(
                    [str(f).rjust(4, " ") for f in Y["train"]["finger"][sample_id]]
                ),
            )
            print(
                "o_f".ljust(4, " "),
                "".join([str(f).rjust(4, " ") for f in train_fin[sample_id]]),
            )

            # Testing
            test_run_list = [update_valid, update_loss, pred_s, logits_p]
            test_feed_dict = {
                x_p: X["test"]["pitch"],
                x_s: X["test"]["start"],
                x_d: X["test"]["duration"],
                x_b: X["test"]["beat_type"],
                x_len: X["test"]["lens"],
                y_s: Y["test"]["string"],
                y_p: Y["test"]["position"],
                y_f: Y["test"]["finger"],
                dropout: 0,
            }
            _, _, test_str, test_pos_scores = sess.run(
                test_run_list, feed_dict=test_feed_dict
            )

            # Get position mask
            test_pos_mask = self.create_pos_mask(
                X["test"]["pitch"][:, self.context_window : -self.context_window],
                test_str,
            )
            # Decode positions using different modes
            test_pos_scores[test_pos_mask == 0] = -self.INFIN
            test_pos_basic = np.argmax(test_pos_scores, axis=2)
            test_pos_lowest = self.decode_pos(test_pos_scores, mode="lowest")
            test_pos_nearest = self.decode_pos(test_pos_scores, mode="nearest")

            # Decode fingers
            test_fin_basic, test_fin_lowest, test_fin_nearest = [], [], []
            prev_test_fin_basic = prev_test_fin_lowest = prev_test_fin_nearest = 0
            for i in range(n_test_samples):
                for j in range(self.n_steps):
                    prev_test_fin_basic = self.predict_finger(
                        X["test"]["pitch"][i, j + self.context_window],
                        test_str[i, j],
                        test_pos_basic[i, j],
                        prev_test_fin_basic,
                    )
                    prev_test_fin_lowest = self.predict_finger(
                        X["test"]["pitch"][i, j + self.context_window],
                        test_str[i, j],
                        test_pos_lowest[i, j],
                        prev_test_fin_lowest,
                    )
                    prev_test_fin_nearest = self.predict_finger(
                        X["test"]["pitch"][i, j + self.context_window],
                        test_str[i, j],
                        test_pos_nearest[i, j],
                        prev_test_fin_nearest,
                    )
                    test_fin_basic.append(prev_test_fin_basic)
                    test_fin_lowest.append(prev_test_fin_lowest)
                    test_fin_nearest.append(prev_test_fin_nearest)
                test_fin_basic = np.reshape(
                    test_fin_basic, newshape=[n_test_samples, self.n_steps]
                )
                test_fin_lowest = np.reshape(
                    test_fin_lowest, newshape=[n_test_samples, self.n_steps]
                )
                test_fin_nearest = np.reshape(
                    test_fin_nearest, newshape=[n_test_samples, self.n_steps]
                )
                # Change str and fin to avoid using the same finger
                test_str_basic, test_pos_basic, test_fin_basic = self.avoid_same_finger(
                    test_str,
                    test_pos_basic,
                    test_fin_basic,
                    X["test"]["pitch"][:, self.context_window : -self.context_window],
                )
                test_str_lowest, test_pos_lowest, test_fin_lowest = (
                    self.avoid_same_finger(
                        test_str,
                        test_pos_lowest,
                        test_fin_lowest,
                        X["test"]["pitch"][
                            :, self.context_window : -self.context_window
                        ],
                    )
                )
                test_str_nearest, test_pos_nearest, test_fin_nearest = (
                    self.avoid_same_finger(
                        test_str,
                        test_pos_nearest,
                        test_fin_nearest,
                        X["test"]["pitch"][
                            :, self.context_window : -self.context_window
                        ],
                    )
                )

            # Calculate performance
            boolean_mask = (
                np.arange(self.n_steps)[None, :]
                < (X["test"]["lens"] - self.context_window * 2)[:, None]
            )
            test_P_s, test_R_s, test_F_s, _ = precision_recall_fscore_support(
                y_true=Y["test"]["string"][boolean_mask],
                y_pred=test_str_basic[boolean_mask],
                average="micro",
            )
            test_P_p, test_R_p, test_F_p, _ = precision_recall_fscore_support(
                y_true=Y["test"]["position"][boolean_mask],
                y_pred=test_pos_basic[boolean_mask],
                average="micro",
            )
            test_P_f, test_R_f, test_F_f, _ = precision_recall_fscore_support(
                y_true=Y["test"]["finger"][boolean_mask],
                y_pred=test_fin_basic[boolean_mask],
                average="micro",
            )

            _, test_losses, test_summary = sess.run(
                [mean_loss, summary_loss, merged],
                feed_dict={f_s: test_F_s, f_p: test_F_p, f_f: test_F_f},
            )
            sess.run([clr_summary_valid, clr_summary_loss])  # clear summaries
            test_writer.add_summary(test_summary, epoch)
            print(
                "----  epoch %d: test_loss %.4f (s %.4f, p %.4f), test_F1: s %.4f, p %.4f, f %.4f ----"
                % (
                    epoch,
                    test_losses[0],
                    test_losses[1],
                    test_losses[2],
                    test_F_s,
                    test_F_p,
                    test_F_f,
                )
            )
            sample_id = random.randint(0, n_test_samples - 1)
            print("len".ljust(6, " "), X["test"]["lens"][sample_id])
            print(
                "x_p".ljust(4, " "),
                "".join(
                    [
                        (
                            pretty_midi.note_number_to_name(p).rjust(4, " ")
                            if p < self.pitch_for_invalid_note
                            else "X".rjust(4, " ")
                        )
                        for p in X["test"]["pitch"][
                            sample_id, self.context_window : -self.context_window
                        ]
                    ]
                ),
            )
            print(
                "y_s".ljust(4, " "),
                "".join([str(s).rjust(4, " ") for s in Y["test"]["string"][sample_id]]),
            )
            print(
                "o_s".ljust(4, " "),
                "".join([str(s).rjust(4, " ") for s in test_str[sample_id]]),
            )
            print(
                "y_p".ljust(4, " "),
                "".join(
                    [str(p).rjust(4, " ") for p in Y["test"]["position"][sample_id]]
                ),
            )
            print(
                "o_p".ljust(4, " "),
                "".join([str(p).rjust(4, " ") for p in test_pos_basic[sample_id]]),
            )
            print(
                "y_f".ljust(4, " "),
                "".join([str(s).rjust(4, " ") for s in Y["test"]["finger"][sample_id]]),
            )
            print(
                "o_f".ljust(4, " "),
                "".join([str(s).rjust(4, " ") for s in test_fin_basic[sample_id]]),
            )

            # Check if early stopping
            if (test_F_s + test_F_p) > sum(best_test_score[:2]):
                best_test_score = [test_F_s, test_F_p, test_F_f]
                best_epoch = epoch
                in_succession = 0
                # Save variables of the model
                print("*saving variables...")
                saver.save(sess, self.save_dir + "/violin_fingering_estimation.ckpt")
            else:
                in_succession += 1
                if in_succession > self.n_in_succession:
                    print("Early stopping.")
                    break
        elapsed_time = time.time() - startTime
        np.set_printoptions(precision=4)
        print("training time = %.2f hr" % (elapsed_time / 3600))
        print("best epoch = ", best_epoch)
        print("best test score =", np.round(best_test_score, 4))


def train(self):
    # Load data
    corpus = self.load_data()
    corpus = self.segment_corpus(corpus, context_window=self.context_window)
    X, Y = self.create_training_and_testing_sets(corpus)
    n_train_samples = X["train"]["pitch"].shape[0]
    n_test_samples = X["test"]["pitch"].shape[0]
    n_iterations_per_epoch = int(np.ceil(n_train_samples / self.batch_size))
    print("n_iterations_per_epoch=", n_iterations_per_epoch)

    x_p, x_s, x_d, x_b, x_len, y_s, y_p, y_f, f_s, f_p, f_f, dropout, global_step = (
        self.create_placeholders()
    )
    logits_s, logits_p = self.build_model(x_p, x_s, x_d, x_b, x_len, dropout)
    (
        loss,
        update_loss,
        update_valid,
        mean_loss,
        clr_summary_loss,
        clr_summary_valid,
        summary_loss,
    ) = self.calculate_loss(logits_s, logits_p, y_s, y_p, x_len)
    train_op = self.optimize(loss, global_step, n_iterations_per_epoch)
    pred_s = self.evaluate(x_p, logits_s, f_s, f_p, f_f)

    self.run_training(
        X,
        Y,
        n_train_samples,
        n_test_samples,
        n_iterations_per_epoch,
        x_p,
        x_s,
        x_d,
        x_b,
        x_len,
        y_s,
        y_p,
        y_f,
        f_s,
        f_p,
        f_f,
        dropout,
        global_step,
        loss,
        update_loss,
        update_valid,
        mean_loss,
        clr_summary_loss,
        clr_summary_valid,
        train_op,
        pred_s,
        logits_p,
    )


if __name__ == "__main__":

    # Training
    model = model = violin_fingering_model()
    model.train()

    # Inference
    pitches = [55, 57, 59, 60, 62, 64, 66, 67] # G scale
    n_events = len(pitches)
    starts = [i * 1 for i in range(n_events)]
    durations = [1 for _ in range(n_events)]
    beat_types = [3 for _ in range(n_events)] # {'': 0, '1th': 1, '2th': 2, '4th': 3, '8th': 4, '16th': 5, '32th': 6}
    strings = [0 for _ in range(n_events)]
    positions = [0 for _ in range(n_events)]
    fingers = [0 for _ in range(n_events)]
    
    model = violin_fingering_model()
    pred_str, pred_pos, pred_fin = model.inference(pitches=pitches,
                                                   starts=starts,
                                                   durations=durations,
                                                   beat_types=beat_types,
                                                   strings=strings,
                                                   positions=positions,
                                                   fingers=fingers,
                                                   mode='basic') # valid mode = {'basic', 'lowest', 'nearest'}
    
    # Print the estimations
    string_classes = ['N', 'G', 'D', 'A', 'E']
    n_notes = len(pitches)
    print('pitch'.ljust(9), ''.join([pretty_midi.note_number_to_name(number).rjust(4) for number in pitches]))
    print('string'.ljust(9), ''.join([string_classes[s].rjust(4) for s in pred_str[0, :n_notes]]))
    print('position'.ljust(9), ''.join([str(p).rjust(4) for p in pred_pos[0, :n_notes]]))
    print('finger'.ljust(9), ''.join([str(f).rjust(4) for f in pred_fin[0, :n_notes]]))
