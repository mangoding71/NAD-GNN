from collections import deque
import numpy as np
from typing import List, Tuple, Set
from tensorflow import keras
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split



class _epsilon_neighborhood_of_p:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.center_point = (2, 2)
        self.test_points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        
    def _epsilon_neighborhood(self):
        """计算点p的epsilon邻域"""
        neighborhood = []
        for q in self.test_points:
            if self._distance(self.center_point, q) <= self.epsilon:
                neighborhood.append(q)
        return self.center_point,neighborhood

    def _distance(self, p: Tuple[float, float], q: Tuple[float, float]) -> float:
        """计算两点之间的欧氏距离"""
        return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _expand_cluster(self, p_index: int, neighbors: List[int], points: List[Tuple[float, float]], cluster_id: int):
    """从核心点p_index开始扩展簇"""
    self.cluster_label[p_index] = cluster_id
    i = 0
    while i < len(neighbors):
        q_index = neighbors[i]
        if q_index not in self.visited:
            self.visited.add(q_index)
            q_neighbors = self._epsilon_neighborhood(points[q_index], points)
            if len(q_neighbors) >= self.min_pts:
                neighbors.extend(q_neighbors)
        if q_index not in self.cluster_label:
            self.cluster_label[q_index] = cluster_id
        i += 1

def fit(self, points: List[Tuple[float, float]]) -> None:
    """拟合数据"""
    cluster_id = 0
    for i, p in enumerate(points):
        if i in self.visited:
            continue
        self.visited.add(i)
        neighbors = self._epsilon_neighborhood(p, points)
        if len(neighbors) < self.min_pts:
            self.cluster_label[i] = -1  # 标记为噪声点
        else:
            cluster_id += 1
            self._expand_cluster(i, neighbors, points, cluster_id)

def get_clusters(self) -> List[Set[Tuple[float, float]]]:
    """获取簇"""
    clusters = {}
    for point_index, cluster_index in self.cluster_label.items():
        if cluster_index == -1:
            continue
        if cluster_index not in clusters:
            clusters[cluster_index] = set()
        clusters[cluster_index].add(point_index)
    return list(map(lambda indices: set(points[i] for i in indices), clusters.values()))




class CoreTimeSeriesClusterIdentification:
    def __init__(self,
                alerts,
                labels,
                X,
                y,
                latent_dim,
                batch_size,
                epochs,
                max_single_channel_length, 
                max_output_length=None, 
                multi_channel=True, 
                norm=True, 
                verbose=0,
                epsilon = 0.5,
                minPts = 5):
        
        self.latent_dim, self.batch_size, self.max_single_channel_length = latent_dim, batch_size, max_single_channel_length
        self.epochs = epochs
        self.verbose = verbose
        self.train_x,self.test_x,self.train_y,self.test_y = train_test_split(X,y,test_size=0.2)
        
        input_characters = list(set(alerts)) + ['eos']

        # target_characters = list(set(labels))+['sos','eos']
        tmp = []
        for yi in y:
            tmp.extend(yi)
        target_characters = list(set(tmp))

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in X])
        self.max_decoder_seq_length = max([len(txt) for txt in y])

        self.max_output_length = max_output_length
        if not max_output_length:
            self.max_output_length = self.max_decoder_seq_length 

        print('Number of samples:', len(X))
        tmp = []
        for c in target_characters:
            if c == "s1":
                tmp.append("c1")
            elif c == "s2":
                tmp.append("c2")
            elif c == "s3":
                tmp.append("c3")
            elif c == "s5":
                tmp.append("c4")    
            elif c == "s4":
                tmp.append("c5")         
            else:
                tmp.append(c)
        print('Core Time Series Cluster:', tmp)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        if not multi_channel:
            self._build_single_model()
            self.channels = 1
        else:
            self.channels = (self.max_encoder_seq_length) // self.max_single_channel_length
            if (self.max_encoder_seq_length) % self.max_single_channel_length != 0:
                self.channels += 1
            self._build_multi_model(norm = norm)
        
    def _prepare_data(self,x,y,max_encoder_seq_length, num_encoder_tokens,max_decoder_seq_length, num_decoder_tokens):
        encoder_input_data = np.zeros(
            (len(x), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(x), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(x), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(x, y)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, self.input_token_index['eos']] = 1.
            for t, char in enumerate(target_text):
                # decoder 输出比输入早一个step 
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, self.target_token_index['eos']] = 1.  #长度不足，用‘eos’填充
            decoder_target_data[i, t, self.target_token_index['eos']] = 1.  # 长度不足，用‘eos’填充
        return encoder_input_data, decoder_input_data, decoder_target_data

    def _build_single_model(self):
        self.encoder_inputs = keras.layers.Input(shape=(None, self.num_encoder_tokens))
        encoder = keras.layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]

        #decode

        self.decoder_inputs = keras.layers.Input(shape=(None, self.num_decoder_tokens))
        self.decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,initial_state = self.encoder_states)
        self.decoder_dense = keras.layers.Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = self.decoder_dense(decoder_outputs)

        #model
        self.model = keras.Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

    def _build_multi_model(self,norm = False):
        self.encoder_inputs = []
        H, C = [], []
        for i in range(self.channels):
            self.encoder_inputs.append(keras.layers.Input(shape=(None, self.num_encoder_tokens)))
            encoder = keras.layers.LSTM(self.latent_dim, return_state=True)
            encoder_outputs, state_h, state_c = encoder(self.encoder_inputs[i])
            H.append(state_h)
            C.append(state_c)
        if norm == True:
            self.encoder_states = [tf.reduce_mean(H,axis=0),tf.reduce_mean(C,axis=0)]
        else:
            sum_h, sum_c = H[index[0]], C[index[0]]
            for ix in range(1,len(index)):
                sum_h = keras.layers.add([H[index[ix]],sum_h]) 
                sum_c = keras.layers.add([C[index[ix]],sum_c]) 
            self.encoder_states = [sum_h,sum_c]


        #decode

        self.decoder_inputs = keras.layers.Input(shape=(None, self.num_decoder_tokens))
        self.decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,initial_state = self.encoder_states)
        self.decoder_dense = keras.layers.Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = self.decoder_dense(decoder_outputs)

        #model
        self.model = keras.Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

    def save(self, model_name):
        self.model.save(model_name, save_format="tf")
        print("save ", model_name)

    def load(self, model_name):
        self.model = keras.models.load_model(model_name)
        print("load ",  model_name)

    def train(self):
        t0 = time.time()
        # Run training
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        encoder_input_data, decoder_input_data, decoder_target_data = self._prepare_data(
            x = self.train_x,
            y = self.train_y,
            max_encoder_seq_length = self.max_encoder_seq_length, #modefiy 0808
            num_encoder_tokens = self.num_encoder_tokens,
            max_decoder_seq_length = self.max_decoder_seq_length, 
            num_decoder_tokens = self.num_decoder_tokens
        )

        inputs = []
        for i in range(self.channels):
            inputs.append(encoder_input_data[:,i::self.channels])
        inputs.append(decoder_input_data)

        callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_accuracy',patience=1)]

        self.model.fit(inputs, decoder_target_data,
                  batch_size = self.batch_size,
                  epochs = self.epochs,
                  validation_split = 0.1,
                  verbose = self.verbose)

        t1 = time.time()
        self.training_time = (t1 - t0)/len(self.train_y)
        print("training spend time:{}".format(self.training_time * 10))
        print("train_loss:", self.model.history.history['loss'][-1]) #添加的loss输出

    def train_without_fit(self, model_name):
        t0 = time.time()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        encoder_input_data, decoder_input_data, decoder_target_data = self._prepare_data(
            x = self.train_x,
            y = self.train_y,
            max_encoder_seq_length = self.max_encoder_seq_length, 
            num_encoder_tokens = self.num_encoder_tokens,
            max_decoder_seq_length = self.max_decoder_seq_length,
            num_decoder_tokens = self.num_decoder_tokens
        )

        inputs = []
        for i in range(self.channels):
            inputs.append(encoder_input_data[:,i::self.channels])
        inputs.append(decoder_input_data)

        callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_accuracy',patience=1)]

        t1 = time.time()
        self.training_time = (t1 - t0)/len(self.train_y)
        print("training spend time:{}".format(self.training_time))

        self.model = keras.models.load_model(model_name)
        print("load ",  model_name)

        
    def expand_cluster(self, X , epsilon_neighborhood, C_i):
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())


        self.encoder_model = keras.Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = keras.layers.Input(shape=(self.latent_dim,))
        decoder_state_input_c = keras.layers.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state = decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)


        encoder_input_data, decoder_input_data, decoder_target_data = self._prepare_data(
            x = self.test_x,
            y = self.test_y,
            max_encoder_seq_length = self.max_encoder_seq_length, 
            num_encoder_tokens = self.num_encoder_tokens,
            max_decoder_seq_length = self.max_decoder_seq_length, 
            num_decoder_tokens = self.num_decoder_tokens
        )

        y_true = [yi[1:-1] for yi in self.test_y]

        input_seq = [encoder_input_data[:,c::self.channels] for c in range(self.channels)]


        decode_output = self._batch_decode_sequence(input_seq)

        return y_true, decode_output
        
    def _batch_decode_sequence(self,input_seq):
        number_samples = input_seq[0].shape[0]

        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((number_samples, 1, self.num_decoder_tokens))
        for i in range(number_samples):
            target_seq[i, 0, self.target_token_index['sos']] = 1.

        decoded_sentence = [[] for _ in range(number_samples)]

        stop_condition = [False for _ in range(number_samples)]
        while not all(stop_condition):

            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens,axis = 2)
            sampled_char = [self.reverse_target_char_index[index[0]] for index in sampled_token_index]

            for i in range(number_samples):

                if sampled_char[i] == 'eos':# or len(decoded_sentence[i]) >= self.max_output_length:
                    stop_condition[i] = True
                else:
                    decoded_sentence[i].append(sampled_char[i])

            target_seq = np.zeros((number_samples, 1, self.num_decoder_tokens))
            for i in range(number_samples):
                target_seq[i, 0, sampled_token_index[i]] = 1.

            states_value = [h, c]
        return decoded_sentence




