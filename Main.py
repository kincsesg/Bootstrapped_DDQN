import sys
sys.path.append('/usr/local/lib/python3.5/site-packages')
from History import *
from DQN import *
import gym as gym
import tensorflow as tf
import numpy as np
import time
import cv2
import gc  # garbage colloector
import os
import shutil

gc.enable()

params = {
    'session_name': '1', # name of the current execution
    'network_type': 'Bootstrapped-DDQN',  # learning method/network type
    'ckpt_file': None, # checkpoint file for save-restore functionality
    'save_dir': '/home/tr3zsor/Work/Dipterv2/repos/Bootstrapped_DDQN/ckpt', # checkpoint directory
    'save_fname':'model.ckpt', # checkpoint filename
    'TB_logpath': '/home/tr3zsor/Work/Dipterv2/repos/Bootstrapped_DDQN/TB_log', # path for TensorBoaard log file
    'environment': 'Breakout-v3',  # used game/environment
    'head_count': 20, # number of heads
    'head_dist': 'uniform', # the probability distribution of head choosing
    'steps_per_epoch': 100000,  # steps during an epoch
    'num_epochs': 150,  # number of epochs
    'eval_freq': 20000,  # the frequency of the evaluation in steps
    'steps_per_eval': 10000,  # steps per evaluation cycles
    'intra_epoch_log_rate': 10,  # logging frequency for stepwise statistics in steps
    'TB_count_train': 0,  # summary index for Tensorboard
    'TB_count_eval': 0,  # summary index for Tensorboard
    'copy_freq': 10000,  # frequency of network copying in steps (tragetnet <- qnet), if 0, only qnet is used
    'db_size': 1000000,  # number of observation stored in the replay memory
    'batch': 32,  # size of a batch during network training
    'num_act': 0,  # the number of possible steps in a game
    'net_train': 10,  # network retrain frequency in steps
    'eps': 1.0,  # the initial epsilon (and actual) value for epsilon greedy strategy
    'eps_min': 0.1,  # the value of epsilon cannot decreease below this value during the training
    'eps_eval': 0.05,  # the epsilon value used for evaluation
    'discount': 0.99,  # discount factor
    'lr': 0.00025,  # learning rate
    'rms_decay': 0.99,  # decay parameter for rmsprop algorithm
    'rms_eps': 1e-6, # epsilon parameter for rmsprop algorithm
    'train_start': 10000,  # the number of steps at the beginnig of the training while the agent does nothing, but collects observations through random wandering
    'img_scale': 255.0,  # the scaling (0..1) variable for grayscale pictures
    'clip_delta': 0,
    'gpu_fraction': 0.85,  # gpu memory fraction used by tensorflow
    'batch_accumulator': 'mean',  # the method for accumulation the loss for batches during the training of the network
}


class bootstrapped_DDQN:
    def __init__(self, params):
        print('Initializing Module...')
        self.params = params

        self.gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.params['gpu_fraction']))

        self.sess = tf.Session(config=self.gpu_config)
        self.DB = History(self.params)
        self.engine = gym.make(self.params['environment'])
        self.params['num_act'] = self.engine.action_space.n
        self.qnet_list = []
        self.targetnet_list = []
        self.cp_ops_list = []
        self.current_head = self.choose_head()

        # region Checking for checkpoint
        if os.path.exists(self.params['save_dir']):
            self.params['ckpt_file'] = self.params['save_dir'] + '/' + self.params['save_fname']
        # endregion

        self.training = True
        self.hist_collecting = True
        self.net_retraining = False
        self.loaded = True
        self.build_net()

        # region Initializing log variables
        # by epoch statistics
        if self.params['ckpt_file'] is None:
            self.by_ep_maxrew_tr = 0
            self.by_ep_sumrew_tr = 0
            self.by_ep_maxQ_tr = 0
            self.by_ep_sumQ_tr = 0
            self.by_ep_maxloss_tr = 0
            self.by_ep_sumloss_tr = 0
            self.by_ep_maxrew_ev = 0
            self.by_ep_sumrew_ev = 0
            self.by_ep_maxQ_ev = 0
            self.by_ep_sumQ_ev = 0

        # final statistics
        self.total_maxrew_tr = 0
        self.total_sumrew_tr = 0
        self.total_maxQ_tr = 0
        self.total_sumQ_tr = 0
        self.total_maxloss_tr = 0
        self.total_sumloss_tr = 0
        self.total_maxrew_ev = 0
        self.total_sumrew_ev = 0
        self.total_maxQ_ev = 0
        self.total_sumQ_ev = 0

        self.reset_statistics('all')
        self.train_cnt = 0
        # endregion

        # region Initializing the inner variables of the algorithm
        self.state_proc = np.zeros((84, 84, 4))
        self.action = -1
        self.terminal = False
        self.reward = 0
        self.state = self.engine.reset()
        self.engine.render()
        self.state_resized = cv2.resize(self.state, (84, 110))
        self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
        self.state_gray_old = None
        self.state_proc[:, :, 3] = self.state_gray[26:110, :] / self.params['img_scale']

        self.loss = 0.0

        if self.params['ckpt_file'] is None:
            self.step = 0
            self.steps_train = 0
        self.steps_eval = 0
        # endregion

    def build_net(self):
        self.saver_dict = {}

        # region Common parts
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.increment_global_step_op = tf.assign(self.global_step, tf.add(self.global_step, 1))

        # For tensorboard
        self.intra_ep_tr_rew = tf.placeholder(tf.float32, shape=())  # input
        self.intra_ep_tr_q = tf.placeholder(tf.float32, shape=())  # input
        self.intra_ep_tr_loss = tf.placeholder(tf.float32, shape=())  # input
        self.intra_ep_ev_rew = tf.placeholder(tf.float32, shape=())  # input
        self.intra_ep_ev_q = tf.placeholder(tf.float32, shape=())  # input

        self.sum_tr_rew = tf.summary.scalar('Train_reward', self.intra_ep_tr_rew)
        self.sum_tr_q = tf.summary.scalar('Train_Q', self.intra_ep_tr_q)
        self.sum_tr_loss = tf.summary.scalar('Train_loss', self.intra_ep_tr_loss)
        self.sum_ev_rew = tf.summary.scalar('Eval_reward', self.intra_ep_ev_rew)
        self.sum_ev_q = tf.summary.scalar('Eval_Q', self.intra_ep_ev_q)

        self.merged_train = tf.summary.merge([self.sum_tr_rew, self.sum_tr_q, self.sum_tr_loss])
        self.merged_eval = tf.summary.merge([self.sum_ev_rew, self.sum_ev_q])
        # endregion

        # region Qnet body
        self.network_name = 'qnet'

        self.x_qnet = tf.placeholder('float32', [None, 84, 84, 4], name=self.network_name + '_x')  # input
        self.q_t_qnet = tf.placeholder('float32', [None], name=self.network_name + '_q_t')  # input
        self.actions_qnet = tf.placeholder('float32', [None, params['num_act']], name=self.network_name + '_actions')  # input
        self.rewards_qnet = tf.placeholder('float32', [None], name=self.network_name + '_rewards')  # input
        self.terminals_qnet = tf.placeholder('float32', [None], name=self.network_name + '_terminals')  # input

        # conv1
        layer_name = 'conv1'
        size = 8
        channels = 4
        filters = 32
        stride = 4
        self.w1_qnet = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b1_qnet = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c1_qnet = tf.nn.conv2d(self.x_qnet, self.w1_qnet, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o1_qnet = tf.nn.relu(tf.add(self.c1_qnet, self.b1_qnet), name=self.network_name + '_' + layer_name + '_activations')

        # conv2
        layer_name = 'conv2'
        size = 4
        channels = 32
        filters = 64
        stride = 2
        self.w2_qnet = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b2_qnet = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c2_qnet = tf.nn.conv2d(self.o1_qnet, self.w2_qnet, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o2_qnet = tf.nn.relu(tf.add(self.c2_qnet, self.b2_qnet), name=self.network_name + '_' + layer_name + '_activations')

        # conv3
        layer_name = 'conv3'
        size = 3
        channels = 64
        filters = 64
        stride = 1
        self.w3_qnet = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b3_qnet = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c3_qnet = tf.nn.conv2d(self.o2_qnet, self.w3_qnet, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o3_qnet = tf.nn.relu(tf.add(self.c3_qnet, self.b3_qnet), name=self.network_name + '_' + layer_name + '_activations')
        # endregion

        # region Targetnet body
        self.network_name = 'targetnet'

        self.x_targetnet = tf.placeholder('float32', [None, 84, 84, 4], name=self.network_name + '_x')  # input
        self.q_t_targetnet = tf.placeholder('float32', [None], name=self.network_name + '_q_t')  # input
        self.actions_targetnet = tf.placeholder('float32', [None, params['num_act']], name=self.network_name + '_actions')  # input
        self.rewards_targetnet = tf.placeholder('float32', [None], name=self.network_name + '_rewards')  # input
        self.terminals_targetnet = tf.placeholder('float32', [None], name=self.network_name + '_terminals')  # input

        # conv1
        layer_name = 'conv1'
        size = 8
        channels = 4
        filters = 32
        stride = 4
        self.w1_targetnet = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b1_targetnet = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c1_targetnet = tf.nn.conv2d(self.x_targetnet, self.w1_targetnet, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o1_targetnet = tf.nn.relu(tf.add(self.c1_targetnet, self.b1_targetnet), name=self.network_name + '_' + layer_name + '_activations')

        # conv2
        layer_name = 'conv2'
        size = 4
        channels = 32
        filters = 64
        stride = 2
        self.w2_targetnet = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b2_targetnet = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c2_targetnet = tf.nn.conv2d(self.o1_targetnet, self.w2_targetnet, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o2_targetnet = tf.nn.relu(tf.add(self.c2_targetnet, self.b2_targetnet), name=self.network_name + '_' + layer_name + '_activations')

        # conv3
        layer_name = 'conv3'
        size = 3
        channels = 64
        filters = 64
        stride = 1
        self.w3_targetnet = tf.Variable(tf.random_normal([size, size, channels, filters], stddev=0.01), name=self.network_name + '_' + layer_name + '_weights')
        self.b3_targetnet = tf.Variable(tf.constant(0.1, shape=[filters]), name=self.network_name + '_' + layer_name + '_biases')
        self.c3_targetnet = tf.nn.conv2d(self.o2_targetnet, self.w3_targetnet, strides=[1, stride, stride, 1], padding='VALID', name=self.network_name + '_' + layer_name + '_convs')
        self.o3_targetnet = tf.nn.relu(tf.add(self.c3_targetnet, self.b3_targetnet), name=self.network_name + '_' + layer_name + '_activations')
        # endregion

        self.saver_dict.update({
            'qw1': self.w1_qnet, 'qb1': self.b1_qnet,
            'qw2': self.w2_qnet, 'qb2': self.b2_qnet,
            'qw3': self.w3_qnet, 'qb3': self.b3_qnet,
            'step': self.global_step
        })

        for i in range(self.params['head_count']):
            print('Building heads of QNet and targetnet ' + str(i) + '...')
            self.qnet_list.append(DQN(self.params, 'qnet', bootstrap_index=i,last_layer_out=self.o3_qnet,rews=self.rewards_qnet,terms=self.terminals_qnet,q_t=self.q_t_qnet,acts=self.actions_qnet))
            self.targetnet_list.append(DQN(self.params, 'targetnet', bootstrap_index=i,last_layer_out=self.o3_targetnet,rews=self.rewards_targetnet,terms=self.terminals_targetnet,q_t=self.q_t_targetnet,acts=self.actions_targetnet))
            self.saver_dict.update({
                'qw4_' + str(i): self.qnet_list[i].w4, 'qb4_' + str(i): self.qnet_list[i].b4,
                'qw5_' + str(i): self.qnet_list[i].w5, 'qb5_' + str(i): self.qnet_list[i].b5,
                'tw4_' + str(i): self.targetnet_list[i].w4, 'tb4_' + str(i): self.targetnet_list[i].b4,
                'tw5_' + str(i): self.targetnet_list[i].w5, 'tb5_' + str(i): self.targetnet_list[i].b5
            })
            self.cp_ops_list.append([
                self.w1_targetnet.assign(self.w1_qnet), self.b1_targetnet.assign(self.b1_qnet),
                self.w2_targetnet.assign(self.w2_qnet), self.b2_targetnet.assign(self.b2_qnet),
                self.w3_targetnet.assign(self.w3_qnet), self.b3_targetnet.assign(self.b3_qnet),
                self.targetnet_list[i].w4.assign(self.qnet_list[i].w4), self.targetnet_list[i].b4.assign(self.qnet_list[i].b4),
                self.targetnet_list[i].w5.assign(self.qnet_list[i].w5), self.targetnet_list[i].b5.assign(self.qnet_list[i].b5)
            ])
        self.TB_writer = tf.summary.FileWriter(self.params['TB_logpath'], graph=tf.get_default_graph())
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(self.saver_dict)

        for i in range(self.params['head_count']):
            self.sess.run(self.cp_ops_list[i])

        if self.params['ckpt_file'] is not None:
            print('\x1b[1;30;41m RUN LOAD \x1b[0m')
            self.load()

        print('Networks had been built!')
        sys.stdout.flush()

    def start(self):

        # region Creating tables for logging
        # intra epoch tables
        if self.params['ckpt_file'] is not None:
            try:
                self.log_intra_epoch_train = open('intra_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_intra_epoch_train = open('intra_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_intra_epoch_train.write('epoch,step,reward,Q,time\n')
            try:
                self.log_intra_epoch_eval = open('intra_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_intra_epoch_eval = open('intra_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_intra_epoch_eval.write('epoch,eval_step,reward,Q,time\n')
        else:
            self.log_intra_epoch_train = open('intra_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_intra_epoch_train.write('epoch,step,reward,Q,time\n')

            self.log_intra_epoch_eval = open('intra_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_intra_epoch_eval.write('epoch,eval_step,reward,Q,time\n')

        # by epoch tables
        if self.params['ckpt_file'] is not None:
            try:
                self.log_by_epoch_train = open('by_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_by_epoch_train = open('by_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_by_epoch_train.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxloss,sumloss,avgloss,time\n')
            try:
                self.log_by_epoch_eval = open('by_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_by_epoch_eval = open('by_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_by_epoch_eval.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')
        else:
            self.log_by_epoch_train = open('by_epoch_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_by_epoch_train.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxloss,sumloss,avgloss,time\n')

            self.log_by_epoch_eval = open('by_epoch_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_by_epoch_eval.write('epoch,step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')

        # final tables
        if self.params['ckpt_file'] is not None:
            try:
                self.log_total_train = open('total_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'a')
            except:
                self.log_total_train = open('total_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_total_train.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxloss,sumloss,avgloss,time\n')
            try:
                self.log_total_eval = open('total_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'e')
            except:
                self.log_total_eval = open('total_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
                self.log_total_eval.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')
        else:
            self.log_total_train = open('total_train_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_total_train.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,maxloss,sumloss,avgloss,time\n')

            self.log_total_eval = open('total_eval_' + self.params['network_type'] + '_' + self.params['session_name'] + '.csv', 'w')
            self.log_total_eval.write('epochs,steps_per_epoch,hist_size,tr_step,eval_step,maxrew,sumrew,avgrew,maxQ,sumQ,avgQ,time\n')
        # endregion

        self.start_time = time.time()
        print(self.params)
        print('Start training!')
        print('Collecting replay memory for ' + str(self.params['train_start']) + ' steps')

        # the algorithm itself is in this cycle
        while self.step <= self.params['steps_per_epoch'] * self.params['num_epochs']:

            # region Recalculating cycle_state variables, and reset step vars
            if self.DB.get_size() == self.params['train_start']:
                self.hist_collecting = False

            if self.training and self.loaded and not self.hist_collecting and self.steps_train > 0 and self.steps_train % self.params['net_train'] == 0:
                self.net_retraining = True
            else:
                self.net_retraining = False

            if self.training and self.loaded and not self.hist_collecting and self.steps_train > 0 and self.steps_train % self.params['eval_freq'] == 0:
                self.training = False
                self.steps_eval = 0

            if not self.training and not self.hist_collecting and self.steps_eval > 0 and self.steps_eval % self.params['steps_per_eval'] == 0:
                self.training = True
                self.save()

            if self.training and self.steps_train > 0 and self.steps_train % self.params['steps_per_epoch'] == 0:
                self.write_log_train('epoch_end')
                self.write_log_eval('epoch_end')
                self.reset_statistics('epoch_end')
                self.steps_train = 0
                self.current_head = self.choose_head()

            if self.training and self.step > 0 and self.step % (self.params['steps_per_epoch'] * self.params['num_epochs']) == 0:
                self.write_log_train('final')
                self.write_log_eval('final')
                break

            if not self.loaded and not self.hist_collecting:
                self.loaded = True
            # endregion

            # region Increasing step variables
            if self.training:
                if not self.hist_collecting:
                    self.step += 1
                    self.steps_train += 1
            else:
                if not self.net_retraining:
                    self.steps_eval += 1
            # endregion

            # region Printing actual state
            if self.hist_collecting:
                print('History size: ' + str(self.DB.get_size()))
            if self.training and not self.hist_collecting:
                print('Epoch: ' + str(self.step // self.params['steps_per_epoch'] + 1) + ' Training step: ' + str(self.steps_train))
            if not self.training and not self.hist_collecting:
                print('Epoch: ' + str(self.step // self.params['steps_per_epoch'] + 1) + ' Eval. step: ' + str(self.steps_eval))
            sys.stdout.flush()
            # endregion


            if self.training or self.net_retraining:

                # region Stores the current [state,reward,action,terminal] tuple to the history
                if self.state_gray_old is not None and self.training:
                    self.DB.insert(self.state_gray_old[26:110, :], self.reward_scaled, self.action_idx, self.terminal, self.choose_head())
                # endregion

                if self.hist_collecting:

                    # region Set epsilon for history collection
                    self.params['eps'] = 1.0
                    # endregion

                else:

                    # region Copy network at every <copy_freq> steps
                    if self.params['copy_freq'] > 0 and self.step % self.params['copy_freq'] == 0:
                        print('&&& Copying Qnet to targetnet\n')
                        self.sess.run(self.cp_ops_list[self.current_head])
                    # endregion

                    if self.net_retraining:

                        # region Retrain network
                        bat_s, bat_a, bat_t, bat_n, bat_r = self.DB.get_batches(self.current_head)
                        bat_a = self.get_onehot(bat_a)

                        if self.params['copy_freq'] > 0:  # get predictions from the network for the input batches
                            feed_dict = {self.x_qnet: bat_n}
                            q_tmp = self.sess.run(self.qnet_list[self.current_head].y, feed_dict=feed_dict)
                            a_nextbest = np.argmax(q_tmp, axis=1)  # best action in the next state given by the q-network

                            feed_dict = {self.x_targetnet: bat_n}
                            q_t = self.sess.run(self.targetnet_list[self.current_head].y, feed_dict=feed_dict)
                        else:
                            feed_dict = {self.x_qnet: bat_n}
                            q_tmp = self.sess.run(self.qnet_list[self.current_head].y, feed_dict=feed_dict)
                            a_nextbest = np.argmax(q_tmp, axis=1)  # best action in the next state given by the q-network

                            feed_dict = {self.x_qnet: bat_n}
                            q_t = self.sess.run(self.qnet_list[self.current_head].y, feed_dict=feed_dict)

                        q_t = q_t[range(len(q_t)), a_nextbest]  # value of the action given by the q-network calculated by the target network

                        feed_dict = {self.x_qnet: bat_s, self.q_t_qnet: q_t, self.actions_qnet: bat_a, self.terminals_qnet: bat_t, self.rewards_qnet: bat_r}
                        _, self.loss = self.sess.run([self.qnet_list[self.current_head].rmsprop, self.qnet_list[self.current_head].loss], feed_dict=feed_dict)  # retraining
                        self.train_cnt = self.sess.run(self.increment_global_step_op) # increase global step

                        self.by_ep_sumloss_tr += np.sqrt(self.loss)
                        self.total_sumloss_tr += np.sqrt(self.loss)

                        if self.by_ep_maxloss_tr < np.sqrt(self.loss):
                            self.by_ep_maxloss_tr = np.sqrt(self.loss)

                        if self.total_maxloss_tr < np.sqrt(self.loss):
                            self.total_maxloss_tr = np.sqrt(self.loss)
                        print('Network retrained!')
                        # endregion

                    # region Decrease epsilon for training
                    self.params['eps'] = max(self.params['eps_min'], 1.0 - (1.0 - self.params['eps_min']) * (float(self.step) + float(self.params['steps_per_epoch'])) / (float(self.params['steps_per_epoch']) * float(self.params['num_epochs'])))
                    # endregion

                    # region Log training
                    self.write_log_train('in_epoch')
                    # endregion

            else:

                # region Set epsilon for eval
                self.params['eps'] = self.params['eps_eval']
                # endregion

                # region Log evaluation
                self.write_log_eval('in_epoch')
                # endregion

            # region Reset game if a terminal state reached
            if self.terminal:
                self.reset_game()
            # endregion

            # region Chose action
            self.action_idx, self.action, self.maxQ = self.select_action(self.state_proc)

            if not self.hist_collecting:
                if self.by_ep_maxQ_tr < self.maxQ:
                    self.by_ep_maxQ_tr = self.maxQ
                if self.total_maxQ_tr < self.maxQ:
                    self.total_maxQ_tr = self.maxQ

                if self.by_ep_maxQ_ev < self.maxQ:
                    self.by_ep_maxQ_ev = self.maxQ
                if self.total_maxQ_ev < self.maxQ:
                    self.total_maxQ_ev = self.maxQ

                self.by_ep_sumQ_tr += self.maxQ
                self.total_sumQ_tr += self.maxQ

                self.by_ep_sumQ_ev += self.maxQ
                self.total_sumQ_ev += self.maxQ
            # endregion

            # region Execute action, observe environment and scale reward
            self.state, self.reward, t, _ = self.engine.step(self.action)
            extr_rew = self.extract_reward(self.state)
            self.engine.render()
            self.terminal = int(t)
            # self.reward_scaled = self.reward // max(1, abs(self.reward))
            self.reward_scaled = self.reward * extr_rew
            if abs(self.reward) > 0.0:
                print('\x1b[1;30;41m' + str(self.reward) + '\x1b[0m')
                print('\x1b[1;30;41m' + str(self.reward_scaled) + '\x1b[0m')

            if not self.hist_collecting:
                if self.by_ep_maxrew_tr < self.reward_scaled:
                    self.by_ep_maxrew_tr = self.reward_scaled
                if self.total_maxrew_tr < self.reward_scaled:
                    self.total_maxrew_tr = self.reward_scaled

                if self.by_ep_maxrew_ev < self.reward_scaled:
                    self.by_ep_maxrew_ev = self.reward_scaled
                if self.total_maxrew_ev < self.reward_scaled:
                    self.total_maxrew_ev = self.reward_scaled

                self.by_ep_sumrew_tr += self.reward_scaled
                self.total_sumrew_tr += self.reward_scaled

                self.by_ep_sumrew_ev += self.reward_scaled
                self.total_sumrew_ev += self.reward_scaled
            # endregion

            # region s <- s'
            self.state_gray_old = np.copy(self.state_gray)
            self.state_proc[:, :, 0:3] = self.state_proc[:, :, 1:4]
            self.state_resized = cv2.resize(self.state, (84, 110))
            self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
            self.state_proc[:, :, 3] = self.state_gray[26:110, :] / self.params['img_scale']
            # endregion

            # TODO : add video recording

    def reset_game(self):
        self.state_proc = np.zeros((84, 84, 4))
        self.action = -1
        self.terminal = False
        self.reward = 0
        self.state = self.engine.reset()
        self.engine.render()
        self.state_resized = cv2.resize(self.state, (84, 110))
        self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
        self.state_gray_old = None
        self.state_proc[:, :, 3] = self.state_gray[26:110, :] / self.params['img_scale']

    def reset_statistics(self, state):
        if state == 'epoch_end':
            self.by_ep_maxrew_tr = 0
            self.by_ep_sumrew_tr = 0
            self.by_ep_maxQ_tr = 0
            self.by_ep_sumQ_tr = 0
            self.by_ep_maxloss_tr = 0
            self.by_ep_sumloss_tr = 0
            self.by_ep_maxrew_ev = 0
            self.by_ep_sumrew_ev = 0
            self.by_ep_maxQ_ev = 0
            self.by_ep_sumQ_ev = 0

    def select_action(self, st):
        if np.random.rand() > self.params['eps']:
            # greedy with random tie-breaking
            Q_pred = self.sess.run(self.qnet_list[self.current_head].y, feed_dict={self.x_qnet: np.reshape(st, (1, 84, 84, 4))})[0]
            a_winner = np.argwhere(Q_pred == np.amax(Q_pred))
            if len(a_winner) > 1:
                act_idx = a_winner[np.random.randint(0, len(a_winner))][0]
                return act_idx, act_idx, np.amax(Q_pred)
            else:
                act_idx = a_winner[0][0]
                return act_idx, act_idx, np.amax(Q_pred)
        else:
            # random
            act_idx = np.random.randint(0, self.engine.action_space.n)
            Q_pred = self.sess.run(self.qnet_list[self.current_head].y, feed_dict={self.x_qnet: np.reshape(st, (1, 84, 84, 4))})[0]
            return act_idx, act_idx, Q_pred[act_idx]

    def get_onehot(self, actions):
        actions_onehot = np.zeros((self.params['batch'], self.params['num_act']))

        for i in range(self.params['batch']):
            actions_onehot[i, int(actions[i])] = 1

        return actions_onehot

    def choose_head(self):
        head_num = -1

        if (self.params['head_dist'] == 'uniform'):
            head_num = np.random.randint(self.params['head_count'])

        return head_num

    def extract_reward(self, state):
        if (self.params['environment'] == 'Breakout-v3'):
            state_resized = cv2.resize(state, (84, 110))
            state_gray = cv2.cvtColor(state_resized, cv2.COLOR_BGR2GRAY)
            extract_source = state_gray[30:49, 4:80]
            return 1444 - np.count_nonzero(extract_source)
        else:
            return 1

    def save(self):
        directory = self.params['save_dir']
        filename = self.params['save_fname']
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('Saving checkpoint : ' + directory + '/' + filename)
        self.saver.save(self.sess, directory + '/' + filename)
        sys.stdout.write('$$$ Model saved : %s\n\n' % str(directory + '/' + filename))
        sys.stdout.flush()

        self.saved_stat = open(directory + '/model.csv', 'w')
        self.saved_stat.write('by_ep_maxrew_tr,by_ep_sumrew_tr,by_ep_maxQ_tr,by_ep_sumQ_tr,by_ep_maxloss_tr,by_ep_sumloss_tr,by_ep_maxrew_ev,'
                              'by_ep_sumrew_ev,by_ep_maxQ_ev,by_ep_sumQ_ev\n')
        self.saved_stat.write(str(self.by_ep_maxrew_tr) + ',' + str(self.by_ep_sumrew_tr) + ',' + str(self.by_ep_maxQ_tr) + ',' + str(self.by_ep_sumQ_tr) + ',' +
                              str(self.by_ep_maxloss_tr) + ',' + str(self.by_ep_sumloss_tr) + ',' + str(self.by_ep_maxrew_ev) + ',' + str(self.by_ep_sumrew_ev) + ',' +
                              str(self.by_ep_maxQ_ev) + ',' + str(self.by_ep_sumQ_ev) + '\n')
        self.saved_stat.close()

    def load(self):
        print('Loading checkpoint : ' + self.params['ckpt_file'])
        self.saver.restore(self.sess, self.params['ckpt_file'])
        temp_train_cnt = self.sess.run(self.global_step)
        self.step = temp_train_cnt * self.params['net_train']
        self.steps_train = self.step % self.params['steps_per_epoch']
        self.steps_eval = self.params['steps_per_eval']
        self.loaded = False

        self.saved_stat = open(str.split(self.params['ckpt_file'], '.')[0] + '.csv', 'r')
        self.saved_stat.readline()
        stats = self.saved_stat.readline()
        splitted_list = stats.split(',')
        self.by_ep_maxrew_tr = float(splitted_list[0])
        self.by_ep_sumrew_tr = float(splitted_list[1])
        self.by_ep_maxQ_tr = float(splitted_list[2])
        self.by_ep_sumQ_tr = float(splitted_list[3])
        self.by_ep_maxloss_tr = float(splitted_list[4])
        self.by_ep_sumloss_tr = float(splitted_list[5])
        self.by_ep_maxrew_ev = float(splitted_list[6])
        self.by_ep_sumrew_ev = float(splitted_list[7])
        self.by_ep_maxQ_ev = float(splitted_list[8])
        self.by_ep_sumQ_ev = float(splitted_list[9])

    def write_log_train(self, cycle_state):
        epoch = self.step // self.params['steps_per_epoch'] + 1
        if cycle_state == 'in_epoch':
            if self.steps_train % self.params['intra_epoch_log_rate'] == 0:
                self.log_intra_epoch_train.write(str(epoch) + ',' + str(self.steps_train) + ',' + str(self.reward_scaled) + ',' +
                                                 str(self.maxQ) + ',' + str(time.time()) + '\n')
                self.log_intra_epoch_train.flush()

                # for Tensorboard
                feed_dict = {self.intra_ep_tr_rew: self.reward_scaled, self.intra_ep_tr_q: self.maxQ, self.intra_ep_tr_loss: self.loss}

                summary = self.sess.run(self.merged_train, feed_dict=feed_dict)
                self.TB_writer.add_summary(summary, self.params['TB_count_train'])
                self.params['TB_count_train'] += 1

            sys.stdout.write('Epoch : %d , Step : %d , Reward : %f, Q : %.3f , Time : %.1f\n' % (epoch, self.step, self.reward_scaled, self.maxQ, time.time()))
            sys.stdout.flush()

        if cycle_state == 'epoch_end':
            self.log_by_epoch_train.write(str(epoch) + ',' + str(self.step) + ',' +
                                          str(self.by_ep_maxrew_tr) + ',' + str(self.by_ep_sumrew_tr) + ',' + str(self.by_ep_sumrew_tr / self.params['steps_per_epoch']) + ',' +
                                          str(self.by_ep_maxQ_tr) + ',' + str(self.by_ep_sumQ_tr) + ',' + str(self.by_ep_sumQ_tr / self.params['steps_per_epoch']) + ',' +
                                          str(self.by_ep_maxloss_tr) + ',' + str(self.by_ep_sumloss_tr) + ',' + str(self.by_ep_sumloss_tr / self.params['steps_per_epoch']) + ',' +
                                          str(time.time()) + '\n')
            self.log_by_epoch_train.flush()

        if cycle_state == 'final':
            self.log_total_train.write(str(epoch) + ',' + str(self.params['steps_per_epoch']) + ',' + str(self.params['db_size']) + ',' +
                                       str(self.params['eval_freq']) + ',' + str(self.params['steps_per_eval']) + ',' +
                                       str(self.total_maxrew_tr) + ',' + str(self.total_sumrew_tr) + ',' + str(
                self.total_sumrew_tr / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(self.total_maxQ_tr) + ',' + str(self.total_sumQ_tr) + ',' + str(self.total_sumQ_tr / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(self.total_maxloss_tr) + ',' + str(self.total_sumloss_tr) + ',' + str(
                self.total_sumloss_tr / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                       str(time.time()) + '\n')
            self.log_total_train.flush()

    def write_log_eval(self, cycle_state):
        epoch = self.step // self.params['steps_per_epoch'] + 1
        if cycle_state == 'in_epoch':
            if self.steps_eval % self.params['intra_epoch_log_rate'] == 0:
                self.log_intra_epoch_eval.write(str(epoch) + ',' + str(self.steps_eval) + ',' + str(self.reward_scaled) + ',' +
                                                str(self.maxQ) + ',' + str(time.time()) + '\n')
                self.log_intra_epoch_eval.flush()

                # for Tensorboard
                feed_dict = {self.intra_ep_ev_rew: self.reward_scaled, self.intra_ep_ev_q: self.maxQ}

                summary = self.sess.run(self.merged_eval, feed_dict=feed_dict)
                self.TB_writer.add_summary(summary, self.params['TB_count_eval'])
                self.params['TB_count_eval'] += 1

            sys.stdout.write(
                'Epoch : %d , Eval.Step : %d , Reward : %f, Q : %.3f , Time : %.1f\n' % (epoch, self.steps_eval, self.reward_scaled, self.maxQ, time.time()))
            sys.stdout.flush()

        if cycle_state == 'epoch_end':
            self.log_by_epoch_eval.write(str(epoch) + ',' + str(self.step) + ',' +
                                         str(self.by_ep_maxrew_ev) + ',' + str(self.by_ep_sumrew_ev) + ',' + str(self.by_ep_sumrew_ev / self.params['steps_per_epoch']) + ',' +
                                         str(self.by_ep_maxQ_ev) + ',' + str(self.by_ep_sumQ_ev) + ',' + str(self.by_ep_sumQ_ev / self.params['steps_per_epoch']) + ',' +
                                         str(time.time()) + '\n')
            self.log_by_epoch_eval.flush()

        if cycle_state == 'final':
            self.log_total_eval.write(str(epoch) + ',' + str(self.params['steps_per_epoch']) + ',' + str(self.params['db_size']) + ',' +
                                      str(self.params['eval_freq']) + ',' + str(self.params['steps_per_eval']) + ',' +
                                      str(self.total_maxrew_ev) + ',' + str(self.total_sumrew_ev) + ',' + str(
                self.total_sumrew_ev / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                      str(self.total_maxQ_ev) + ',' + str(self.total_sumQ_ev) + ',' + str(self.total_sumQ_ev / (self.params['steps_per_epoch'] * self.params['num_epochs'])) + ',' +
                                      str(time.time()) + '\n')
            self.log_total_eval.flush()


# entry point
if __name__ == "__main__":
    bootstrapped_ddqn = bootstrapped_DDQN(params)
    bootstrapped_ddqn.start()