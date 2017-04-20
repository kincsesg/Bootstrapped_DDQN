import numpy as np


class History:
    def __init__(self, params):
        self.size = params['db_size']
        self.img_scale = params['img_scale']
        self.states = np.zeros([self.size, 84, 84], dtype='uint8')  # image dimensions
        self.actions = np.zeros(self.size, dtype='float32')
        self.terminals = np.zeros(self.size, dtype='float32')
        self.rewards = np.zeros(self.size, dtype='float32')
        self.indecies = np.zeros(self.size, dtype='float32')
        self.bat_size = params['batch']
        self.bat_s = np.zeros([self.bat_size, 84, 84, 4])
        self.bat_a = np.zeros([self.bat_size])
        self.bat_t = np.zeros([self.bat_size])
        self.bat_n = np.zeros([self.bat_size, 84, 84, 4])
        self.bat_r = np.zeros([self.bat_size])

        self.counter = 0  # keep track of next empty state
        self.flag = False
        return

    def get_batches(self, bootstrap_index):
        obs_head_i = np.where(self.indecies == bootstrap_index)[0] # observation indecies for the current head

        for i in range(self.bat_size):
            idx = 0
            while idx < 3 or (idx > len(obs_head_i) - 2 and idx < len(obs_head_i) + 3):
                idx = np.random.randint(3, len(obs_head_i) - 1)
            self.bat_s[i] = np.transpose(self.states[obs_head_i[idx - 3:idx + 1], :, :], (1, 2, 0)) / self.img_scale
            self.bat_n[i] = np.transpose(self.states[obs_head_i[idx - 2:idx + 2], :, :], (1, 2, 0)) / self.img_scale
            self.bat_a[i] = self.actions[obs_head_i[idx]]
            self.bat_t[i] = self.terminals[obs_head_i[idx]]
            self.bat_r[i] = self.rewards[obs_head_i[idx]]

        return self.bat_s, self.bat_a, self.bat_t, self.bat_n, self.bat_r

    def insert(self, prevstate_proc, reward, action, terminal, bootstrap_index):
        self.states[self.counter] = prevstate_proc
        self.rewards[self.counter] = reward
        self.actions[self.counter] = action
        self.terminals[self.counter] = terminal
        self.indecies[self.counter] = bootstrap_index
        # update counter
        self.counter += 1
        if self.counter >= self.size:
            self.flag = True
            self.counter = 0
        return

    def get_size(self):
        if self.flag == False:
            return self.counter
        else:
            return self.size