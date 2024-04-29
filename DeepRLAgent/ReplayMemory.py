from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','label','next_label'))


class ReplayMemory(object):

    def __init__(self, capacity,mode:str='normal'):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        if mode == 'SC':
            self.Transition = namedtuple('Transition', ('done','state', 'action', 'next_state', 'reward','label','next_label'))
        else:
            self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



