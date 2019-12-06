from ReplayBuffer.sum_tree import SumTree
import numpy as np
import math


class PriorityBuffer():
    def __init__(self, memory_size, batch_size, alpha, mu, seed):
        """ Prioritized experience replay buffer initialization.
        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.__e = 0.01
        self.__mu = mu
        np.random.seed(seed)

    def __len__(self):
        return self.tree.filled_size()

    def add(self, data, error, gradient):
        """ Add new sample.
        Parameters
        ----------
        data : object
            new sample
        error : float
            sample's td-error
        """
        priority = self.__getPriority(error, gradient)
        if(math.isnan(np.sum(priority))):
            print("**********************")
            print("error:", error)
            print("gradient:", gradient)
            print("priority:", priority)
            input()
        self.tree.add(data, np.sum(priority))

    def __getPriority(self, error, gradient):
        priority = self.__mu * \
            np.array(error) + (1 - self.__mu) * np.array(gradient)
        idx = np.where(priority < 0.)
        if len(idx[0]) != 0:
            for i in idx[0]:
                priority[i] = 0.
        return (priority + self.__e) ** self.alpha

    def select(self, beta):
        """ The method return samples randomly.
        Parameters
        ----------
        beta : float
        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < self.batch_size:
            print('LESS and LESS')
            return None, None, None

        out = []
        indices = []
        weights = []

        segment = self.tree.root / self.batch_size

        for i in range(self.batch_size):
            min_val = segment * i
            max_val = segment * (i + 1)
            r = np.random.uniform(min_val, max_val)
            data, priority, index = self.tree.find(r, norm=False)

            weights.append((1. / self.memory_size / priority)
                           ** beta if priority > 1e-16 else 0.)
            indices.append(index)
            out.append(data)

        weights /= max(weights)  # Normalize for stability

        return out, weights, indices

    def priority_update(self, indices, error, gradient):
        """ The methods update samples's priority.
        Parameters
        ----------
        indices :
            list of sample indices
        """
        priorities = self.__getPriority(error, gradient)
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.
        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [(self.tree.get_val(i) + self.__e) ** -
                      old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)
