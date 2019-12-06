import numpy as np


class MyExplorer:
    def __init__(self, action_space, exploring_rate=0.1):
        self.exploring_rate_ = exploring_rate
        self.action_space_ = action_space

    def get_action(self, action, switch=1):
        value = np.random.random()

        if value > self.exploring_rate_ and switch:
            random_value = [np.random.random() for x in range(len(action))]
            index = 0
            new_action = []
            for eachsize in self.action_space_:
                tmpval = [action[x + index]+random_value[x + index]
                          for x in range(eachsize)]
                the_sum = sum(tmpval)

                new_action += [round(x/the_sum, 4) for x in tmpval]
                index += eachsize
            return new_action
        return action


if __name__ == "__main__":
    shape = [1, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 2, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3]
    explore = MyExplorer(shape)
    action_ = []
    mixed_action_ = []
    for item in shape:
        action_ += [round(1, 3)/item for x in range(item)]
    print("default: ", action_)

    for x in range(1000):
        # print(np.random.random())
        nac_ = explore.get_action(action_)
        print(nac_[1])
