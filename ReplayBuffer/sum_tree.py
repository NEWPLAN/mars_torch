import math


class SumTree(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = int(math.ceil(math.log(max_size + 1, 2)) + 1)
        self.tree_size = int(2 ** self.tree_level - 1)
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor_data = 0
        self.tree_index_base = self.max_size-1

    @property
    def root(self):
        return self.tree[0]

    def add(self, contents, value):
        index = self.cursor_data
        self.cursor_data = (self.cursor_data + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = int(2 ** (self.tree_level - 1) - 1 + index)
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = int(2 ** (self.tree_level - 1) - 1 + index)
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2 ** (self.tree_level - 1) - 1 <= index:
            return self.data[int(index - (2 ** (self.tree_level - 1) - 1))], self.tree[index], index - (
                2 ** (self.tree_level - 1) - 1)

        left = self.tree[2 * index + 1]

        if value <= left + 1e-9:
            return self._find(value, 2 * index + 1)
        else:
            return self._find(value - left, 2 * (index + 1))

    def print_tree(self):
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=' ')
            print()

        print(self.data)

    def filled_size(self):
        return self.size


if __name__ == '__main__':
    s = SumTree(10)
    for i in range(10):
        s.add(2 ** i, i)

    s.print_tree()
    for i in range(10, 15):
        s.add(2 ** i, i)

    s.print_tree()
    print(s.find(0.5))
