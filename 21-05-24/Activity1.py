# create an iterator that returns numbers starting with 1 and each sequence will increace by 1

class x:
    def __iter__(self):
        self.n = 1
        return self

    def __next__(self):
        y = self.n
        self.n += 1
        return y


my_class = x()


iterator = iter(my_class)


print(next(iterator))
