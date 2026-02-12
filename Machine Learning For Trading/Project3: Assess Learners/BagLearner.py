import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost= False, verbose= False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags,
        self.boost = boost
        self.verbose = verbose


def add_evidence(self, train_x, train_y):
    self.learners = []
    n = train_x.shape[0]
    for i in range(0, self.bags):
        indices = np.random.choice(n, size=n, replace=True)

        random_sample_x = train_x[indices]
        random_sample_y = train_y[indices]

        learner = self.learner(**self.kwargs)
        learner.add_evidence(random_sample_x, random_sample_y)
        self.learners.append(learner)


