import numpy as np

class BagLearner(object):

    #store all the arguments needed for baglearner
    def __init__(self, learner, kwargs, bags, boost= False, verbose= False):
        #store learner to self for use in next function
        self.learner = learner
        #store kwargs to self for use in next function
        self.kwargs = kwargs
        #store num of bags to self for use in next function
        self.bags = bags
        #store boost to self for use in next function
        self.boost = boost
        #store verbose to self for use in next function
        self.verbose = verbose


    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "mmiah32"  # replace tb34 with your Georgia Tech username

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return "904188350"

    def study_group(self):
        return 'mmiah32', 'discord channel'

    def add_evidence(self, train_x, train_y):
        #to hold each learner
        self.learners = []
        n = train_x.shape[0]
        for i in range(0, self.bags):
            #choose rows at random with replacement
            indices = np.random.choice(n, size=n, replace=True)

            #train features with above random indices
            random_sample_x = train_x[indices]
            #choose same rows for y data
            random_sample_y = train_y[indices]

            #store learner and pass in learner arguments
            learner = self.learner(**self.kwargs)
            #trainer learner on data
            learner.add_evidence(random_sample_x, random_sample_y)
            #append to learners array to hold
            self.learners.append(learner)


    def query(self, points):
        #create outputs for y values
        y_outputs = []
        #query using test points
        for learner in self.learners:
            #append outcome to y_ouputs
            y_outputs.append(learner.query(points))
        #convert to np array
        predictions = np.array(y_outputs)
        #take mean across each datapoint (column axis)
        mean_predicts = np.mean(predictions, axis = 0)

        return mean_predicts

