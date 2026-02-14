import numpy as np
from BagLearner import BagLearner
from LinRegLearner import LinRegLearner

class InsaneLearner(object):
    #only add in self and verbose since
    #baglearner is hardcoded
    def __init__(self, verbose = False):
        #store verbose inside self
        self.verbose = verbose

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return 'mmiah32'


    def add_evidence(self, train_x, train_y):
        #learners array to hold each bag learner instance
        self.learners = []
        #20 bag learners
        for i in range(20):
            #create bag learner instance which will be created 20 times since loop
            bag = BagLearner(learner=LinRegLearner, kwargs={"verbose": False},
                bags=20, boost=False, verbose=self.verbose)
            #call bag learners own add_evidence function to train model on data
            bag.add_evidence(train_x, train_y)
            #append each baglearner model to learners array
            self.learners.append(bag)

    def query(self, points):
        #for each learner, pass in data points for querying
        predictions = [learner.query(points) for learner in self.learners]
        #get mean across each data point along the columns
        mean_predicts = np.mean(predictions, axis = 0)
        return mean_predicts


