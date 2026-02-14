""""""
"""  		  	   		 		  			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 		  			  		 			     			  	 

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 		  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 		  			  		 			     			  	 
All Rights Reserved  		  	   		 		  			  		 			     			  	 

Template code for CS 4646/7646  		  	   		 		  			  		 			     			  	 

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 		  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 		  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 		  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 		  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 		  			  		 			     			  	 
or edited.  		  	   		 		  			  		 			     			  	 

We do grant permission to share solutions privately with non-students such  		  	   		 		  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 		  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 		  			  		 			     			  	 
GT honor code violation.  		  	   		 		  			  		 			     			  	 

-----do not edit anything above this line---  		  	   		 		  			  		 			     			  	 
"""

import numpy as np


class DTLearner(object):
    """
    This is a Regression Decision Tree Learner.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size = 1, verbose=False):
        """
        Constructor method
        """
        #store leaf size into self for use in next function
        self.leaf_size = leaf_size
        #store verbose into self for use in next function
        self.verbose = verbose
        #store tree into self for use in next function
        self.tree = None

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
        return 904188350

    def study_group(self):
        return 'mmiah32', 'discord channel'

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        #take feature predictors (data_x)
        #take dependent variable (data_y)
        #combine into one dataset
        data = np.column_stack((data_x, data_y))
        #build tree using data
        self.tree = self.build_tree(data, self.leaf_size)



    def build_tree(self, data, leaf_size):
        #first base case
        if data.shape[0] <= leaf_size:
            #if less nodes (rows) than leaf size
            #take mean of all nodes (rows) y values
            #to account for all y values
            #-1 to represent leaf
            #no offset since no subtrees
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
        #second base case
        elif data.shape[0] == 1:
            #if only one row (node) is passed
            #turn into a leaf, return y value
            #no offset since no subtrees
            return np.array([[-1, data[0, -1], np.nan, np.nan]])
        #third base case
        elif np.all(data[:, -1] == data[0, -1]):
            #if all y values are the same
            #turn to leaf, take value of
            #first row, since all y values
            #are the same
            #no offset since no subtrees
            return np.array([[-1, data[0, -1], np.nan, np.nan]])

        #set abs value of correlation of first feature
        max_corr = np.abs(np.corrcoef(data[:, 0], data[:, -1])[0, 1])
        #take index of first feature
        feature_index = 0
        #loop across columns except last column (y)
        for i in range(data.shape[1] - 1):
            #if abs value of correlation of any column is greater than
            #currently set correlation
            if np.abs(np.corrcoef(data[:, i], data[:, -1])[0, 1]) > max_corr:
                #change max_correlation to current column
                max_corr = np.abs(np.corrcoef(data[:, i], data[:, -1])[0, 1])
                #set index of column to current index
                feature_index = i

        #set split value to median of column with highest
        #correlation
        split_val = np.median(data[:, feature_index])

        #split data to left group (left subtree)
        left_data = data[data[:, feature_index] <= split_val]
        #split data to right group (right subtree)
        right_data = data[data[:, feature_index] > split_val]

        # Handle edge case where split doesn't divide the data
        # data ends up on one side of the tree
        if left_data.shape[0] == 0 or right_data.shape[0] == 0:
            #turn to leaf
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        #build left tree using recursion
        left_tree = self.build_tree(left_data, leaf_size)
        #build right tree using recursion
        right_tree = self.build_tree(right_data, leaf_size)

        #root goes last because we must know shape of left subtree first
        #since recursive call root changes at each level of the tree
        root = np.array([[feature_index, split_val, 1, 1 + left_tree.shape[0]]])

        return np.vstack([root, left_tree, right_tree])


    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        #prefill array (technique used in martingale)
        predictions = np.zeros(points.shape[0])  # creates array of correct size

        #for the rows in points
        for i in range(points.shape[0]):
            #store node (row) for each node(row) in points
            test_point = points[i]
            #store row (node) index
            row = 0

            #while the node is not a leaf
            while (self.tree[row, 0] != -1):
                # Get which feature this node splits on
                feature_index = int(self.tree[row, 0])
                # Look up the test point's value for that feature
                feature_value = test_point[feature_index]
                # Get the split threshold for this node
                split_value = self.tree[row, 1]
                # Get navigation offsets
                left_offset = int(self.tree[row, 2])
                right_offset = int(self.tree[row, 3])

                # Decide which direction to go based on comparison
                if feature_value <= split_value:
                    row += left_offset #Go left
                else:
                    row += right_offset #Go right
            # At leaf node - store its prediction
            predictions[i] = self.tree[row, 1]
        return predictions


