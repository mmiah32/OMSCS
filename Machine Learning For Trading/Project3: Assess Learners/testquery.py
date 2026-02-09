import numpy as np
from DTLearner import DTLearner

# Create a learner
learner = DTLearner()

# Manually build a simple tree
# Row 0: split on feature 0 at value 25
# Row 1: left leaf predicting 6.5
# Row 2: right leaf predicting 13.5
learner.tree = np.array([
    [0, 25, 1, 2],
    [-1, 6.5, -999, -999],
    [-1, 13.5, -999, -999]
])

# Create test points
test_points = np.array([[15], [35], [20]])

# Make predictions
predictions = learner.query(test_points)

print("Test points:", test_points.flatten())
print("Predictions:", predictions)
print("Expected: [6.5, 13.5, 6.5]")