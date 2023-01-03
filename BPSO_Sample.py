# https://pyswarms.readthedocs.io/en/development/examples/feature_subset_selection.html#using-binary-pso
# https://pyswarms.readthedocs.io/en/development/examples/visualization.html

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score

# Import PySwarms
import pyswarms as ps

try:
    from IPython import get_ipython
    get_ipython().magic('%load_ext autoreload')    
    get_ipython().magic('%autoreload 2')    
   # get_ipython().magic('%matplotlib inline')    
except:
    pass

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=15, n_classes=3,
                           n_informative=4, n_redundant=1, n_repeated=2,
                           random_state=1)

'''
# Plot toy dataset per feature
df = pd.DataFrame(X)
df['labels'] = pd.Series(y)

sns.pairplot(df, hue='labels')
'''


from sklearn import linear_model

# Create an instance of the classifier
classifier = linear_model.LogisticRegression()

# Define objective function
def f_per_particle(m, alpha):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = 15
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j

def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)





# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

# Call instance of PSO
dimensions = 15 # dimensions should be the number of features
#optimizer.reset()   

optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
# Perform optimization
#cost, pos = optimizer.optimize(f, print_step=100, iters=1000, verbose=2)
cost, pos = optimizer.optimize(f, iters=100, n_processes=None)





# Create two instances of LogisticRegression
classfier0 = linear_model.LogisticRegression()
classfier = linear_model.LogisticRegression()

# Get the selected features from the final positions
X_selected_features = X[:,pos==1]  # subset

# Perform classification and store performance in P
c0 = classfier0.fit(X, y)
c1 = classifier.fit(X_selected_features, y)

# Compute performance
main_performance = (c0.predict(X) == y).mean()
subset_performance = (c1.predict(X_selected_features) == y).mean()

c0Pred = c0.predict(X)
accuracy = accuracy_score(y, c0Pred)
accuracy = round(accuracy, 4)
print("Main Accuracy: %.2f%%" % (accuracy * 100.0))

c1Pred = c1.predict(X_selected_features)
sbaccuracy = accuracy_score(y, c1Pred)
sbaccuracy = round(accuracy, 4)
print("Subset Accuracy: %.2f%%" % (sbaccuracy * 100.0))

print('Main performance: %.3f' % (main_performance))
print('Subset performance: %.3f' % (subset_performance))


'''
# Plot toy dataset per feature
df1 = pd.DataFrame(X_selected_features)
df1['labels'] = pd.Series(y)

sns.pairplot(df1, hue='labels')
'''





'''----- VISUALATION ------'''

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
# Set-up optimizer
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options)
optimizer.optimize(fx.sphere, iters=100)


# Plot the cost
plot_cost_history(optimizer.cost_history)
plt.show()




from pyswarms.utils.plotters.formatters import Mesher, Designer
# Plot the sphere function's mesh for better plots
m = Mesher(func=fx.sphere,
           limits=[(-1,1), (-1,1)])
# Adjust figure limits
d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)],
             label=['x-axis', 'y-axis', 'z-axis'])


plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0))


pos_history_3d = m.compute_history_3d(optimizer.pos_history) # preprocessing
animation3d = plot_surface(pos_history=pos_history_3d,
                           mesher=m, designer=d,
                           mark=(0,0,0))  
