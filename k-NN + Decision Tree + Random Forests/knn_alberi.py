########################################
# Imports
########################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml as get_dataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from tqdm import tqdm_notebook

plt.style.use('seaborn')

########################################
# Dataset
########################################

# Load the dataset
x, y = get_dataset('mnist_784', version=1, return_X_y=True)

# Function for splitting the dataset
def split_data(x, y, train_ratio=0.8, seed=None):
    idx_samples = np.arange(len(x), dtype=np.int32)
    if seed is not None:
        np.random.seed(seed)

    ### HERE YOUR CODE ###

    # Shuffle the idx samples
    np.random.shuffle(idx_samples)
    # compute the train set size
    train_size = int(np.ceil(len(x) * train_ratio))

    # Split the idx into train and validation idx
    idx_train, idx_valid = idx_samples[:train_size], idx_samples[train_size:]
    ### END CODE ###

    return x[idx_train], y[idx_train], x[idx_valid], y[idx_valid]

# Let's work with a subset of the dataset, only for reducing time execution
x, y = x[:2000], y[:2000]

# Split data into train and test sets
x_train, y_train, x_test, y_test = split_data(x, y, train_ratio=0.8,
                                                seed=1234)

# Process the dataset
x_train, x_test = x_train / 255, x_test / 255 # input in range [0, 1]
y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

# Print summary of the dataset
print('################################################################################')
print('# SUMMARY')
print('################################################################################\n')
print('Train set size:', x_train.shape[0])
print('Test set size:', x_test.shape[0])
print('Number of pixels per image:', x_train.shape[1], '= 28x28')
print()

# Plot some samples of the dataset
print('################################################################################')
print('# SHOW SAMPLES')
print('################################################################################\n')
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.title(f'Label: {y_train[i]}', fontweight='bold')
    plt.imshow(x_train[i].reshape(-1, 28), cmap='gray')
    plt.axis('off')
plt.show()


########################################
# k-Fold Cross Validation
########################################

def k_fold_cv(x, y, k=5, seed=None):
    '''
    input x: input samples ndarray of shape (num_samples, feat_dim)
    input y: labels ndarray of shape (num_samples)
    input k: number of folds
    input seed: seed for random shuffle
    '''
    idx_samples = np.arange(len(x), dtype=np.int32)
    if seed is not None:
        np.random.seed(seed)

    ### HERE YOUR CODE ###

    # Shuffle the samples indices
    np.random.shuffle(idx_samples)
    # Split the idx samples into k-folds
    idx_sample_folds = np.split(idx_samples, k)
    ### END CODE ###

    x_train_folds, y_train_folds = [], []
    x_valid_folds, y_valid_folds = [], []
    for idx_k in range(k):
        idx_train, idx_valid = [], []
        for idx_fold in range(k):
            fold = idx_sample_folds[idx_fold]
            if idx_k == idx_fold:
                idx_valid += [fold]
            else:
                idx_train += [fold]

        ### HERE YOUR CODE ###
        # Concatenate folds
        x_train_folds += [np.concatenate([x[fold] for fold in idx_train], axis=0)]
        y_train_folds += [np.concatenate([y[fold] for fold in idx_train], axis=0)]
        x_valid_folds += [np.concatenate([x[fold] for fold in idx_valid], axis=0)]
        y_valid_folds += [np.concatenate([y[fold] for fold in idx_valid], axis=0)]

    ### END CODE ###

    return x_train_folds, y_train_folds, x_valid_folds, y_valid_folds


# Split train set into k folds, k-1 for train and 1 for validation
x_train_folds, y_train_folds, x_valid_folds, y_valid_folds = k_fold_cv(x_train, y_train, k=5, seed=1234)

########################################
# Distance functions
########################################

# L2 distance
def dist_l2(x1, x2):

    ### HERE YOUR CODE ###
    d_12 = np.sqrt(((x1 - x2) ** 2).sum(axis=-1))
    ### END CODE ###
    return d_12

# L1 distance
def dist_l1(x1, x2):

    ### HERE YOUR CODE ###
    d_12 = np.abs(x1 - x2).sum(axis=-1)
    ### END CODE ###
    return d_12


########################################
# K-NN Model class
########################################

class KNNModel(object):
    def __init__(self, x, y, k=1, num_classes=10):
        self.k = k
        self.x = x
        self.y = y
        self.num_classes = num_classes

    def predict(self, x, dist_func, get_freq=False):
        def get_k_closest_points(x_i, x_list):

            ### HERE YOU CODE ###
            d_list = dist_func(x_i, x_list)
            idx_k = np.argsort(d_list)[:self.k]
            ### END CODE ###

            return idx_k

        # Compute the distance between x and self.x using dist_func
        dist_matrix_k = np.zeros([x.shape[0], self.k], dtype=np.int32)
        for i, x_i in enumerate(x):
            dist_matrix_k[i, :] = get_k_closest_points(x_i, self.x)

        # Voting
        y_pred_k = self.y[dist_matrix_k]

        # Get the most frequent class and also all the voting frequency
        if get_freq:
            y_pred_freq = np.zeros([len(y_pred_k), self.num_classes], dtype=np.float32)
            for cl in range(self.num_classes):
                idx = np.where(y_pred_k == cl)
                idx_row = np.array(list(set(idx[0].tolist())), dtype=np.int32)
                y_pred_freq[idx_row, cl] = (y_pred_k[idx_row] == cl).sum(axis=-1)
            y_pred_freq = y_pred_freq / self.k
            return y_pred_freq.argmax(axis=-1), y_pred_freq
        else:

            ### HERE YOUR CODE ###
            mode = stats.mode(y_pred_k.T)[0]
            ### END CODE ###

            return mode


def get_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


k_list = np.arange(1, 11, dtype=np.int32)

k_list = np.arange(1, 11, dtype=np.int32)

# Loop over k values
acc, acc_std = {}, {}
for k in k_list:

    # Loop over 5-Folds
    acc_folds = []
    for x_train, y_train, x_valid, y_valid in tqdm_notebook(zip(x_train_folds,
                                                                y_train_folds,
                                                                x_valid_folds,
                                                                y_valid_folds),
                                                            total=5,
                                                            desc=f'k = {k}'):
        # Create the k-nn model
        knn = KNNModel(x_train, y_train, k=k)

        ### HERE YOUR CODE ###
        # Predict
        y_valid_pred = knn.predict(x_valid, dist_l2)
        ### END CODE ###

        # Evaluate model
        acc_fold = get_accuracy(y_valid, y_valid_pred)
        acc_folds += [acc_fold]

    # Average accuracies over folds
    acc[k] = np.mean(acc_folds)
    acc_std[k] = np.std(acc_folds)

# Plot accuracy over parameter k
plt.figure(figsize=(20, 6))
plt.title('Accuracy over parameter k of k-NN', fontweight='bold')
plt.bar(list(acc.keys()), list(acc.values()))
for k in acc.keys():
    plt.plot([k, k], [acc[k] - acc_std[k], acc[k] + acc_std[k]], c='k', lw=8, alpha=0.8)
plt.ylim(0.8, 0.95)
plt.xlabel('k', fontweight='bold')
plt.ylabel('Validation Accuracy', fontweight='bold')
yticks, _ = plt.yticks()
plt.yticks(yticks, [f'{int(100 * yy)}%' for yy in yticks])
plt.xticks(k_list)
plt.show()

# Best k
idx_k_best = np.array(list(acc.values())).argmax()
k_best = list(acc.keys())[idx_k_best]
print(f'Best k: {k_best}')

# Best model
knn = KNNModel(x_train, y_train, k=k_best)

# Predictions with best model
pred_valid = knn.predict(x_valid, dist_l2)
pred_test, pred_freq_test = knn.predict(x_test, dist_l2, get_freq=True)

# Evaluation
acc_valid = get_accuracy(y_valid, pred_valid)
acc_test = get_accuracy(y_test, pred_test)
print(f'Validation accuracy: {100 * acc_valid:.2f}%')
print(f'Test accuracy: {100 * acc_test:.2f}%')


def plot_predictions(x, y, pred_freq, num_col=5, show_max=25, seed=None):
    if len(x) > show_max:
        if seed is not None:
            np.random.seed(seed)
        idx_random = np.random.randint(len(x), size=show_max)
        x = x[idx_random]
        y = y[idx_random]
        pred_freq = pred_freq[idx_random]
    num_rows = int(np.ceil(len(x) / num_col))
    fig, axs = plt.subplots(num_rows, num_col, figsize=(25, num_rows * 3))
    fig.suptitle('Test Images', fontweight='bold', fontsize=28)

    for idx_sample in range(len(x)):
        idx_row = idx_sample // num_col
        idx_col = idx_sample % num_col
        ax = axs[idx_row, idx_col]
        ax.set_title(f'Label: {y[idx_sample]}', fontweight='bold')
        ax.imshow(x[idx_sample].reshape(-1, 28),
                  cmap='gray')
        ax.set_aspect(1.)
        ax.axis('off')
        divider = make_axes_locatable(ax)
        axHisty = divider.append_axes("right", 1.5, pad=0.4, sharey=ax)
        axHisty.barh(np.linspace(0, 27, 10),
                     pred_freq[idx_sample], height=2.0, color='r')
        axHisty.set_xticks(np.linspace(0, 1, 5))
        axHisty.set_title(f'Prediction: {pred_freq[idx_sample].argmax()}',
                          fontweight='bold')
        axHisty.set_xticklabels([f'{int(xx * 100)}%' for xx in np.linspace(0, 1, 5)], fontweight='bold')
        axHisty.set_ylim(30, -2)
        axHisty.set_yticks(np.linspace(0.5, 27.5, 10))
        axHisty.set_yticklabels(np.arange(10), fontweight='bold')
    plt.show()


# Show test results
axs = plot_predictions(x_test, y_test, pred_freq_test, show_max=25, seed=None)

########################################################################################################################
########################################################################################################################

from sklearn.tree import DecisionTreeClassifier

#### HERE YOUR PARAMETERS ####
criterion =  'gini'
splitter  = 'random'
tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, random_state=1)
tree.fit(x_train, y_train)
print('Accuracy score: {}'.format(tree.score(x_valid, y_valid)))


#Test here your best model on the test set
print('Accuracy score: {}'.format(tree.score(x_test, y_test)))

print('Accuracy score on training set: {}'.format(tree.score(x_train, y_train)))

from sklearn.ensemble import RandomForestClassifier
#### HERE YOUR PARAMETERS ####
criterion = 'gini'
# n_estimators = 'random'
random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=1, criterion=criterion)
random_forest.fit(x_train, y_train)
print('Accuracy score: {}'.format(random_forest.score(x_valid, y_valid)))

#Test here your best model on the test set
print('Accuracy score: {}'.format(random_forest.score(x_test, y_test)))

plt.figure(figsize=(20, 7))
res = []
for i in range(1, 150):
    criterion = 'gini'
    random_forest = RandomForestClassifier(n_estimators=1, random_state=1, criterion=criterion)
    random_forest.fit(x_train, y_train)
    res.append(random_forest.score(x_valid,y_valid))

plt.plot(range(1, 150), res, lw=4, zorder=5, label='Random Forest Accuracy')
plt.ylabel('Validation Accuracy', fontweight='bold', fontsize=15)
plt.xlabel('Number of estimators', fontweight='bold', fontsize=15)
plt.legend(fontsize='x-large')
