import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits import mplot3d


def check(func, print_log=True):
    # Set seed
    np.random.seed(1234)

    # Select function to check
    if func.__name__ == 'predict':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'w': np.random.normal(0, 1, size=(501, 1)),
        }
        res = 84.37925870442523
        res_ = func(**args).sum()
        cond = (res_.sum() - res) < 1e-8

    elif func.__name__ == 'compute_cost':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'y': np.random.normal(0, 1, size=(10, 1)),
            'w': np.random.normal(0, 1, size=(501, 1)),
        }
        res = 131.13466658728424
        res_ = func(**args)
        cond = (res_ - res) < 1e-8

    elif func.__name__ == 'compute_cost_multivariate':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'y': np.random.normal(0, 1, size=(10, 1)),
            'w': np.random.normal(0, 1, size=(501, 1)),
        }
        res = 131.13466658728424
        res_ = func(**args)
        cond = (res_ - res) < 1e-8

    elif func.__name__ == 'gradient_descent':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'y': np.random.normal(0, 1, size=(10, 1)),
            'w': np.random.normal(0, 1, size=(501, 1)),
            'learning_rate': 0.005,
            'num_iters': 10,
        }
        res = [
            158.544797156765,
            2.203001889427611,
            25.109538060963843,
        ]  # Sums of the arryays
        res_ = func(**args)
        cond = all([(r_.sum() - r) < 1e-8 for r, r_ in zip(res, res_)])

    else:
        raise Exception(f'Error. The check of the function {func.__name__} is not implemented.')

    if cond:
        print(f'Your function "{func.__name__}" is correct!')
    else:
        print(f'Your function "{func.__name__}" is NOT correct!')

    # Print output log
    if print_log:
        if isinstance(res, list):
            for r, r_ in zip(res, res_):
                if isinstance(r_, np.ndarray):
                    r_ = r_.sum()
                print(f'Your output: {r_}, expected output: {r}')
        if isinstance(res, float) or isinstance(res, int) or isinstance(res, str):
            print(f'Your output: {res_}, expected output: {res}')

# Settings
plt.style.use('seaborn-white')

# Load dataset
dataset = load_boston()

# Print dataset info
print(dataset.data.shape)
print(dataset.keys())
print(dataset.feature_names)
print(dataset.DESCR)

# Retrieve input features and target prices
x = np.array(dataset.data)  # Input data of shape [num_samples, num_feat]
y = np.array(dataset.target)  # Targets of shape [num_samples]

# Info
print(f'Dataset shape -> {x.shape}, target variable shape -> {y.shape}')

# Price distribution
plt.figure(figsize=(10, 5))
kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=30, ec="k")
plt.hist(y, **kwargs)
plt.xlabel("House prices in $1000")
plt.show()
plt.show()

# Plot (feature, target) plot, for each single feature
plt.figure(figsize=(25, 25))
for idx_f, feat_name in enumerate(dataset.feature_names):
    plt.subplot(5, 3, idx_f + 1)
    plt.scatter(x[:, idx_f], y, marker='o')
    plt.xlabel(feat_name, fontweight='bold')
    plt.ylabel('House prices in $1000', fontweight='bold')
    plt.grid(True)
plt.show()

# Select LSTAT feature from X
x_lstat = x[:, -1].reshape(-1, 1)
y_price = y.reshape(-1, 1)

# Print info
print(x_lstat.shape)
print(y_price.shape)

x_train, x_test, y_train, y_test = train_test_split(x_lstat, y_price, test_size=0.2, random_state=5)
num_tr = len(x_train)
num_feat = x_train.shape[-1]

# Append intercept term to x_train
print(x_train.shape)
x_train = np.concatenate([x_train, np.ones([num_tr, 1])], -1)
print(x_train.shape)
print(f'Total samples in X_train: {num_tr}')


##################################################
# Compute the prediction
##################################################

# Risultato atteso: 84.37925870442523
def predict(x, w):
    """
  Compute the prediction of a linear model.
  Inputs:
      x: np.ndarray input data of shape [num_samples, num_feat + 1]
      w: np.ndarray weights of shape [num_feat + 1, 1]
  Outputs:
      h: np.ndarray predictions of shape [num_samples, 1]
  """

    # return np.transpose(w) * x
    return np.dot(x, w)


# Test your code -> uncomment
check(predict)


##################################################
# Loss function (mean squared error -> MSE)
##################################################

def compute_cost(x, y, w):
    """
    Inputs:
        x: np.ndarray input data of shape [num_samples, num_feat + 1]
        y: np.ndarray targets data of shape [num_samples, 1]
        w: np.ndarray weights of shape [num_feat + 1, 1]
    Outputs:
        mse: scalar.
    """

    ##### WRITE YOUR CODE HERE #####
    # const = 1 / (2 * len(x))
    # cost = 0
    #
    # for i in range(len(x)):
    #   cost += (predict(x[i], w).sum() - y[i]) ** 2
    # ################################

    # return cost * const

    h = predict(x, w)
    mse = ((y - h) ** 2) / (2 * x.shape[0])
    return mse.sum()


# Test your code -> uncomment
check(compute_cost)

def gradient_descent(x, y, w, learning_rate, num_iters):
    """
  Inputs:
      x: np.ndarray input data of shape [num_samples, num_feat + 1]
      y: np.ndarray targets data of shape [num_samples, 1]
      w: np.ndarray weights of shape [num_feat + 1, 1]
      learning_rate: scalar, the learning rate.
      num_iters: int, the number of iterations.
  Outputs:
      j_hist: list of loss values of shape [num_iters]
      w_opt: [num_feat + 1, 1]
      w_hist: [num_feat + 1, num_iters + 1]
  """
    ##### WRITE YOUR CODE HERE #####

    num_samples, num_feat = len(x), len(w) - 1
    j_hist = np.zeros([num_iters])
    w_hist = np.zeros([num_feat + 1, num_iters + 1])
    w_hist[:, 0] = w.T

    for i in range(num_iters):
        h = np.dot(x, w)  # Shape [num_samples, 1]
        # Compute gradient
        dw = 1 / num_samples * np.dot((h - y).T, x).T  # Shape [num_feat + 1, 1]
        w = w - learning_rate * dw
        w_hist[:, i + 1] = w.T
        j_hist[i] = compute_cost(x, y, w)

    return j_hist, w, w_hist

    ################################

# Test your code -> uncomment
check(gradient_descent)


# Initialize the parameters of the linear model
w = np.zeros([num_feat + 1, 1])

# Parameters for the gradient descent
num_iters = 3000
learning_rate = 0.005

# Compute the initial cost
initial_cost = compute_cost(x_train, y_train, w)
print("Initial cost is: ", initial_cost)

# Apply gradient descent algorithm
j_hist, w_opt, w_hist = gradient_descent(x_train, y_train, w, learning_rate, num_iters)
print("Optimal parameters are: \n", w_opt)
print("Final cost is: ", j_hist[-1])

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(range(len(j_hist)), j_hist, lw=3)
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.grid()
plt.show()

# Plot the model fitted line on the output variable
plt.figure(figsize=(7, 7))
prediction_space = np.linspace(x_lstat.min(), x_lstat.max()).reshape(-1,1)
prediction_space = np.concatenate([prediction_space, np.ones([len(prediction_space), 1])], -1)
plt.scatter(x_train[:, 0], y_train)
plt.plot(
    prediction_space[:, 0],
    predict(
        prediction_space,
        w_opt
    ),
    color='r', linewidth=4
)
plt.ylabel('value of house/1000($)')
plt.xlabel('LSTAT (% lower status of the population)')
plt.show()

fig = plt.figure(figsize=(14,8)) # create the canvas for plotting
ax1 = fig.add_subplot(1,2,1)

# Create grid of w0/w1 values
w0_values = np.linspace(-50, 50, 100);
w1_values = np.linspace(-20, 20, 100);
W0, W1 = np.meshgrid(w0_values, w1_values)
J = np.zeros((w0_values.shape[0] * w1_values.shape[0]))

# Compute cost function for each point in the grid
for i, (w0,w1) in enumerate(zip(np.ravel(W0), np.ravel(W1))):
  w = [w0, w1]
  J[i] = compute_cost(x_train, y_train, w)

J = J.reshape(w0_values.shape[0], w1_values.shape[0])

# Plot cost function
plt.contour(W0, W1, J, 100)
plt.colorbar()

# Plot params history
plt.plot(w_hist[0, 0], w_hist[1, 0], 'bx', # Initial position
         w_hist[0, -1], w_hist[1, -1], 'rx', # Final position
         w_hist[0, ::10], w_hist[1, ::10], 'r--')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(W0, W1, J, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax2.set_title('J(w0, w1)')


# Split the dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
num_tr = len(x_train)
num_feat = x_train.shape[-1]

# Normalize the data
mu = x_train.mean(axis=0) # Mean from the train set
sigma = x_train.std(axis=0) # STD from the train set
x_train = (x_train - mu) / sigma
x_test = (x_test - mu) / sigma # Normalize with stats from the train
y_train = y_train.reshape(-1, 1)

# Add intercept term
x_train = np.concatenate([x_train, np.ones([num_tr, 1])], -1)
x_test = np.concatenate([x_test, np.ones([len(x_test), 1])], -1)
print(x_train.shape)


##################################################
# Loss function (mean squared error -> MSE)
##################################################

def compute_cost_multivariate(x, y, w):
  """
  Inputs:
      x: np.ndarray input data of shape [num_samples, num_feat + 1]
      y: np.ndarray targets data of shape [num_samples, 1]
      w: np.ndarray weights of shape [num_feat + 1, 1]
  Outputs:
      mse: scalar.
  """

  num_samples = len(x)
  h = predict(x, w)
  mse = np.dot((h - y).T, h - y)[0, 0] / (2 * num_samples)
  return mse

# Test your code -> uncomment
check(compute_cost_multivariate, print_log=True)

# Initialize the weights of the linear model
w = np.zeros((num_feat + 1, 1))

# Parameters of the gradient descent
num_iters = 5000
learning_rate = 0.01

# Comput the initial cost
initial_cost = compute_cost_multivariate(x_train, y_train, w)
print("Initial cost is: ", initial_cost, "\n")

# Gradient descent
(j_hist, w_opt, _) = gradient_descent(x_train, y_train, w, learning_rate, num_iters)

# Output info
print("Optimal parameters are: \n", w_opt, "\n")
print("Final cost is: ", j_hist[-1])
plt.figure(figsize=(10, 5))
plt.plot(range(len(j_hist)), j_hist, lw=3)
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.grid()
plt.show()

# Predicted prices
y_test_pred = predict(x_test, w_opt)

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual House Prices ($1000)")
plt.ylabel("Predicted House Prices: ($1000)")
plt.xticks(range(0, int(max(y_test)), 2))
plt.yticks(range(0, int(max(y_test_pred)), 2))
plt.title("Actual Prices vs Predicted prices")


def score(y, y_pred):
  score = 1 - (((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
  return score


sklearn_regressor = LinearRegression().fit(x_train, y_train)
sklearn_train_accuracy = sklearn_regressor.score(x_train, y_train)
sklearn_test_accuracy = sklearn_regressor.score(x_test, y_test)

# Prediction for training set
y_train_pred = predict(x_train, w_opt)
train_accuracy = score(y_train, y_train_pred)
test_accuracy = score(y_test[:, np.newaxis], y_test_pred)
print("Training accuracy   Our model -> %f\tSklearn's implementation -> %f" % (train_accuracy, sklearn_train_accuracy))
print("Test accuracy       Our model -> %f\tSklearn's implementation -> %f" % (test_accuracy, sklearn_test_accuracy))

