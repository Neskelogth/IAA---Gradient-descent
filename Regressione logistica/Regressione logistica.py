import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Check function. Used for checking your code, you can ignore this.
def check_func(func, *args):
    res = {
        'sigmoid': np.array([4.5397868702434395e-05, 0.0066928509242848554,
                   0.11920292202211755, 0.5, 0.8807970779778823,
                   0.9933071490757153, 0.9999546021312976]).reshape(-1, 1),
        'xent': [1.3758919771597742, 0.8487364948617685, 0.8616020843171404,
                    1.2847647859647024, 1.0979517701821886, 1.2448204497955682,
                    1.148747135298692, 0.9142123727250151, 0.9503784164146648,
                    0.5259295090148516],
        'gradient': [0.05517542101762578, 0.056956027283867096,
                     0.10246057091641199, 0.1729957759994022,
                     -0.07420252599770667, -0.10440419638201064,
                     0.0814283804697549, -0.1722842404987195,
                     0.16445779692602908, -0.0962341984706773],
        'predict_lc': [1., -1., 1., 1., -1., -1., 1., 1., -1., 1.],
        'predict_lr': [1., 0., 1., 1., 0., 0., 1., 1., 0., 1.],
        'mse': [0.20665046013997404, 0.12503917069875575, 0.15871653661209398,
                0.24302395420833073, 0.19655530689347242, 0.19455858448384966,
                0.2078562677711177, 0.1626431893970271, 0.12018726319497597,
                0.0969268504978511]
    }
    with open('./ex2data1.txt', 'r') as txt:
        y = np.array([[float(line.strip().split(',')[2])]
                      for line in txt.readlines()], dtype=np.float32)
    res['evaluate_lr'] = [y]
    res['evaluate_lc'] = [np.array([-1 if y_i < 0.5 else 1 for y_i in y.reshape(-1)], dtype=np.float32).reshape(-1, 1)]
    print('CHECK RESULTS:')
    print('\n' + ''.join(['=' for _ in range(40)]))
    are_correct = []
    for idx, y in enumerate(res[func.__name__]):
        arg = [a[idx] for a in args]
        y_ = func(*arg)
        if func.__name__ == 'evaluate_lr':
            acc = (1.0 * (y_ == y)).mean()
            y_, y = acc, 0.88
            are_correct += [f'{y_:.4f}' > f'{y:.4f}']
            print(f'Your train accuracy: {100 * y_:.2f}%, Expected train accuracy: > {100 * y:.2f}%')
        elif func.__name__ == 'evaluate_lc':
            acc = (1.0 * (y_ == y)).mean()
            y_, y = acc, 0.88
            are_correct += [f'{y_:.4f}' > f'{y:.4f}']
            print(f'Your train accuracy: {100 * y_:.2f}%, Expected train accuracy: > {100 * y:.2f}%')
        else:
            if isinstance(y_, np.ndarray):
                y_ = y_.reshape(-1)[0]
            if isinstance(y, np.ndarray):
                y = y.reshape(-1)[0]
            are_correct += [f'{y_:.4f}' == f'{y:.4f}']
            print(f'Your result: {y_:.4f}, Expected: {y:.4f}')
    print('\n' + ''.join(['=' for _ in range(40)]))
    if all(are_correct):
        print('Function is correct! Well done!'.upper())
    else:
        print('Function is not correct. Find the bug.')

# Load the dataset
x, y = [], []
with open('./ex2data1.txt', 'r') as txt:
    for line in txt.readlines():
        vals = line.strip().split(',')
        x += [[float(vals[0]), float(vals[1])]]
        y += [[float(vals[2])]]
x_raw = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)
y_lc = np.array([-1 if y_i < 0.5 else 1 for y_i in y], dtype=np.float32).reshape(-1, 1)

# Plot the dataset
plt.figure(figsize=(10, 7))
plt.title('Dataset', fontweight='bold', fontsize=25)
plt.scatter([x_i[0] for i, x_i in enumerate(x_raw) if y[i] == 0],
            [x_i[1] for i, x_i in enumerate(x_raw) if y[i] == 0],
            label='Not Admited', s=60, marker='x')
plt.scatter([x_i[0] for i, x_i in enumerate(x_raw) if y[i] == 1],
            [x_i[1] for i, x_i in enumerate(x_raw) if y[i] == 1],
            label='Admited')
plt.grid(True, alpha=0.6, zorder=0, ls='--')
plt.xlabel('Exam 1 Score', fontweight='bold')
plt.ylabel('Exam 2 Score', fontweight='bold')
plt.legend()
plt.show()

# Normalize the dataset
x_mu, x_std = x_raw.mean(axis=0), x_raw.std(axis=0)
x = (x_raw - x_mu) / x_std

# Append ones column
x = np.concatenate([x, np.ones([len(x), 1], dtype=np.float32)], axis=-1)

def predict_lc(x, theta):
    '''
    input x: np.ndarray of shape (m, 3)
    input theta: np.ndarray of shape (3, 1)
    output y: np.ndarray of shape (m, 1)
    '''

    h = np.dot(x, theta)
    y = np.sign(h)

    return y

# Check function
np.random.seed(1234)
_x = np.random.uniform(-1, 1, size=(10, 10, 3))
_theta = np.random.uniform(-1, 1, size=(10, 3, 1))
check_func(predict_lc, _x, _theta)

def mse(y_true, y_pred):
    '''
    input y_true: np.ndarray of shape (m, 1)
    input y_pred: np.ndarray of shape (m, 1)
    '''

    sum = ((y_true - y_pred) ** 2).sum()
    J = sum * (1 / (2 * len(y_true)))

    return J

# Check function
np.random.seed(1234)
y_true = np.random.randint(2, size=(10, 10))
y_pred = np.random.uniform(0, 1, size=(10, 10))
check_func(mse, y_true, y_pred)

def gradient(y_true, y_pred, x):
    '''
    input y_true: np.ndarray of shape (m,)
    input y_pred: np.ndarray of shape (m,)
    input x: np.ndarray of shape (m, 3)
    output dJ: np.array of shape (3, 1)
    '''

    # Reshape arrays
    y_true = y_true.reshape(-1, 1)  # now shape (m, 1)
    y_pred = y_pred.reshape(-1, 1)  # now shape (m, 1)


    y_true = y_true.reshape(-1, 1)  # now shape (m, 1)
    y_pred = y_pred.reshape(-1, 1)  # now shape (m, 1)
    dJ = x.T.dot((y_pred - y_true)) / len(x)

    return dJ


# Check function
np.random.seed(1234)
y_true = np.random.randint(2, size=(10, 10))
y_pred = np.random.uniform(0, 1, size=(10, 10))
x_in = np.random.uniform(0, 1, size=(10, 10, 1))
check_func(gradient, y_true, y_pred, x_in)


def gradient_descent(x, y, activation_func, cost_func, gradient_func,
                     epochs=400, seed=1234, lr=0.01, print_every=10):
    # Initialize theta parameters
    np.random.seed(seed)
    theta = np.random.normal(0, 0.001, size=(x.shape[1], 1)) / np.sqrt(2)

    # Iterations of gradient descent
    loss = []
    print('Training...')
    print(''.join(['=' for _ in range(40)]))
    for epoch in range(epochs + 1):
        loss_epoch = []

        # Model prediction
        z = x.dot(theta)
        h = activation_func(z) if activation_func is not None else z
        loss += [cost_func(y, h)]

        # Parameters update
        dJ = gradient(y, h, x)
        theta = theta - lr * dJ

        # Print loss info
        if epoch % print_every == 0:
            print(f'Epoch {epoch}: Loss {loss[-1]}')

    return theta, loss


theta_lc, loss_lc = gradient_descent(x, y_lc, activation_func=None,
                                     cost_func=mse,
                                     gradient_func=gradient, epochs=10000,
                                     lr=0.001, print_every=1000)


# Plot loss
plt.figure(figsize=(10, 5))
plt.title('Train the Linear Classification Model', fontweight='bold',
          fontsize=25)
plt.plot(loss_lc, lw=2)
plt.grid(True, zorder=0, alpha=0.6, ls='--')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.show()

# Evaluate Linear Classification
def evaluate_lc(x, theta, y):
    y_ = predict_lc(x, theta)
    return y_

# Check function
check_func(evaluate_lc, [x], [theta_lc], [y_lc])

# Plot decision boundary
boundary = lambda theta, x: (- theta[0] / theta[1] * (x - x_mu[0]) / x_std[0] - theta[2] / theta[1]) * x_std[1] + x_mu[1]

# Plot the dataset
plt.figure(figsize=(10, 7))
plt.title('Decision Boundary', fontweight='bold', fontsize=25)
plt.scatter([x_i[0] for i, x_i in enumerate(x_raw) if y_lc[i] == -1],
            [x_i[1] for i, x_i in enumerate(x_raw) if y_lc[i] == -1],
            label='Not Admited', s=60, marker='x')
plt.scatter([x_i[0] for i, x_i in enumerate(x_raw) if y[i] == 1],
            [x_i[1] for i, x_i in enumerate(x_raw) if y[i] == 1],
            label='Admited')
plt.plot(np.linspace(30, 100, 100), boundary(theta_lc, np.linspace(30, 100, 100)),
         c='y', lw=2, label='Decision boundary')
plt.grid(True, alpha=0.6, zorder=0, ls='--')
plt.xlabel('Exam 1 Score', fontweight='bold')
plt.ylabel('Exam 2 Score', fontweight='bold')
plt.legend()
plt.show()

def sigmoid(z):
    '''
    input z: np.ndaray of shape (m, 3)
    output s: np.ndarray of shape (m, 3) where s[i, j] = g(z[i, j])
    '''

    s = 1.0 / ( 1.0 + np.exp(- z) )

    return s

# Check function
z = np.array([-10, -5, -2, -0, 2, 5, 10]).reshape(-1, 1)
check_func(sigmoid, z)

# Plot sigmoid function
plt.figure(figsize=(10, 7))
plt.title('Sigmoid Function Plot', fontweight='bold', fontsize=25)
plt.plot(np.linspace(-10, 10, 100),
         [sigmoid(z) for z in np.linspace(-10, 10, 100)], lw=4)
plt.ylim(-0.2, 1.2)
plt.grid(True, ls='--', alpha=0.6, zorder=0)
plt.xlabel('z', fontweight='bold')
plt.ylabel('Sigmoid(z)', fontweight='bold')
plt.show()

def xent(y_true, y_pred):
    '''
    input y_true: np.ndarray of shape (m,)
    input y_pred: np.ndarray of shape (m,)
    output J: float
    '''

    J = - (y_true * np.log(y_pred) + ( 1.0 - y_true) * np.log( 1.0 - y_pred)).mean()

    return J

# Check function
np.random.seed(1234)
y_true = np.random.randint(2, size=(10, 10))
y_pred = np.random.uniform(0, 1, size=(10, 10))
check_func(xent, y_true, y_pred)

def predict_lr(x, theta):

    z = x.dot(theta)
    h = sigmoid(z)
    y = np. round (h)

    return y

# Check function
np.random.seed(1234)
_x = np.random.uniform(-1, 1, size=(10, 10, 3))
_theta = np.random.uniform(-1, 1, size=(10, 3, 1))
check_func(predict_lr, _x, _theta)

theta_lr, loss_lr = gradient_descent(x, y, activation_func=sigmoid,
                                          cost_func=xent,
                                          gradient_func=gradient, epochs=200000,
                                          lr=0.001, print_every=10000)

# Plot loss
plt.figure(figsize=(10, 5))
plt.title('Train the Logistic Regression Model', fontweight='bold',
          fontsize=25)
plt.plot(loss_lr, lw=2)
plt.grid(True, zorder=0, alpha=0.6, ls='--')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.show()

# Evaluate Logistic Regression
def evaluate_lr(x, theta, y):
    y_ = predict_lr(x, theta)
    return y_

check_func(evaluate_lr, [x], [theta_lr], [y])

# Plot decision boundary
boundary = lambda theta, x: (- theta[0] / theta[1] * (x - x_mu[0]) / x_std[0] - theta[2] / theta[1]) * x_std[1] + x_mu[1]

# Plot the dataset
plt.figure(figsize=(10, 7))
plt.title('Decision Boundary', fontweight='bold', fontsize=25)
plt.scatter([x_i[0] for i, x_i in enumerate(x_raw) if y[i] == 0],
            [x_i[1] for i, x_i in enumerate(x_raw) if y[i] == 0],
            label='Not Admited', s=60, marker='x')
plt.scatter([x_i[0] for i, x_i in enumerate(x_raw) if y[i] == 1],
            [x_i[1] for i, x_i in enumerate(x_raw) if y[i] == 1],
            label='Admited')
plt.plot(np.linspace(30, 100, 100), boundary(theta_lr, np.linspace(30, 100, 100)),
         c='r', lw=2, label='LR Decision boundary')
plt.plot(np.linspace(30, 100, 100), boundary(theta_lc, np.linspace(30, 100, 100)),
         c='y', lw=2, label='LC Decision boundary')
plt.grid(True, alpha=0.6, zorder=0, ls='--')
plt.xlabel('Exam 1 Score', fontweight='bold')
plt.ylabel('Exam 2 Score', fontweight='bold')
plt.legend()
plt.show()

