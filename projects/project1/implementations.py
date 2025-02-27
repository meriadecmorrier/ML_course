import numpy as np

threshold = 1e-8

def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e=y-np.dot(tx,w)
    return (1/(2*len(y)))*np.dot(e,e.T)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e=y-np.dot(tx,w)
    gd=(-1/len(y))*np.dot(tx.T,e)
    return(gd)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

############################################################################################################
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ # 
############################################################################################################


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    w = initial_w
    prev_loss = None
    for n_iter in range(max_iters):
        gd=compute_gradient(y,tx,w)
        #update w by gradient
        w=w-gamma*gd
        loss = compute_mse(y,tx,w)
        if n_iter % 50 == 0:
            print(
		        "GD iter. {bi}/{ti}: loss={l}".format(
		            bi=n_iter, ti=max_iters - 1, l=loss
		        )
		    )

        # converge criterion
        if prev_loss is not None and np.abs(prev_loss - loss) < threshold:
            break
        prev_loss = loss
        
        

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss=compute_mse(minibatch_y,minibatch_tx,w)
            gd=compute_gradient(minibatch_y,minibatch_tx,w)
        print(
            "SGD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
        w=w-gamma*gd
    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    
    w=np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    loss=compute_mse(y,tx,w)
    return w,loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """

    a=lambda_*np.identity(len(y))
    b=np.dot(tx.T,tx)
    left_term=np.dot(tx.T,tx)+2*len(y)*lambda_*np.identity(tx.shape[1])
    right_term=np.dot(tx.T,y)
    w=np.linalg.solve(left_term,right_term)
    loss=compute_mse(y,tx,w)
    return w,loss

############################################################################################################
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ # 
############################################################################################################

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return 1/(1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    N = y.shape[0]
    D = w.shape[0]
    assert tx.shape == (N, D)

    s = sigmoid(tx.dot(w))
    return -(y.T.dot(np.log(s)) + (1-y).T.dot(np.log(1 - s)))/N


 
def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    N = y.shape[0]
    D = w.shape[0]
    assert tx.shape == (N, D)
    
    return tx.T.dot(sigmoid(tx.dot(w)) - y)/N

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    N = y.shape[0]
    D = w.shape[0]
    assert tx.shape == (N, D)

    loss = calculate_loss(y, tx, w)
    w = w - gamma * calculate_gradient(y, tx, w)
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    prev_loss = None
	
    # start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if n_iter % 10 == 0:
            print(f"Current iteration={n_iter}, loss={loss}")
        # converge criterion
        if prev_loss and np.abs(prev_loss - loss) < threshold:
            break
        prev_loss = loss
    
    return w, loss

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.63537268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    N = y.shape[0]
    D = w.shape[0]
    assert tx.shape == (N, D), f"tx expected of {N}*{D}, found {tx.shape}"

    # return loss, gradient, and Hessian
    loss = calculate_loss(y, tx, w) + lambda_ * w.T.dot(w)[0][0]
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w

    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.63537268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    N = y.shape[0]
    D = w.shape[0]
    assert tx.shape == (N, D), f"tx expected of {N}*{D}, found {tx.shape}"

    # return loss, gradient
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    # update w
    w -= gamma * gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    prev_loss = None

    # start the logistic regression
    for n_iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if n_iter % 100 == 0:
            print(f"Current iteration={n_iter}, loss={loss}")
        # converge criterion
        if prev_loss is not None and np.abs(prev_loss - loss) < threshold:
            break
    
    return w, loss