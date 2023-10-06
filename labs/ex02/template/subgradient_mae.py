import numpy as np

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    e=y-np.dot(tx,w)
    e=np.sign(e) #we take only the signs according to the first terms of the chain rule which is a subgradient of the absolute function
    return((-1/len(y))*np.dot(tx.T,e))