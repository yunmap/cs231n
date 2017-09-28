import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = W.shape[1]
  N = X.shape[0]

  for i in range(N) : 
    score_i = X[i].dot(W)
    score_set = np.exp(score_i) 
    score_sum = sum(score_set)
    #print (score_set[y[i]])
    a = (-1) * np.log(score_set[y[i]]/score_sum)
    loss += a
    #print loss
    for j in range(C) :
        softmax = score_set[j]/score_sum
        if j==y[i] :
            dW[:,j] += (softmax - 1) * X[i]
        else :
            dW[:,j] += softmax * X[i]
  loss /= N
  loss += 1/2*reg*np.sum(W*W)
  dW /= N
  dW += reg*W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = W.shape[1]
  N = X.shape[0]

  score = (X).dot(W) #N*C matrix
  #print (score)
  score_exp = np.exp(score)
  score_max = np.amax(score_exp,axis=1).reshape(-1,1)
  #print (score_max)
  score_sub_max = score_exp - score_max
  score_sum = np.sum(score_sub_max,axis=1).reshape(-1,1)
  softmax = score_sub_max / score_sum

  loss = (-1) * np.sum(np.log(softmax[range(N),list(y)]))
  dS=softmax.copy()
  dS[range(N), list(y)] -= 1
  dW = (X.T).dot(dS)
  dW /= N
  dW += reg*W
                        
  loss /= N
  loss += 1/2*reg*np.sum(W*W)
  
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

