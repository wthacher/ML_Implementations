import torch

'''
implement naive bayes classifier.

problem setting: predict one of K classes given data x in R^d

model  - assume that for each class C_k, p(x|C_k) can be factored into d gaussians:

p(x | C_k) = \prod_{l=1}^d p(x[l]|C_k), where p(x[l]|C_k) = N(\mu_{kl}, \sigma_{kl}^2)

how do we use such a model? apply bayes rule to compute p(C_k|x).

how do we fit this model? conditioned on the class, no interaction:

use MLE: X_k = subset of data points in class k: (x_i, t_i)

log likelihood of these L(X) = \sum_{i:t_i = k} \sum_{l=1}^d ln( p(x[l]|C_k)  ) 
= \sum_{i:t_i = k} \sum_{l=1}^d ln(1/(2 \sigma_kl)) + -1/2 |x_l - mu_{lk}|^2/(\sigma_{kl}^2)

so to find these parameters: \nabla_{\mu, \sigma} L(X;\mu,\sigma) = 0

look at class k:
\sum_{i:t_i = k} \sum_{l=1}^d (x_l - \mu_{lk}) = 0 -> set mu_{lk} = average over class k, over component l

-1/\sigma_{kl} + |x_l - \mu_{kl}|^2 / (\sigma_{kl})^3 = 0 
-1 + |x_l - \mu_kl|^2/ (\sigma_{kl})^2 = 0
\sigma_{kl} = average over k, over comp l of |x_l - \mu_kl|^2

after we have computed these, how to make a prediction?
compute p(C_k) from fraction of data, then:

p(C_k|x) = p(x|C_k) p(C_k) / p(x)
    compute p(x) = sum_k p(x|C_k) p(C_k)

ln(p(C_k|x)) = ln(p(x|C_k)) + ln(p(C_k)) - ln(p(x))

    
then make a decision about which class you want
'''

##implement this in pytorch
##X will be a tensor of size n by d
##t will be a tensor of size n by 1

##compute the parameters = mu will be k by d, sigma^2 will be k by d as well

##then need a function for computing class probabilities - probably compute log probability then rescale



def compute_parameters(X, t, k):
    d = X.shape[1]

    counts = torch.bincount(t).float()
    mu = torch.zeros(k, d)
    mu.index_add_(0, t, X)
    mu /= counts.unsqueeze(-1).clamp(min=1)
    sq_d = (X - mu[t])**2
    ss = torch.zeros(k, d)
    ss.index_add_(0, t, sq_d)
    ss /= counts.unsqueeze(-1).clamp(min=1)
    return mu, ss, counts/X.shape[0]


from torch.distributions import Normal

def compute_class_log_probabilities(mu, ss,pk, X):

    D = Normal(mu,torch.sqrt(ss))
    p_x_given_C_k = D.log_prob(X.unsqueeze(1)).sum(dim=2)
    p_x_and_C_k = p_x_given_C_k + torch.log(pk)
    p_x = torch.logsumexp(p_x_and_C_k, dim=1, keepdim=True)
    return p_x_and_C_k - p_x

# X = torch.randn((100,10))
# t = torch.randint(0,6,(100,))

# mu,ss,pk = compute_parameters(X,t,6)
# X_t = torch.randn((50,10))
# LP = compute_class_log_probabilities(mu,ss,pk,X_t)


##test on IRIS data cuz I dont want the world to see me
from sklearn.datasets import load_iris
import torch

data = load_iris()
X = torch.tensor(data.data, dtype=torch.float32)
t = torch.tensor(data.target, dtype=torch.long)
N = t.shape[0]
shuff = torch.randperm(N)
t = t[shuff] ##shuffle the data
X = X[shuff,:]



n = X.shape[0]
n_train = int(.75 * n)
mu,ss,pk = compute_parameters(X[:n_train,:], t[:n_train], 3)
LP = compute_class_log_probabilities(mu,ss,pk, X[n_train:, :])

import matplotlib.pyplot as plt
##compute fraction correct

classes = torch.argmax(LP, dim=1)
correct = (classes == t[n_train:])
print(torch.sum(correct)/len(correct))