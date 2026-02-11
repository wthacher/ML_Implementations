import numpy as np


'''
Algorithm: choose number of centers k to use, initialize them from data set
1) compute clusters n by 1 array which records center that each point belongs to
2) compute new centers as means of current clusters

repeat until convergence or until some max number of iterations

post: visualize centers

'''

def compute_clusters(data, centers):
    clusters = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        clusters[i] = np.argmin( np.linalg.norm(data[i,:] - centers, axis=1) )
    return clusters

def compute_centers(clusters, data, k):
    centers = np.zeros( (k, data.shape[1]) )
    tots = np.zeros(k)
    for i in range(data.shape[0]):
        centers[clusters[i]] += data[i,:]
        tots[clusters[i]]+=1;
    tots = np.maximum(tots, 1) ##need at least 1
    return centers / tots.reshape(k,1)

def init_centers(k, data):
    ##randomly select k of the points
    centers = data[np.random.choice(data.shape[0], k, replace=False),:]
    return centers

def run_k_means(k, data, tol, max_iter):
    centers = init_centers(k, data)
    err = 2*tol
    it = 0
    while err > tol and it < max_iter:
        clusters = compute_clusters(data, centers)
        centers_new = compute_centers(clusters,data, k)
        err = np.max( np.linalg.norm(centers - centers_new, axis=1) )
        centers = centers_new

        it+=1

    return clusters, centers



import matplotlib.pyplot as plt
##if data is 2d, vis clusters
def visualize_clusters_2d(data, clusters, centers):
    plt.scatter(centers[:,0], centers[:,1], color='b',marker='x')
    plt.scatter(data[:,0], data[:,1],color='r')
    
    plt.show()


##dimension of data
d = 2
##number of data points
n = 100
##totally random data
data = np.random.randn(100, 3)
data[:50]+=5;
data[50:75]+=10;
max_iter = 100
k = 4
tol = 1e-10

clusters, centers = run_k_means(k, data, tol, max_iter)

visualize_clusters_2d(data, clusters, centers)
