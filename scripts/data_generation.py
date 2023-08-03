import numpy as np

def generate_one_cluster_signal(mu1,mu2,sigma1,sigma2,n_samples,cluster_label):
    '''
    Generate a 2D Gaussian cluster center at (mu1,mu2) with stdev (sigma1,sigma2) with N samples.

    Parameters
    ----------
    mu1,mu2 : float
        center of 2d gaussian cluster on x,y axis

    sigma1,sigma2 : float
        standard deviation of 2d gaussian cluster on x,y axis
        
    n_samples : int
        number of samples in the cluster
    
    cluster label : int 
        label for the cluster

    Returns:
    ---------
    X,y : np.array shape of (n_smaples,1) and (n_samples,1)
        Cluster positions in 2d plane, cluster labels
    '''
    
    X = np.random.multivariate_normal([mu1,mu2],[[sigma1,0],[0,sigma2]],n_samples)
    y = np.full((n_samples,1),cluster_label)
    
    return X,y


def add_noisy_features(x,n_noisy_features):
    '''
    Add #'n_noisy_features' noisy features to the existing feature vector x

    Parameters
    ----------
    x : ndarray like
        clusters' signal feature vector
    n_noisy_features : int
        number of noisy features

    Returns
    ----------
    x : ndarray
        clusters' feature vector with noisy
    '''
    n_samples = len(x)
    for i in range(n_noisy_features):
        x = np.concatenate((x, np.random.uniform(0,1,(n_samples,1))), axis=1)
    return x

def generate_clusters(n_samples=1000,
                      centers=[(0,0),(0.5,0.5)],
                      sigmas=[(0.01,0.01),(0.01,0.01)],
                      n_noisy_features=0,
                      shuffle=True,
                      seed=20200213):
    '''
    generate k clusters with noisy features

    Parameters
    ----------
    n_samples : int
        number of samples in each clusters

    centers : list of 2d tuple 
        center positions of 2d cluster. 

    sigmas : list of 2d tuple
        stdev of each cluster
    
    shuffle : boolean 
        shuffle the data
        
    n_noisy_features : int
        number of noisy features adding to the signal data
        
    Returns:
    ---------
    X ,y : np.array shape (n_samples*number of clusters,2+n_noisy_features),(n_samples*number of clusters,1)
    '''
    
    k = len(centers)
    for i in range(k):
        mu1,mu2 = centers[i]
        sigma1,sigma2 = sigmas[i]
        
        if i == 0:
            X,y = generate_one_cluster_signal(mu1,mu2,sigma1,sigma2,n_samples,i+1)
        else:
            x1,y1 = generate_one_cluster_signal(mu1,mu2,sigma1,sigma2,n_samples,i+1)
            X = np.concatenate((X,x1))
            y = np.concatenate((y,y1))

    X = add_noisy_features(X,n_noisy_features)

    if shuffle:
        total_n_samples = len(X)
        indices = np.arange(total_n_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    return X,y