import numpy as np
import pandas as pd
import pickle as pkl

def cluster_cov_matrix(d,k, c_in=1., c_out=0, c=2.):
    if k > d:
        raise Exception('Can\'t have more variable clusters (k) than variables (d)')

    A = np.zeros((d,d))

    for i in range(d):
        for j in range(i+1,d):
            p = c_out
            div = np.ceil(d/k)
            if int(i / div) == int(j / div):
                p = c_in
            e = 1*(np.random.rand() < p)
            A[i,j] = e
            A[j,i] = e
    z = np.floor(np.array(range(d))/div)

    cp = c
    while np.min(np.linalg.eig(A*cp + np.eye(d))[0]) < 0:
        cp = cp*0.9

    A = A*cp + np.eye(d)

    return A, z

def load_data(fname, frac=1.0):
    D = pkl.load(open(fname, 'rb'))

    X = D['X']
    y = D['y']

    if frac < 1.0:
        n = X.shape[0]
        I = np.random.choice(n, int(frac*n), replace=False)
        X = X.iloc[I]
        y = y[I,]

    return X, y


def synth_setting_A(n=4000, d=20, k=5, c_in=1., c_out=0., p_pos=0., p_neg=1., b_mu=0, b_std=1, eps=1):
    """
    In this setting
        - d features are divided in k clusters
        - In-cluster features have correlation proportional to c_in
        - Cross-cluster features have correlation proportional to c_out
        - The first feature in each cluster has outcome coefficient drawn from N(b_mu,b_std)
        - A cluster is completely missing with prop p_pos if the first feature in the cluster
          has value > -.5
        - A cluster is completely missing with prop p_neg if the first feature in the cluster
          has value <= -.5
    """

    if not int(d/k) == d/k:
        raise Exception('This setting only works if d/k is an integer')

    A, z = cluster_cov_matrix(d, k, c_in=c_in, c_out=c_out)

    X = np.random.multivariate_normal(np.zeros(d), A, size=n)

    b = np.zeros(d)
    Mp = np.zeros((n, d))
    for j in range(k):
        I = np.where(z==j)[0]
        i = I[0]
        b[i] = np.random.randn()*b_std + b_mu

        Mp[:,I] = 1*(X[:,[i]]>(-.5))*p_pos + 1*(X[:,[i]]<=(-.5))*p_neg

    M = 1*(np.random.rand(n,d)<Mp)

    Xm = pd.DataFrame(X.copy())
    Xm[M>0] = np.nan

    y = np.dot(X, b) + eps*np.random.randn(n)
    cfg = {'n': n, 'd': d, 'k': k, 'c_in': c_in, 'c_out': c_out, 'p_pos': p_pos,
           'p_neg': p_neg, 'b_mu': b_mu, 'b_std': b_std, 'eps': eps}

    return X, Xm, y, M, A, cfg


def synth_setting_B(n=4000, d=20, k=5, c_in=0.5, c_out=0., p_pos=0., p_neg=1., b_mu=0, b_std=1, eps=1):
    """
    In this setting
        - d features are divided in k clusters
        - In-cluster features have correlation proportional to c_in
        - Cross-cluster features have correlation proportional to c_out
        - The first feature in each cluster has outcome coefficient drawn from N(b_mu,b_std)
        - Only one feature, with index j, is observed from one cluster uniformly at random
    """

    A, z = cluster_cov_matrix(d, k, c_in=c_in, c_out=c_out)

    X = np.random.multivariate_normal(np.zeros(d), A, size=n)

    b = np.zeros(d)
    Mp = np.zeros((n, d))
    for j in range(k):
        I = np.where(z==j)[0]
        i = I[0]
        b[i] = np.random.randn()*b_std + b_mu

        Mp[:,I] = 1*(X[:,[i]]>(-.5))*p_pos + 1*(X[:,[i]]<=(-.5))*p_neg

    M = 1*(np.random.rand(n,d)<Mp)

    L = int(d/k)
    for i in range(n):
        j = np.random.randint(int(d/k))
        M[i,range(j,d,L)] = 0

    Xm = pd.DataFrame(X.copy())
    Xm[M>0] = np.nan

    y = np.dot(X, b) + eps*np.random.randn(n)

    cfg = {'n': n, 'd': d, 'k': k, 'c_in': c_in, 'c_out': c_out, 'p_pos': p_pos,
           'p_neg': p_neg, 'b_mu': b_mu, 'b_std': b_std, 'eps': eps}

    return X, Xm, y, M, A, cfg
