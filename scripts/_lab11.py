import matplotlib; matplotlib.use('Agg')
import numpy as np, scipy as sp, scipy.linalg, sklearn, pandas as pd, seaborn as sns
print(np.__version__, pd.__version__, sp.__version__, sklearn.__version__, sns.__version__)
SNR = 3
dist=1.; n_classes = 2; n_samples = 1000; n_feat = 3
M_Means = 0.5 * dist * np.array([[1, 1, 1],[-1,-1,-1]]) / np.sqrt(n_feat)
energy = 0
for i in range(0,n_classes):
  energy = energy + np.dot(M_Means[i],M_Means[i])
energy = energy / n_classes
SNR = 10 ** (SNR/10); sig = energy / SNR; sig = sig / n_feat
M_covar = np.zeros(shape=(2,3,3)); sigma = sig * np.array([1,1])
for i in range(0,n_classes):
  M_covar[i,:,:] = sigma[i] * np.eye(N=n_feat,M=n_feat)
print(sigma)
from numpy.random import default_rng
rng = default_rng(seed=5)
xdata = np.empty((0,n_feat), float); labels = np.empty((0), int)
for i in range(0,n_classes):
  ydata = rng.multivariate_normal(mean=M_Means[i,:], cov= M_covar[i,:,:], size= n_samples)
  ylabels = i * np.ones((n_samples),dtype=int)
  xdata = np.append(xdata, ydata, axis=0); labels = np.append(labels, ylabels, axis=0)
print(xdata[:2])
rp = np.random.permutation(len(xdata)); xdatas = xdata[rp]; xlabs = labels[rp]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True).fit(xdatas, xlabs)
print('LDA err', 1-accuracy_score(xlabs, lda.predict(xdatas)), confusion_matrix(xlabs, lda.predict(xdatas)).tolist())
print(lda.covariance_)
qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(xdatas, xlabs)
print('QDA err', 1-accuracy_score(xlabs, qda.predict(xdatas)))
X_test = np.empty((0,n_feat), float); y_test = np.empty((0), int)
for i in range(0,n_classes):
  ydata = rng.multivariate_normal(mean=M_Means[i,:], cov= M_covar[i,:,:], size= n_samples)
  X_test = np.append(X_test, ydata, axis=0); y_test = np.append(y_test, i*np.ones(n_samples,dtype=int), axis=0)
print('LDA test err', 1-accuracy_score(y_test, lda.predict(X_test)), 'QDA test', 1-accuracy_score(y_test, qda.predict(X_test)))
