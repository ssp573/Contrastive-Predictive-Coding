from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

with open('cpc_features_CIFAR10.pkl','rb') as f:
	data=pkl.load(f)
with open('targets_CIFAR10.pkl','rb') as f:
	target=pkl.load(f)

#data = preprocessing.normalize(data)
data2D=TSNE(n_components=2).fit_transform(data)
#data2D=tsne.transform(data)
print("done TSNE")
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(data)
print("done kmeans")
plt.figure('K-means with 10 clusters with real targets')
plt.scatter(data2D[:, 0], data2D[:, 1], c=target)
plt.savefig('CIFAR10-cluster_true_targets.png')
plt.clf()
plt.figure('K-means with 10 clusters with cluster labels')
plt.scatter(data2D[:, 0], data2D[:, 1], c=kmeans.labels_)
plt.savefig('CIFAR10-cluster_labels.png')



'''
correct=0
for i in range(len(kmeans.labels_)):
	if kmeans.labels_[i]==target[i]:
		correct+=1
print("Accuracy:",correct/len(kmeans.labels_))
'''

