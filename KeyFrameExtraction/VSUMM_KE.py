import numpy as np
from sklearn.cluster import KMeans

def VSUMM(descriptors, shot_frame_number):
	"""
	K-means clustering to generate video summary
	:param descriptors: VSUMM descriptors
	:param shot_frame_number: start index of shot
	:return: frame indices
	"""
	# Percentage of keyframes taken
	percent = int(2)

	# Minimum of one frame taken per shot
	num_centroids=int(percent*len(descriptors)/100)
	if num_centroids == 0:
		num_centroids = 1

	# Clustering and transforming using k-means
	kmeans=KMeans(n_clusters=num_centroids).fit(descriptors)
	hist_transform=kmeans.transform(descriptors)

	frame_indices=[]
	# Picking the keyframes from the cluster
	for cluster in range(hist_transform.shape[1]):
		frame_indices.append(np.argmin(hist_transform.T[cluster])+shot_frame_number)

	frame_indices=sorted(frame_indices)

	return frame_indices

