import numpy as np
import cv2
from sklearn.cluster import KMeans

def VSUMM_combi(descriptors, shot_frame_number, skip_num):
	"""
	K-means clustering to generate video summary with histogram threshold
	:param descriptors: VSUMM descriptors
	:param shot_frame_number: start index of shot
	:return: frame indices
	"""

	# Percentage of keyframes taken
	percent = int(2*skip_num) #skip_num: number of frames skipped due to presampling

	# apply histogram threshold to extracted keyframes
	threshold = 0.2

	# converting percentage to actual number
	num_centroids=int(percent*len(descriptors)/100)
	if num_centroids == 0:
		num_centroids = 1
	kmeans=KMeans(n_clusters=num_centroids).fit(descriptors)

	# transforms into cluster-distance space (n_cluster dimensional)
	hist_transform=kmeans.transform(descriptors)

	frame_indices=[]
	for cluster in range(hist_transform.shape[1]):
		frame_indices.append(np.argmin(hist_transform.T[cluster]))
	
	# frames generated in sequence from original video
	frame_indices=sorted(frame_indices)

	# Perform histogram threshold set
	deleted_amnt = 0
	if len(frame_indices) > 1:
		for i in range(1, len(frame_indices)):
			histdiff = cv2.compareHist(descriptors[frame_indices[i-deleted_amnt]], descriptors[frame_indices[i-1-deleted_amnt]], cv2.HISTCMP_BHATTACHARYYA)
			if (histdiff) < threshold:
				frame_indices.pop(i-deleted_amnt) # delete frame because it is too similar to previous
				deleted_amnt += 1

	print("Removed amount of frames after histogram threshold: " + str(deleted_amnt))

	# add shot index to each index
	frame_indices = [element + shot_frame_number for element in frame_indices]

	return frame_indices

