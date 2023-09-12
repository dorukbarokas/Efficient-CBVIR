import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def blockclustering(feature_vectors, shot_frame_number):
    """
    Clusters each frame into 9 blocks for clustering
    :param feature_vectors: flatten histogram vectors for each frame
    :param shot_frame_number: shot index
    :return: indices
    """
    minf = 63  # number of singular values and singular vectors to compute (must satisfy 1 <= k <= kmax,
    # where kmax=min(M, N), M = is 1944 (histogram vector for each frame)
    if (len(feature_vectors) < minf+1):   # for shots with a frame amount smaller than 63
        minf = len(feature_vectors)-1 # scale minf down to the amount of frames -1
    strminf = "v" + str(minf+1)
    arr = np.empty((0, 1944), int)  # initializing 1944 dimensional array to store 'flattened' color histograms

    for feature_vector in feature_vectors:
        arr = np.vstack((arr, feature_vector))  # appending each one-dimensinal vector to generate N*M matrix (where N is number of frames
        # and M is 1944)

    final_arr = arr.transpose()  # transposing so that i will have all frames in columns i.e M*N dimensional matrix
    # where M is 1944 and N is number of frames
    #print(final_arr.shape)
    #print("Original frame amount:" + str(count))

    A = csc_matrix(final_arr, dtype=float) # compressed sparse column matrix (easier to perform arithmetic on)

    #top 63 singular values from 76082 to 508
    u, s, vt = svds(A, k = minf)
    #s: singular values
    #u: unitary matrix
    #vt :

    v1_t = vt.transpose()

    projections = v1_t @ np.diag(s) #the column vectors i.e the frame histogram data has been projected onto the orthonormal basis
    #formed by vectors of the left singular matrix u .The coordinates of the frames in this space are given by v1_t @ np.diag(s)
    # projections is now n_frames * minf ( = 63 by default)

    # dynamic clustering of projected frame histograms to find which all frames are similar i.e make shots
    f = projections
    C = dict()  # empty clusters dict
    for i in range(f.shape[0]):
        C[i] = np.empty((0, minf), int) # minf elements long empty int array

    # initialize by adding first two projected frames in first cluster
    C[0] = np.vstack((C[0], f[0]))
    C[0] = np.vstack((C[0], f[1]))

    E = dict()  # to store centroids of each cluster
    for i in range(projections.shape[0]):
        E[i] = np.empty((0, minf), int) # minf elements long empty int array

    E[0] = np.mean(C[0], axis=0)  # finding centroid of C[0] cluster

    count = 0
    for i in range(2, f.shape[0]): # loop over remaining frames

        # compute similarity to last cluster centroid
        similarity = np.dot(f[i], E[count]) / (
                    (np.dot(f[i], f[i]) ** .5) * (np.dot(E[count], E[count]) ** .5))  # cosine similarity
        # this metric is used to quantify how similar is one vector to other. The maximum value is 1 which indicates they are same
        # and if the value is 0 which indicates they are orthogonal nothing is common between them.
        # Here we want to find similarity between each projected frame and last cluster formed chronologically.

        if similarity < 0.8:  # if the projected frame and last cluster formed  are not similar upto 0.9 cosine value then
            # we assign this data point to newly created cluster and find centroid

            count += 1 # to create a new cluster
            C[count] = np.vstack((C[count], f[i]))  # add frame to newly created cluster
            E[count] = np.mean(C[count], axis=0) # update mean of newly created cluster
        else:  # if they are similar then assign this data point to last cluster formed and update the centroid of the cluster
            C[count] = np.vstack((C[count], f[i]))
            E[count] = np.mean(C[count], axis=0)


    b = []  #find the number of data points in each cluster formed.


    for i in range(f.shape[0]): #loop over all clusters
        b.append(C[i].shape[0]) # add the amount of frames per cluster to b

    last = b.index(0)  #findex index of b where clusters have no frames in them
    b1=b[:last ] # the amount of frames for each cluster with at least one frame in it

    res = [idx for idx, val in enumerate(b1) if val >= 5] #any dense cluster with atleast 5 frames is eligible to
    #make shot.
    #print("Number of keyframes extracted: " + str(len(res))) #so total 25 shots with 46 (71-25) cuts

    GG = C #copying the elements of C to GG, the purpose of  the below code is to label each cluster so later
    #it would be easier to identify frames in each cluster
    for i in range(last):
        p1= np.repeat(i, b1[i]).reshape(b1[i],1)
        GG[i] = np.hstack((GG[i],p1))


    #the purpose of the below code is to append each cluster to get multidimensional array of dimension N*64, N is number of frames
    F=  np.empty((0,minf+1), int)
    for i in range(last):
        F = np.vstack((F,GG[i]))


    #converting F (multidimensional array)  to dataframe
    colnames = []
    for i in range(1, minf+2):
        col_name = "v" + str(i)
        colnames+= [col_name]

    df = pd.DataFrame(F, columns= colnames)

    df[strminf]= df[strminf].astype(int)  #converting the cluster level from float type to integer type
    attribute = strminf
    df1 =  df[getattr(df, attribute).isin(res)]   #filter only those frames which are eligible to be a part of shot or filter those frames who are
    #part of required clusters that have more than 5 frames in it

    new = df1.groupby(strminf).tail(1)[strminf] #For each cluster /group take its last element which summarize the shot i.e key-frame

    new1 = new.index #finding key-frames (frame number so that we can go back get the original picture)
    indices = []
    for i in range(0, len(new1)):
        indices.append(new1[i]+shot_frame_number) # add shot beginning index to each index, to account for the shot boundaries
    return indices

