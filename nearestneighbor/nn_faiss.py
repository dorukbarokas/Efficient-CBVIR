"""
Image based video search engine: prototype
This file contains several implementation for NNS of the FAISS package
"""
import faiss
import time
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d


def faiss_flat(image_features, frame_features, k, use_gpu=False):  # GPU and CPU
    """
    Implementation of the FAISS FLAT method (brute-force FAISS)
    :param image_features: The n-dimensional feature vectors of the query images
    :param frame_features: The n-dimensional feature vectors of the keyframes
    :param k: The amount of nearest neighbours to be found
    :param use_gpu: Specify if GPU should be used
    :return: The k closest neighbors, distance, build time and search time per query
    """

    queries, dim = image_features.shape
    t1 = time.time()
    index = faiss.IndexFlatL2(dim)  # Create FAISS index
    index.metric_type = faiss.METRIC_L2  # Set metric to L2

    if use_gpu:  # Move from CPU to GPU
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
        # k = min(2048, k)  # Limit k to 2048 due to GPU limitations
        print("k", k)

    index.add(frame_features)  # Add the vectors
    t2 = time.time()

    dist, idx = index.search(image_features, k=k)  # Find the k nearest neighbours
    t3 = time.time()

    build_time = t2 - t1
    time_per_query = (t3 - t2) / queries

    return idx, dist, build_time, time_per_query


def faiss_hnsw(image_features, frame_features, k, m=24, ef_const=58):
    """
    Implementation of HNSW in FAISS
    :param image_features: The n-dimensional feature vectors of the query images
    :param frame_features: The n-dimensional feature vectors of the keyframes
    :param k: The amount of nearest neighbours to be found
    :param m: The maximum amount of connections per node in the graph
    :param ef_const: Parameter that controls speed/accuracy trade-off during the index construction.
    :return: The k closest neighbors, distance, build time and search time per query
    """
    queries, dim = image_features.shape
    t1 = time.time()

    index = faiss.IndexHNSWFlat(dim, m)  # Create the HSNW index
    index.metric_type = faiss.METRIC_L2  # Set the metric to L2
    # Set HNSW params
    index.hnsw.efConstruction = ef_const
    index.hnsw.efSearch = k

    # Add the data
    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k=k)  # Find the k nearest neighbours
    t3 = time.time()

    build_time = t2 - t1
    time_per_query = (t3 - t2) / queries

    return idx, dist, build_time, time_per_query


def faiss_lsh(image_features, frame_features, k, bitlength_percentage=0.25):  # GPU and CPU
    """
    Implementation of LSH in FAISS
    :param image_features: The n-dimensional feature vectors of the query images
    :param frame_features: The n-dimensional feature vectors of the keyframes
    :param k: The amount of nearest neighbours to be found
    :param bitlength_percentage: The percentage of the vector length that defines the bitlength for the lsh
    :return: The k closest neighbors, distance, build time and search time per query
    """
    queries, dim = image_features.shape
    n_frames, _ = frame_features.shape
    t1 = time.time()

    bitlength_percentage = interpol_lsh(n_frames)  # Interpolate based on test data

    index = faiss.IndexLSH(dim, int(bitlength_percentage * dim))  # Create the LSH index
    index.metric_type = faiss.METRIC_L2  # Set the metric to L2

    # Partition and add data
    print(index.is_trained)
    index.train(frame_features)
    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k=k)  # Find the k nearest neighbours
    t3 = time.time()

    build_time = t2 - t1
    time_per_query = (t3 - t2) / queries

    return idx, dist, build_time, time_per_query


def faiss_pq(image_features, frame_features, k, vsplits=8, nbits=8):
    """
    Implementation of PQ in FAISS
    :param image_features: The n-dimensional feature vectors of the query images
    :param frame_features: The n-dimensional feature vectors of the keyframes
    :param k: The amount of nearest neighbours to be found
    :param vsplits: The amount of splits for the feature vector
    :param nbits: The number of bits to reduce the data to
    :return: The k closest neighbors, distance, build time and search time per query
    """
    queries, dim = image_features.shape
    t1 = time.time()

    index = faiss.IndexPQ(dim, vsplits, nbits)  # Create PQ index
    index.metric_type = faiss.METRIC_L2  # Set the metric to L2

    # Partition and add data
    index.train(frame_features)
    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k=k)  # Find the k nearest neighbours
    t3 = time.time()

    build_time = t2 - t1
    time_per_query = (t3 - t2) / queries

    return idx, dist, build_time, time_per_query


def faiss_ivf(image_features, frame_features, k, use_gpu, splits=2, nprobe=1):
    """
    Implementation of IVF in FAISS
    :param image_features: The n-dimensional feature vectors of the query images
    :param frame_features: The n-dimensional feature vectors of the keyframes
    :param k: The amount of nearest neighbours to be found
    :param use_gpu: Indicates if the GPU should be used
    :param splits: The amount of voronoi cells the data should be split in
    :param nprobe: The amount of voronoi cells to be accessed during search
    :return: The k closest neighbors, distance, build time and search time per query
    """
    queries, dim = image_features.shape
    n_frames, _ = frame_features.shape

    t1 = time.time()

    # Interpolate the data for ivf if it is above the minimum tested amount
    if n_frames >= 270:
        nprobe, splits = interpol_ivf(n_frames, queries)
        nprobe = int(nprobe)
        splits = int(splits)

    quantizer = faiss.IndexFlatL2(dim)  # Create a FLAT index as quantizer
    index = faiss.IndexIVFFlat(quantizer, dim, splits)  # Create the IVF index using the flat quantizer
    index.metric_type = faiss.METRIC_L2  # Set the metric to L2
    index.nprobe = nprobe

    if use_gpu:  # Move from CPU to GPU
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
        k = min(2048, k)

    # Add the data
    index.train(frame_features)
    index.add(frame_features)
    t2 = time.time()

    dist, idx = index.search(image_features, k=k)  # Find the k nearest neigbours
    t3 = time.time()

    build_time = t2 - t1
    time_per_query = (t3 - t2) / queries

    return idx, dist, build_time, time_per_query


def interpol_ivf(n_frames_inter, n_queries_inter):
    """
    2D interpolation function for the IVF params
    :param n_frames_inter: The number of keyframes
    :param n_queries_inter: The number of query images
    :return: The interpolated values for nprobe and nsplits
    """
    # Interpolation data
    n_frames = [270, 270, 270, 270, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100, 8100]
    n_queries = [1, 122, 123, 1000, 1, 4, 5, 10, 11, 89, 90, 258, 259, 508, 509, 100]
    nprobe = [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    splits = [2, 2, 5, 5, 4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 12, 12]
    # Create the interpolation functions
    interpolfunc_nprobe = LinearNDInterpolator((n_queries, n_frames), nprobe, fill_value=1)
    interpolfunc_splits = LinearNDInterpolator((n_queries, n_frames), splits, fill_value=12)
    # Query the given point
    pts = np.array([n_queries_inter, n_frames_inter])
    return interpolfunc_nprobe(pts), interpolfunc_splits(pts)


def interpol_lsh(n_frames_inter):
    """
    1D interpolation function for the LSH params
    :param n_frames_inter: The amount of frames
    :return: The interpolated value for bit percentage
    """
    # Interpolation data
    n_frames = [270, 8100, 50000]
    bitpercentage = [0.04, 0.09, 0.09]
    # Create the interpolation function
    interpolfunc_bitpercentage = interp1d(n_frames, bitpercentage, kind='linear', bounds_error=False,
                                          fill_value=(0.04, 0.09))
    # Query the given point
    pts = np.array(n_frames_inter)
    return interpolfunc_bitpercentage(pts)
