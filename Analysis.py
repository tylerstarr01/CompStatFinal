from claspy.nearest_neighbour import KSubsequenceNeighbours
from tssb.utils import load_time_series_segmentation_datasets
from tssb.evaluation import covering
from claspy.segmentation import BinaryClaSPSegmentation
import os
import warnings
from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from claspy.clasp import ClaSPEnsemble, ClaSP
from claspy.utils import check_input_time_series, check_excl_radius
from claspy.window_size import map_window_size_methods
from claspy.nearest_neighbour import *
from claspy.nearest_neighbour import _sliding_dot
import hnswlib
import gc
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from sklearn.neighbors import KDTree

def _knn_hnsw(time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
         distance, distance_preprocessing, batch_size=100, num_threads=4):
    l = len(time_series) - window_size + 1
    exclusion_radius = np.int64(window_size / 2)
    dims = time_series.shape[1]
    total_dim = window_size * dims

    # Initialize output arrays
    knns = np.full((end - start, len(tcs) * k_neighbours), -1, dtype=np.int64)
    dists = np.full((end - start, len(tcs) * k_neighbours), np.inf, dtype=np.float32)

    # 1. Memory-efficient preprocessing
    def preprocess_subsequence(ss):
        mean = np.mean(ss)
        std = np.std(ss) + 1e-8
        return (ss - mean) / std

    subsequences = np.zeros((l, total_dim), dtype=np.float32)
    for i in range(l):
        ss = time_series[i:i + window_size].reshape(-1)
        subsequences[i] = preprocess_subsequence(ss)

    # 2. Optimized HNSW parameters
    M = min(32, total_dim)  # Reduced from 64 for better memory usage
    ef_construction = min(200, l // 20)  # Reduced from 400 for faster construction
    ef_query = max(k_neighbours * 4, 50)  # Reduced from k*8 for faster queries

    # 3. Initialize HNSW with optimized parameters
    index = hnswlib.Index(space='l2', dim=total_dim)
    index.init_index(max_elements=l, ef_construction=ef_construction, M=M)
    index.set_num_threads(num_threads)
    
    # 4. Add items in batches with memory management
    for i in range(0, l, batch_size):
        batch = subsequences[i:i + batch_size]
        index.add_items(batch)
        gc.collect()  # Clear memory after each batch

    index.set_ef(ef_query)

    # 5. Parallel query processing
    def process_batch(batch_start, batch_end):
        batch_queries = subsequences[batch_start:batch_end]
        labels, distances = index.knn_query(batch_queries, k=k_neighbours * 2)
        return labels, distances

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            futures.append(executor.submit(process_batch, batch_start, batch_end))

        # Process results with temporal constraints
        for i, future in enumerate(futures):
            batch_start = start + i * batch_size
            labels, distances = future.result()
            
            for j, (query_labels, query_dists) in enumerate(zip(labels, distances)):
                order = batch_start + j
                
                for tc_idx, (lbound, ubound) in enumerate(tcs):
                    if order < lbound or order >= ubound:
                        continue

                    # Filter by temporal constraints and exclusion zone
                    mask = (query_labels >= lbound) & (query_labels < ubound)
                    mask &= ~((query_labels >= order - exclusion_radius) & 
                             (query_labels < order + exclusion_radius))
                    
                    valid_labels = query_labels[mask]
                    valid_dists = query_dists[mask]

                    if len(valid_labels) >= k_neighbours:
                        idx = np.argsort(valid_dists)[:k_neighbours]
                        knns[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = valid_labels[idx]
                        dists[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = valid_dists[idx]
                    else:
                        # Fallback for insufficient neighbors
                        tc_range = np.arange(lbound, ubound - window_size + 1)
                        tc_range = tc_range[(tc_range < order - exclusion_radius) | 
                                          (tc_range >= order + exclusion_radius)]
                        if len(tc_range) >= k_neighbours:
                            knns[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = tc_range[:k_neighbours]
                            dists[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = np.linalg.norm(
                                subsequences[order] - subsequences[tc_range[:k_neighbours]], axis=1
                            )

    return dists, knns

def _get_fjlt_matrix(total_dim, target_dim):
    """Get or create FJLT matrix with caching"""
    # Create a unique hash for the matrix parameters
    matrix_hash = hashlib.md5(f"{total_dim}_{target_dim}".encode()).hexdigest()
    cache_dir = "fjlt_cache"
    cache_file = os.path.join(cache_dir, f"fjlt_matrix_{matrix_hash}.pkl")
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Try to load from cache
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Generate new matrix if not in cache
    R = np.random.choice([-1, 1], size=(target_dim, total_dim))
    H = np.abs(np.fft.fft(R, axis=1)) / np.sqrt(total_dim)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(H, f)
    
    return H

def _knn_fjlt(time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
              distance, distance_preprocessing, target_dim=4, batch_size=100):
    l = len(time_series) - window_size + 1
    exclusion_radius = np.int64(window_size / 2)
    dims = time_series.shape[1]
    total_dim = window_size * dims
    target_dim = int(min(total_dim, 4*np.log(total_dim)))
    
    # Debug print to understand dimensions
    # print(f"Debug - Time series shape: {time_series.shape}")
    # print(f"Debug - Window size: {window_size}, Dimensions: {dims}")
    # print(f"Debug - Total dim: {total_dim}, Target dim: {target_dim}")

    # Initialize output arrays
    knns = np.full((end - start, len(tcs) * k_neighbours), -1, dtype=np.int64)
    dists = np.full((end - start, len(tcs) * k_neighbours), np.inf, dtype=np.float32)

    # 1. Preprocess subsequences
    def preprocess_subsequence(ss):
        mean = np.mean(ss)
        std = np.std(ss) + 1e-8
        return (ss - mean) / std

    subsequences = np.zeros((l, total_dim), dtype=np.float32)
    for i in range(l):
        ss = time_series[i:i + window_size].reshape(-1)
        subsequences[i] = preprocess_subsequence(ss)

    # 2. Get FJLT matrix from cache or generate new one
    H = _get_fjlt_matrix(total_dim, target_dim)
    
    # 3. Apply FJLT to reduce dimensionality
    reduced_subsequences = np.dot(subsequences, H.T)
    
    # Print dimensionality reduction information
    print(f"FJLT Dimensionality Reduction: {total_dim} -> {target_dim} ({(target_dim/total_dim)*100:.2f}% of original dimension)")

    # 4. Process in batches
    for tc_idx, (lbound, ubound) in enumerate(tcs):
        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            if batch_start >= ubound or batch_end <= lbound:
                continue

            batch_queries = reduced_subsequences[batch_start:batch_end]
            tc_subsequences = reduced_subsequences[lbound:ubound - window_size + 1]
            
            for i, query in enumerate(batch_queries):
                order = batch_start + i
                if order < lbound or order >= ubound:
                    continue

                # Get valid indices
                valid_indices = np.arange(lbound, ubound - window_size + 1)
                mask = ~((valid_indices >= order - exclusion_radius) & 
                        (valid_indices < order + exclusion_radius))
                valid_indices = valid_indices[mask]
                
                if len(valid_indices) >= k_neighbours:
                    # Compute distances
                    distances = np.linalg.norm(query - tc_subsequences[valid_indices - lbound], axis=1)
                    
                    # Get k nearest neighbors
                    idx = np.argsort(distances)[:k_neighbours]
                    nn_indices = valid_indices[idx]
                    nn_distances = distances[idx]

                    knns[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = nn_indices
                    dists[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = nn_distances

    return dists, knns

def _knn_fjlt_kdtree(time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
                     distance, distance_preprocessing, target_dim=4, batch_size=100):
    """
    Alternate FJLT+KNN implementation using a k-d tree for neighbor search after projection.
    """
    l = len(time_series) - window_size + 1
    exclusion_radius = np.int64(window_size / 2)
    dims = time_series.shape[1]
    total_dim = window_size * dims
    target_dim = int(min(total_dim, 4 * np.log(total_dim)))

    # Preprocess subsequences
    def preprocess_subsequence(ss):
        mean = np.mean(ss)
        std = np.std(ss) + 1e-8
        return (ss - mean) / std

    subsequences = np.zeros((l, total_dim), dtype=np.float32)
    for i in range(l):
        ss = time_series[i:i + window_size].reshape(-1)
        subsequences[i] = preprocess_subsequence(ss)

    # FJLT projection
    H = _get_fjlt_matrix(total_dim, target_dim)
    reduced_subsequences = np.dot(subsequences, H.T)
    print(f"FJLT Dimensionality Reduction (KDTree): {total_dim} -> {target_dim} ({(target_dim/total_dim)*100:.2f}% of original dimension)")

    # Output arrays
    knns = np.full((end - start, len(tcs) * k_neighbours), -1, dtype=np.int64)
    dists = np.full((end - start, len(tcs) * k_neighbours), np.inf, dtype=np.float32)

    # For each temporal constraint, build a KDTree and query
    for tc_idx, (lbound, ubound) in enumerate(tcs):
        tc_indices = np.arange(lbound, ubound - window_size + 1)
        if len(tc_indices) == 0:
            continue
        tc_subsequences = reduced_subsequences[tc_indices]
        kdtree = KDTree(tc_subsequences)

        for order in range(start, end):
            if order < lbound or order >= ubound:
                continue
            query = reduced_subsequences[order].reshape(1, -1)
            # Exclude indices within exclusion radius
            valid_indices = tc_indices[(tc_indices < order - exclusion_radius) | (tc_indices >= order + exclusion_radius)]
            if len(valid_indices) < k_neighbours:
                continue
            valid_subseqs = reduced_subsequences[valid_indices]
            valid_tree = KDTree(valid_subseqs)
            dist, ind = valid_tree.query(query, k=k_neighbours)
            knns[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = valid_indices[ind[0]]
            dists[order - start, tc_idx * k_neighbours:(tc_idx + 1) * k_neighbours] = dist[0]

    return dists, knns

class BaseKNN(KSubsequenceNeighbours):
    def __init__(self, window_size=10, k_neighbours=10, distance="euclidean",
                 n_jobs=1, **kwargs):
        super().__init__(window_size=window_size, k_neighbours=k_neighbours,
                        distance=distance, n_jobs=n_jobs)
        self.kwargs = kwargs

    def fit(self, time_series, temporal_constraints=None):
        time_series = check_input_time_series(time_series)

        if time_series.shape[0] < self.window_size * self.k_neighbours:
            raise ValueError("Time series must at least have k_neighbours*window_size data points.")

        self.time_series = time_series

        if temporal_constraints is None:
            self.temporal_constraints = np.asarray([(0, time_series.shape[0])], dtype=int)
        else:
            self.temporal_constraints = temporal_constraints

        pranges = []
        n_jobs = self.n_jobs

        while time_series.shape[0] // n_jobs < self.window_size * self.k_neighbours and n_jobs != 1:
            n_jobs -= 1

        bin_size = time_series.shape[0] // n_jobs

        for idx in range(n_jobs):
            start = idx * bin_size
            end = min((idx + 1) * bin_size, len(time_series) - self.window_size + 1)
            if end > start: pranges.append((start, end))

        self.distances, self.offsets = self._parallel_knn(
            time_series,
            self.window_size,
            self.k_neighbours,
            pranges,
            self.temporal_constraints,
            self.distance,
            self.distance_preprocessing,
            **self.kwargs
        )
        return self

    def _parallel_knn(self, time_series, window_size, k_neighbours, pranges, tcs, distance, distance_preprocessing, **kwargs):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=len(pranges)//2, mp_context=ctx) as executor:
            futures = []
            for start, end in pranges:
                futures.append(executor.submit(
                    self._knn_chunk,
                    time_series.copy(),
                    start,
                    end,
                    window_size,
                    k_neighbours,
                    tcs,
                    distance,
                    distance_preprocessing,
                    **kwargs
                ))

            results = [f.result() for f in futures]

        # Reassemble results
        dists = np.vstack([r[0] for r in results])
        knns = np.vstack([r[1] for r in results])
        return dists, knns

    def _knn_chunk(self, ts_copy, start, end, window_size, k_neighbours, tcs, distance, distance_preprocessing, **kwargs):
        # Precompute dot products locally
        l = len(ts_copy) - window_size + 1
        dot_first = np.zeros((ts_copy.shape[1], l), dtype=np.float64)
        dot_ref = np.zeros((ts_copy.shape[1], l), dtype=np.float64)
        dot_rolled = dot_first.copy()

        for dim in range(ts_copy.shape[1]):
            dot_first[dim] = _sliding_dot(ts_copy[:window_size, dim], ts_copy[:, dim])
            dot_ref[dim] = dot_first[dim].copy()

        return self._knn_impl(
            ts_copy, start, end, window_size, k_neighbours, tcs,
            dot_first, dot_ref, distance, distance_preprocessing, **kwargs
        )

    def _knn_impl(self, *args, **kwargs):
        """To be implemented by subclasses"""
        raise NotImplementedError

class HNSWKNN(BaseKNN):
    def _knn_impl(self, time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
                  distance, distance_preprocessing, batch_size=100, num_threads=4):
        return _knn_hnsw(time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
                   distance, distance_preprocessing, batch_size, num_threads)

class FJLTKNN(BaseKNN):
    def _knn_impl(self, time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
                  distance, distance_preprocessing, target_dim=32, batch_size=100):
        return _knn_fjlt_kdtree(time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
                        distance, distance_preprocessing, target_dim, batch_size)

class AltKNN(BaseKNN):
    def __init__(self, window_size=10, k_neighbours=10, distance="euclidean",
                 n_jobs=1, method="hnsw", **kwargs):
        super().__init__(window_size=window_size, k_neighbours=k_neighbours,
                        distance=distance, n_jobs=n_jobs, **kwargs)
        self.method = method
        self.impl = HNSWKNN(window_size, k_neighbours, distance, n_jobs, **kwargs) if method == "hnsw" else \
                   FJLTKNN(window_size, k_neighbours, distance, n_jobs, **kwargs)

    def _knn_impl(self, *args, **kwargs):
        return self.impl._knn_impl(*args, **kwargs)

class AltClaSPEnsemble(ClaSPEnsemble):
    def __init__(self, method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def fit(self, time_series, knn=None, validation="significance_test", threshold=1e-15):
        time_series = check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series
        tcs = self._calculate_temporal_constraints()

        if knn is None:
            knn = AltKNN(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
                n_jobs=self.n_jobs,
                method=self.method
            ).fit(time_series, temporal_constraints=tcs)

        best_score, best_tc, best_clasp = -np.inf, None, None

        for idx, (lbound, ubound) in enumerate(tcs):
            clasp = ClaSP(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                score=self.score_name,
                excl_radius=self.excl_radius,
                n_jobs=self.n_jobs
            ).fit(time_series[lbound:ubound], knn=knn.constrain(lbound, ubound))

            clasp.profile = (clasp.profile + (ubound - lbound) / time_series.shape[0]) / 2

            if clasp.profile.max() > best_score or best_clasp is None and idx == tcs.shape[0] - 1:
                best_score = clasp.profile.max()
                best_tc = (lbound, ubound)
                best_clasp = clasp
            else:
                if self.early_stopping is True: break

            if self.early_stopping is True and best_clasp.split(validation=validation, threshold=threshold) is not None:
                break

        self.profile = np.full(shape=time_series.shape[0] - self.window_size + 1, fill_value=-np.inf, dtype=np.float64)

        if best_clasp is not None:
            self.knn = best_clasp.knn
            self.lbound, self.ubound = best_tc
            self.profile[self.lbound:self.ubound - self.window_size + 1] = best_clasp.profile
        else:
            self.knn = knn
            self.lbound, self.ubound = 0, self.time_series.shape[0]

        self.is_fitted = True
        return self

class AltBinaryClaSPSegmentation(BinaryClaSPSegmentation):
    def __init__(self, method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        # self.window_size = 'fft'

    def fit(self, time_series):
        time_series = check_input_time_series(time_series)

        if isinstance(self.window_size, str):
            window_sizes = []
            print(f"\nCalculating window size for dataset with shape: {time_series.shape}")
            
            for dim in range(time_series.shape[1]):
                raw_window_size = map_window_size_methods(self.window_size)(time_series[:, dim])
                # print(self.window_size)
                adjusted_window_size = max(3, raw_window_size // 2)
                window_sizes.append(adjusted_window_size)
                # print(f"Dimension {dim}: Raw window size = {raw_window_size}, Adjusted = {adjusted_window_size}")

            # Use mean window size instead of min to better capture dataset characteristics
            self.window_size = int(np.mean(window_sizes)) if len(window_sizes) > 0 else 10
            print(f"Final window size: {self.window_size}\n")

        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            warnings.warn(
                "Time series must at least have 2*min_seg_size data points for segmentation. Try setting "
                "a smaller window size.")
            self.n_segments = 1
            self.window_size = min(self.window_size, time_series.shape[0] // 2)

        self.time_series = time_series
        self.n_timepoints = time_series.shape[0]

        if self.threshold == "default":
            if self.validation == "score_threshold":
                self.threshold = 0.75
            elif self.validation == "significance_test":
                if self.time_series.shape[1] == 1:
                    self.threshold = 1e-15
                else:
                    self.threshold = 1e-30

        self.queue = PriorityQueue()
        self.clasp_tree = []

        if self.n_segments == "learn":
            self.n_segments = time_series.shape[0] // self.min_seg_size

        if self.n_segments > 1:
            prange = 0, time_series.shape[0]
            clasp_model = AltClaSPEnsemble(
                n_estimators=self.n_estimators,
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
                score=self.score,
                early_stopping=self.early_stopping,
                excl_radius=self.excl_radius,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                method=self.method
            )

            clasp = clasp_model.fit(time_series, validation=self.validation, threshold=self.threshold)

            cp = clasp.split(validation=self.validation, threshold=self.threshold)

            if cp is not None and self._cp_is_valid(cp, []):
                self.clasp_tree.append((prange, clasp))
                self.queue.put((-clasp.profile[cp], len(self.clasp_tree) - 1))

            profile = clasp.profile
        else:
            profile = np.full(shape=self.n_timepoints - self.window_size + 1, fill_value=-np.inf, dtype=np.float64)

        change_points = []
        scores = []

        for idx in range(self.n_segments - 1):
            # happens if no valid change points exist anymore
            if self.queue.empty() is True: break

            priority, clasp_tree_idx = self.queue.get()
            (lbound, ubound), clasp = self.clasp_tree[clasp_tree_idx]
            cp = lbound + clasp.split(validation=self.validation, threshold=self.threshold)

            profile[lbound:ubound - self.window_size + 1] = np.max(
                [profile[lbound:ubound - self.window_size + 1], clasp.profile], axis=0)

            change_points.append(cp)
            scores.append(-priority)

            if len(change_points) == self.n_segments - 1: break

            lrange, rrange = (lbound, cp), (cp, ubound)

            for prange in (lrange, rrange):
                self._local_segmentation(*prange, change_points)

        sorted_cp_args = np.argsort(change_points)
        self.change_points, self.scores = np.asarray(change_points)[sorted_cp_args], np.asarray(scores)[sorted_cp_args]

        profile[np.isinf(profile)] = np.nan
        self.profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

        self.is_fitted = True
        return self

    def constrain(self, lbound, ubound):
        if (lbound, ubound) not in self.temporal_constraints:
            raise ValueError(f"({lbound},{ubound}) is not a valid temporal constraint.")

        for idx, tc in enumerate(self.temporal_constraints):
            if tuple(tc) == (lbound, ubound):
                tc_idx = idx

        ts = self.time_series[lbound:ubound]
        distances = self.distances[lbound:ubound - self.window_size + 1,
                    tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours]
        offsets = self.offsets[lbound:ubound - self.window_size + 1,
                  tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours] - lbound

        knn = AltKNN(
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            distance=self.distance_name,
            method=self.method
        )

        knn.time_series = ts
        knn.temporal_constraints = np.asarray([(0, ts.shape[0])], dtype=int)
        knn.distances, knn.offsets = distances, offsets
        return knn

import time

def run_analysis(tssb, method):
    results = []
    for _, (ts_name, window_size, cps, ts) in tssb.iterrows():
        start_time = time.time()
        found_cps = method.fit_predict(ts)
        end_time = time.time()
        score = covering({0: cps}, found_cps, ts.shape[0])
        
        try:
            m = method.method
        except:
            m = "ClaSP"

        results.append({
            'Algo': m,
            'Time Series': ts_name,
            'Found Change Points': found_cps.tolist(),
            'True Change Points': cps,
            'Score': score,
            'Run Time': (end_time - start_time)
        })
    return results

def generate_synthetic_multivariate_ts(N=10000, k=5, dims=100, seed=42):
    """
    Generate a synthetic multivariate time series with k change points.
    Between change points, data is generated using different stochastic processes.
    
    Parameters:
    -----------
    N : int
        Length of the time series
    k : int
        Number of change points
    dims : int
        Number of dimensions (features) in the time series
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing the time series data in TSSB format
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate k+1 segments with random lengths
    rv = np.random.rand(k+1)
    rvs = rv.sum()
    segment_lengths = np.random.multinomial(N - k - 1, rv/rvs) + 1
    # segment_lengths = np.random.uniform(N/k, size=k).astype(int)
    segment_lengths = np.array([100, 50, 400, 150])
    change_points = np.cumsum(segment_lengths[:-1])
    # change_points = (np.array([990, 3430, 4250])/10).astype(int)
    # segment_lengths = (np.array([990, 3430-990, 4250-3430, N-4250])/10).astype(int)
    # Initialize time series array
    ts = np.zeros((N, dims))
    
    # Define different stochastic processes
    def ar_process(length, dims, params):
        """Generate AR(1) process"""
        noise = np.random.normal(0, 1, (length, dims))
        data = np.zeros((length, dims))
        data[0] = noise[0]
        for i in range(1, length):
            data[i] = params['phi'] * data[i-1] + noise[i]
        return data
    
    def ma_process(length, dims, params):
        """Generate MA(1) process"""
        noise = np.random.normal(0, 1, (length+1, dims))
        data = np.zeros((length, dims))
        for i in range(length):
            data[i] = noise[i] + params['theta'] * noise[i+1]
        return data
    
    def random_walk(length, dims, params):
        """Generate random walk process"""
        steps = np.random.normal(0, params['sigma'], (length, dims))
        return np.cumsum(steps, axis=0)
    
    def wiener_process(length, dims, params):
        """Generate Wiener process (standard Brownian motion)"""
        dt = 0.5  # time step
        noise = np.random.normal(0, 1, (length, dims))
        data = np.zeros((length, dims))
        data[0] = params.get('initial_value', 0.0)  # default initial value is 0.0
        
        # Wiener process: W_t = W_0 + σ√t * Z, where Z ~ N(0,1)
        for i in range(1, length):
            data[i] = data[i-1] + params['sigma'] * np.sqrt(dt) * noise[i]
        
        return data
    
    # List of possible processes with their parameters
    processes = [
        (ar_process, {'phi': 0.8}),
        (ar_process, {'phi': -0.8})
        # (ma_process, {'theta': 0.8}),
        # (random_walk, {'sigma': 0.75}),
        # (random_walk, {'sigma': 0.1}),
        # (wiener_process, {'sigma': 0.3, 'initial_value': 0.0}),
        # (wiener_process, {'sigma': 0.6, 'initial_value': 0.0}),
        # (wiener_process, {'sigma': 0.1, 'initial_value': 0.0})  # Standard Brownian motion
        # (wiener_process, {'sigma': 0.6, 'initial_value': 0.0})   # Higher volatility Brownian motion
    ]
    

    indices_chosen = []
    # Generate each segment
    start_idx = 0
    np.random.seed(seed)
    my_processes = [0, 1, 1, 0]
    for i, length in enumerate(segment_lengths):
        # Randomly select a process for this segment
        idx_chosen = np.random.randint(len(processes))
        # print(idx_chosen)
        idx_chosen = my_processes[i]
        # print(idx_chosen)
        process_func, params = processes[idx_chosen]
        indices_chosen.append(idx_chosen)
        
        # Generate the segment
        # segment = process_func(length, dims, params)
        # segment = wiener_process(length, dims, {'sigma': 0.3, 'initial_value': 0.0})
        segment = process_func(length, dims, params)
        
        # Add some random scaling and shifting to make segments more distinct
        # scale = np.random.uniform(0.5, 2.0, dims)
        # shift = np.random.uniform(-5, 5, dims)
        # segment = segment * scale + shift
        
        # Add the segment to the time series
        ts[start_idx:start_idx+length] = segment
        start_idx += length
    
    # Create output in TSSB format
    print(indices_chosen)
    true_change_points = []
    last = -1
    incl_zero = [0] + change_points.tolist()
    for i, cp in zip(indices_chosen, incl_zero):
        if i != last:
            if cp != 0:
                true_change_points.append(cp)
            last = i
    # print(change_points)
    # print(indices_chosen)
    print(true_change_points)

    return {
        'dataset': f'Synthetic_{dims}',
        'window_size': 100,  # Default window size
        'change_points': true_change_points,
        'time_series': ts
    }

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate synthetic data
    synthetic_data = [generate_synthetic_multivariate_ts(N=700, k=3, dims=x, seed=42) for x in [1]]
    
    # Create DataFrame for synthetic data
    synthetic_df = pd.DataFrame(synthetic_data)
    plt.plot(synthetic_df.time_series.values[0], linewidth=0.5)
    plt.title(f'Time Series Example with Change Points')
    # Add vertical lines at true change points
    for cp in synthetic_df.change_points.values[0]:
        plt.axvline(cp, color='red', linestyle='--', linewidth=1)
    plt.show()
    
    # Load univariate datasets
    datasets = ["ArrowHead"]
    tssb = load_time_series_segmentation_datasets(names=datasets)
    
    # # Load multivariate datasets
    # multi_datasets = ['BeetleFly', 'BirdChicken']
    # multi_data = load_time_series_segmentation_datasets(names=multi_datasets)
    print(synthetic_df)
    
    # Create a single entry with combined multivariate time series
    # multi_ts = np.stack(multi_data['time_series'].values)
    
    # Create a new DataFrame for multivariate analysis
    # multi_df = pd.DataFrame({
    #     'dataset': ['Multi'],
    #     'window_size': [10],
    #     'change_points': [[1280]],
    #     'time_series': [multi_ts.T]
    # })
    
    # Combine all datasets
    combined_tssb = pd.concat([tssb, synthetic_df])
    # combined_tssb = synthetic_df
    # print(combined_tssb)
    
    # Run analysis on all datasets
    classifiers = [
        BinaryClaSPSegmentation(window_size='fft'),
        AltBinaryClaSPSegmentation(method="fjlt"),
        # AltBinaryClaSPSegmentation(method="hnsw")
    ]
    results = [pd.DataFrame(run_analysis(combined_tssb, x)) for x in classifiers]
    a = pd.concat(results)
    print(a.sort_values(by=['Time Series', 'Algo'], ascending=False))
    a.sort_values(by=['Time Series', 'Algo'], ascending=False).to_csv('simulated_kdtree.csv', index=False)
    print(combined_tssb.change_points)

