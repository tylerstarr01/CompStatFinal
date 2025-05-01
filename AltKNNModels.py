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
        return _knn_fjlt(time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref,
                        distance, distance_preprocessing, target_dim, batch_size)

class AltKNN(BaseKNN):