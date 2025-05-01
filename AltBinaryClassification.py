
class AltBinaryClaSPSegmentation(BinaryClaSPSegmentation):
    def __init__(self, method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def fit(self, time_series):
        time_series = check_input_time_series(time_series)

        if isinstance(self.window_size, str):
            window_sizes = []

            for dim in range(time_series.shape[1]):
                window_sizes.append(max(3, map_window_size_methods(self.window_size)(time_series[:, dim]) // 2))

            self.window_size = int(np.min(window_sizes)) if len(window_sizes) > 0 else 10

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
