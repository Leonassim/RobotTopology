import numpy as np

class PointCloudLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.points = None

    def load_points(self):
        try:
            self.points = np.loadtxt(self.filepath)
            assert self.points.shape[1] == 3
            print(f"[OK] Loaded {self.points.shape[0]} points from {self.filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to load points: {e}")
            self.points = None

    def get_summary(self):
        if self.points is None:
            print("No points loaded.")
            return None

        summary = {
            "num_points": self.points.shape[0],
            "mean": np.mean(self.points, axis=0).tolist(),
            "min": np.min(self.points, axis=0).tolist(),
            "max": np.max(self.points, axis=0).tolist()
        }
        return summary

    def downsample(self, voxel_size):
        if self.points is None:
            return None

        # Voxel grid downsampling à la main (simple version)
        rounded = np.round(self.points / voxel_size)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        downsampled = self.points[unique_indices]
        print(f"[INFO] Downsampled to {len(downsampled)} points (voxel size = {voxel_size})")
        return downsampled
    
    def filter_by_y_range(self, y_min, y_max):
        if self.points is None:
            print("No points loaded.")
            return None

        mask = (self.points[:, 1] >= y_min) & (self.points[:, 1] <= y_max)
        filtered = self.points[mask]
        print(f"[INFO] Filtered {len(filtered)} points in Y ∈ [{y_min}, {y_max}]")
        return filtered

