import numpy as np
import casadi as ca

class TrackBoundary:
    def __init__(self, track_data):
        """
        Initialize track boundaries.
        
        Args:
            track_data (dict):
                - center_line: Nx2 array of (x, y) centerline points
                - left_boundary: Nx2 array of left boundary points
                - right_boundary: Nx2 array of right boundary points
                - track_width: track width (m)
        """
        self.center_line = track_data['center_line']
        self.left_boundary = track_data['left_boundary']
        self.right_boundary = track_data['right_boundary']
        self.track_width = track_data['track_width']
        
    def get_closest_point(self, x, y):
        """
        Find closest point on centerline to given position.
        
        Returns:
            idx: index of closest centerline point
            distance: distance to centerline
            progress: progress along track (m)
        """
        distances = np.sqrt((self.center_line[:, 0] - x)**2 + 
                           (self.center_line[:, 1] - y)**2)
        idx = np.argmin(distances)
        
        # Calculate progress (arc length along centerline)
        progress = self._calculate_arc_length(idx)
        
        return idx, distances[idx], progress
    
    def _calculate_arc_length(self, idx):
        """Calculate arc length to given index."""
        if idx == 0:
            return 0.0
        
        segments = np.diff(self.center_line[:idx+1], axis=0)
        distances = np.sqrt(np.sum(segments**2, axis=1))
        return np.sum(distances)
    
    def boundary_violation_cost(self, x, y):
        """
        Potential field cost for boundary violation.
        Returns high cost when approaching boundaries.
        
        This is the key innovation from the paper - soft constraints!
        """
        # Find closest boundaries
        left_dist = self._distance_to_boundary(x, y, self.left_boundary)
        right_dist = self._distance_to_boundary(x, y, self.right_boundary)
        
        # Potential field parameters (tune these!)
        safety_margin = 0.3  # Start penalizing 0.3m from boundary
        k_boundary = 100.0   # Penalty weight
        
        # Exponential potential field
        cost_left = k_boundary * ca.exp(-left_dist / safety_margin)
        cost_right = k_boundary * ca.exp(-right_dist / safety_margin)
        
        return cost_left + cost_right
    
    def _distance_to_boundary(self, x, y, boundary_points):
        """Calculate minimum distance to boundary."""
        # For CasADi compatibility, we'll use a simplified version
        # In practice, you might want to use nearest segments
        distances = ca.sqrt((boundary_points[:, 0] - x)**2 + 
                           (boundary_points[:, 1] - y)**2)
        return ca.mmin(distances)
    
    @staticmethod
    def create_simple_oval_track(length=10.0, width=5.0, track_width=2.0):
        """
        Create a simple oval track for testing.
        
        Args:
            length: straight section length (m)
            width: track width (m)
            track_width: racing surface width (m)
        """
        # Create centerline (oval)
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Parametric oval
        a = length / 2  # Semi-major axis
        b = width / 2   # Semi-minor axis
        
        x_center = a * np.cos(theta)
        y_center = b * np.sin(theta)
        
        center_line = np.column_stack([x_center, y_center])
        
        # Create boundaries (offset from centerline)
        normals = np.column_stack([
            -np.sin(theta),
            np.cos(theta)
        ])
        
        left_boundary = center_line + normals * (track_width / 2)
        right_boundary = center_line - normals * (track_width / 2)
        
        track_data = {
            'center_line': center_line,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary,
            'track_width': track_width
        }
        
        return TrackBoundary(track_data)