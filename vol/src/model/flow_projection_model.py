import numpy as np
####Input
# Lidar 2x(N,3)
# RGB 2x(H, W, 3)
# Flow (H, W, 2): (2D flow --> GmFlow)

#####Output: 
# H, W, 14 -> Image Pairs(H, W, 6), Depth Map Pair(H, W, 6), Flow (H, W, 2)



def lidarToCamProjection(lidar_pts, T_lidar_to_camera, K, H, W):
    """
    Projects lidar points to the camera image plane and generates a depth map (H, W, 3)
    
    Parameters:
    - lidar_pts: np.ndarray of shape (N, 3), where each row is a 3D lidar point (x, y, z).
    - T_lidar_to_camera: np.ndarray of shape (4, 4), the transformation matrix from lidar to camera coordinates.
    - K: np.ndarray of shape (3, 3), the camera intrinsic matrix.
    - H: int, height of the image.
    - W: int, width of the image.
    
    Returns:
    - depth_map: np.ndarray of shape (H, W, 3), where each pixel contains the (X, Y, Z) depth values.
    - valid_indices: np.ndarray of shape (N,), boolean array indicating valid points.
    """
    
    # Convert to homogeneous coordinates (N, 4)
    homogenous_lidar_pts = np.hstack((lidar_pts, np.ones((lidar_pts.shape[0], 1)))) 
    # Transform lidar points to the camera frame (N, 4)
    projected_lidar_pts = np.dot(T_lidar_to_camera, homogenous_lidar_pts.T).T
    
    # Filter points with positive depth (z > 0)
    valid_indices = projected_lidar_pts[:, 2] > 0
    projected_lidar_pts = projected_lidar_pts[valid_indices]

    # Retain x, y, z in camera frame
    projected_lidar_pts = projected_lidar_pts[:, :3]

    # Project to image plane (u, v) using intrinsic matrix K
    img_pts = (K @ projected_lidar_pts.T).T  # Project (X, Y, Z) -> (u, v, z)
    
    # Normalize by depth (z) to get pixel coordinates
    img_pts[:, 0] /= img_pts[:, 2]  # u = X / Z
    img_pts[:, 1] /= img_pts[:, 2]  # v = Y / Z
    
    # Round to nearest integer and clip to image dimensions
    u = np.clip(np.round(img_pts[:, 0]).astype(int), 0, W-1)
    v = np.clip(np.round(img_pts[:, 1]).astype(int), 0, H-1)

    # Create an empty depth map (H, W, 3)
    depth_map = np.zeros((H, W, 3), dtype=np.float32)
    
    # Assign (X, Y, Z) to the respective pixel (v, u) coordinates
    depth_map[v, u, 0] = projected_lidar_pts[:, 0]  # X values
    depth_map[v, u, 1] = projected_lidar_pts[:, 1]  # Y values
    depth_map[v, u, 2] = projected_lidar_pts[:, 2]  # Z values
    
    return depth_map, valid_indices

 
def fuseInputs(lidar_seq, img_seq, flow, T_lidar_to_camera, K):
    """
    Fuse the lidar data, image sequence, and flow data into a single output.

    Parameters:
    - lidar_seq: np.ndarray of shape (N, 3, 2), where N is the number of lidar points and 2 represents time steps t-1 and t.
    - img_seq: np.ndarray of shape (H, W, 3, 2), RGB image sequence with 2 images (t-1 and t).
    - flow: np.ndarray of shape (H, W, 2), flow between image sequence at time t-1 and t.
    - T_lidar_to_camera: np.ndarray of shape (4, 4), the transformation matrix from lidar to camera coordinates.
    - K: np.ndarray of shape (3, 3), the camera intrinsic matrix.
    
    Returns:
    - output_: np.ndarray of shape (H, W, 14), concatenated data of images, lidar points and flow.
    """
    
    # Unpack LiDAR point clouds and image sequence
    lidar_pts_T0, lidar_pts_T1 = lidar_seq
    img_T0, img_T1 = img_seq

    # Extract image dimensions
    H, W = img_T0.shape[:2]

    # Lidar to Camera Projection for time-step t-1
    depth_map_T0, _ = lidarToCamProjection(lidar_pts_T0, T_lidar_to_camera, K, H, W)
    # Lidar to Camera Projection for time-step t
    depth_map_T1, _ = lidarToCamProjection(lidar_pts_T1, T_lidar_to_camera, K, H, W)


    # Concatenate images, lidar depth maps,and flow (H, W, 14)
    output_ = np.concatenate((img_T0, img_T1, depth_map_T0, depth_map_T1, flow), axis=-1)
    
    return output_