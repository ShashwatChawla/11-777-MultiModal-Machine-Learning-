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

# Vectorized Lidar Projection
def lidarToCamProjectionBatch(lidar_pts_batch, T_lidar_to_camera, K, H, W):
    """
    Projects batched lidar points to the camera image plane and generates depth maps (B, H, W, 3).

    Parameters:
    - lidar_pts_batch: np.ndarray of shape (B, N, 3), batch of lidar points.
    - T_lidar_to_camera: np.ndarray of shape (4, 4), shared transformation matrix.
    - K: np.ndarray of shape (3, 3), camera intrinsic matrix (shared across batch).
    - H: int, height of the image.
    - W: int, width of the image.

    Returns:
    - depth_map_batch: np.ndarray of shape (B, H, W, 3), depth maps for each batch element.
    - valid_indices_batch: np.ndarray of shape (B, N), boolean mask indicating valid points.
    """
    B, N, _ = lidar_pts_batch.shape
    # Add homogeneous coordinate to lidar points
    homogenous_lidar_pts_batch = np.concatenate(
        (lidar_pts_batch, np.ones((B, N, 1))), axis=-1
    )  # (B, N, 4)

    # Transform lidar points to camera frame
    projected_lidar_pts_batch = np.einsum(
        "ij,bnj->bni", T_lidar_to_camera, homogenous_lidar_pts_batch
    )  # (B, N, 4)
    
    # Filter points with positive depth (z > 0)
    valid_indices_batch = projected_lidar_pts_batch[..., 2] > 0
    projected_lidar_pts_valid = projected_lidar_pts_batch[valid_indices_batch].reshape(-1, 4)

    # Project points onto the image plane
    img_pts_batch = np.dot(K, projected_lidar_pts_valid[:, :3].T).T  # (M, 3)
    img_pts_batch[:, 0] /= img_pts_batch[:, 2]
    img_pts_batch[:, 1] /= img_pts_batch[:, 2]

    # Round and clip to valid pixel coordinates
    u = np.clip(np.round(img_pts_batch[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(img_pts_batch[:, 1]).astype(int), 0, H - 1)

    # Create depth maps
    depth_map_batch = np.zeros((B, H, W, 3), dtype=np.float32)
    batch_indices = np.repeat(np.arange(B), N)[valid_indices_batch.flatten()]
    depth_map_batch[batch_indices, v, u] = projected_lidar_pts_valid[:, :3]
    return depth_map_batch, valid_indices_batch
 
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

# Vectorized fused inputs
def fuseInputsBatch(lidar_seq_batch, img_seq_batch, flow_batch, T_lidar_to_camera, K):
    """
    Fuses batched lidar data, image sequence, and flow data into a single output.

    Parameters:
    - lidar_seq_batch: np.ndarray of shape (B, N, 3, 2), batch of lidar points for two time steps.
    - img_seq_batch: np.ndarray of shape (B, H, W, 3, 2), RGB image sequences for two time steps.
    - flow_batch: np.ndarray of shape (B, H, W, 2), flow data between the two time steps.
    - T_lidar_to_camera: np.ndarray of shape (4, 4), shared transformation matrix.
    - K: np.ndarray of shape (3, 3), camera intrinsic matrix (shared across batch).

    Returns:
    - output_batch: np.ndarray of shape (B, H, W, 14), concatenated data for each batch element.
    """
    B, H, W = img_seq_batch.shape[:3]

    # Unpack lidar point clouds and image sequence for each batch
    lidar_pts_T0_batch = lidar_seq_batch[..., 0]  # (B, N, 3)
    lidar_pts_T1_batch = lidar_seq_batch[..., 1]  # (B, N, 3)
    img_T0_batch = img_seq_batch[..., 0]  # (B, H, W, 3)
    img_T1_batch = img_seq_batch[..., 1]  # (B, H, W, 3)

    # Project lidar to camera frame for both time steps
    depth_map_T0_batch, _ = lidarToCamProjectionBatch(lidar_pts_T0_batch, T_lidar_to_camera, K, H, W)
    depth_map_T1_batch, _ = lidarToCamProjectionBatch(lidar_pts_T1_batch, T_lidar_to_camera, K, H, W)

    # Concatenate images, depth maps, and flow data (B, H, W, 14)
    output_batch = np.concatenate(
        (img_T0_batch, img_T1_batch, depth_map_T0_batch, depth_map_T1_batch, flow_batch), axis=-1
    )

    return output_batch