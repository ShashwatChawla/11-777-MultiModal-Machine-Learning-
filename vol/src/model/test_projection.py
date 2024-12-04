import numpy as np
from flow_projection_model import lidarToCamProjectionBatch, fuseInputsBatch

def lidarToCamProjectionBatch():

    # Batch size, number of points, and image dimensions
    B, N, H, W = 20, 1000, 256, 256

    # Generate random LiDAR points in the shape (B, N, 3)
    lidar_pts_batch = np.random.uniform(-10, 10, size=(B, N, 3))
    # Ensure positive z-values for projection validity
    lidar_pts_batch[:, :, 2] = np.abs(lidar_pts_batch[:, :, 2])

    # Transformation matrix from LiDAR to camera (identity for simplicity)
    T_lidar_to_camera = np.eye(4)

    # Camera intrinsic matrix
    K = np.array([
        [500, 0, W / 2],  # fx, 0, cx
        [0, 500, H / 2],  # 0, fy, cy
        [0, 0, 1]         # 0, 0, 1
    ])

    # Get output
    depth_maps, valid_indices = lidarToCamProjectionBatch(lidar_pts_batch, T_lidar_to_camera, K, H, W)

    # Validate output shapes
    assert depth_maps.shape == (B, H, W, 3), f"Depth maps shape mismatch: {depth_maps.shape}"
    assert valid_indices.shape == (B, N), f"Valid indices shape mismatch: {valid_indices.shape}"

    # Print summary
    print("Test passed! Outputs:")
    print(f"Depth Maps Shape: {depth_maps.shape}")
    print(f"Valid Indices Shape: {valid_indices.shape}")
    # print(f"Sample Depth Map (Batch 0):\n{depth_maps[0]}")

def test_fuseInputsBatch():
    # Test parameters
    B = 2  # Batch size
    N = 10  # Number of lidar points per batch
    H, W = 64, 64  # Image dimensions

    # Generate synthetic data
    lidar_seq_batch = np.random.rand(B, N, 3, 2).astype(np.float32)  # Random lidar points
    img_seq_batch = np.random.rand(B, H, W, 3, 2).astype(np.float32)  # Random image sequence
    flow_batch = np.random.rand(B, H, W, 2).astype(np.float32)  # Random flow data
    T_lidar_to_camera = np.eye(4, dtype=np.float32)  # Identity transformation
    K = np.array([[500, 0, W // 2],
                  [0, 500, H // 2],
                  [0, 0, 1]], dtype=np.float32)  # Example camera intrinsic matrix

    # Call the function
    output_batch = fuseInputsBatch(
        lidar_seq_batch,
        img_seq_batch,
        flow_batch,
        T_lidar_to_camera,
        K
    )

    # Verify output shape
    assert output_batch.shape == (B, H, W, 14), f"Expected shape (B, H, W, 14), got {output_batch.shape}"

    # Verify data ranges
    assert np.all(output_batch >= 0), "Output contains negative values"
    assert np.all(output_batch <= 1), "Output contains values greater than 1"

    # Print success message
    print("fuseInputsBatch test passed! Output shape:", output_batch.shape)

# Run the test
if __name__ == "__main__":
    test_fuseInputsBatch()