import sys
sys.path.append('/ocean/projects/cis220039p/schawla1/11-777-MultiModal-Machine-Learning-/vol/src')
import torch
import torch.nn.functional as F

import wandb
from tqdm import tqdm

from configs.base_config import BaseConfig, KittiConfig
from model.vol_net import VOLNet
from training.losses import compute_loss
from kitti.kitti_pytorch import OdometryDataset


def preprocess_image(batch_images, target_size=640):
    """
    Resizes batch of images from (b, h, w, c) to (b, 3, 640, 640).
    Maintains aspect ratio of the images
    """

    # If already (b, c, h, w)
    if batch_images.shape[1] != 3:  
        batch_images = batch_images.permute(0, 3, 1, 2)
    
    b, c, h, w = batch_images.shape
    processed_images = []

    
    for i in range(b):
        image = batch_images[i]  

        # Compute scaling factor
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize to maintain aspect ratio
        resized_image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

        # Create padded img w/t target size
        padded_image = torch.zeros((c, target_size, target_size), dtype=image.dtype)

        # Compute offsets for centering
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2

        padded_image[:, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        processed_images.append(padded_image)


    batch_tensor = torch.stack(processed_images)
    return batch_tensor

def preprocess_lidar(lidar_points: torch.Tensor, P, target_shape=(3, 370, 1126)):
    """
    Note: Lidar pts projected to cam using instrincs(P) and XYZ vals stored at those indices
    """

    # Zero-out negative z values (Velodyne pc has -ve vals)
    lidar_points[:, :, 2] = torch.where(lidar_points[:, :, 2] < 0, torch.tensor(0, dtype=lidar_points.dtype, device=lidar_points.device), lidar_points[:, :, 2])

    batch_size = lidar_points.shape[0]
    num_pts = lidar_points.shape[1]
    
    ones = torch.ones((batch_size, num_pts, 1), dtype=lidar_points.dtype, device=lidar_points.device)
    # Shape (b, num_pts, 4)
    lidar_points_hom = torch.cat([lidar_points, ones], dim=-1)  
    
    # Projection 
    lidar_points_proj = torch.matmul(lidar_points_hom, P.T) 
    # Homogeneous to Pixel Coordinates
    lidar_points_proj[:, :, 0] /= lidar_points_proj[:, :, 2]
    lidar_points_proj[:, :, 1] /= lidar_points_proj[:, :, 2]

    # Valid projection checks 
    valid_points = lidar_points_proj[:, :, 2] > 0  
    
    u = torch.clamp(lidar_points_proj[:, :, 0], min=0, max=target_shape[2] - 1)
    v = torch.clamp(lidar_points_proj[:, :, 1], min=0, max=target_shape[1] - 1)
    u = u * valid_points
    v = v * valid_points

    # Target Tensor (X, Y, Z)
    lidar_values = torch.zeros((batch_size, 3, target_shape[1], target_shape[2]), dtype=lidar_points.dtype, device=lidar_points.device)

    
    u_int = u.to(torch.long)
    v_int = v.to(torch.long)
    
    # Fill the tensor
    lidar_values[torch.arange(batch_size).unsqueeze(1), 0, v_int, u_int] = lidar_points[:, :, 0]  # X value
    lidar_values[torch.arange(batch_size).unsqueeze(1), 1, v_int, u_int] = lidar_points[:, :, 1]  # Y value
    lidar_values[torch.arange(batch_size).unsqueeze(1), 2, v_int, u_int] = lidar_points[:, :, 2]  # Z value

    return lidar_values


def preprocess_data(data, device, config):
    """
        Convert KITTI Data as per VOL input
    """

    img2, img1, pos2, pos1, q_gt, t_gt, P_cam = data

    # Process Images
    img1 = preprocess_image(img1, config.vol_input_shape[2]).unsqueeze(1).float()
    img2 = preprocess_image(img2, config.vol_input_shape[2]).unsqueeze(1).float()
    # Process Lidar Pts 
    pos1 = preprocess_lidar(pos1, P_cam[0, :, :])
    pos2 = preprocess_lidar(pos2, P_cam[0, :, :])
    # Reshape
    pos1 = preprocess_image(pos1, config.vol_input_shape[2]).unsqueeze(1).float()
    pos2 = preprocess_image(pos2, config.vol_input_shape[2]).unsqueeze(1).float()

    # Convert to VOL data-loader format
    data = {
            "images": torch.cat((img1, img2), dim=1), 
            "pointmaps": torch.cat((pos1, pos2), dim=1),
            "translations": t_gt[:, :, 0],
            "rotations": q_gt 
            }
    
    data = {key: value.to(device) for key, value in data.items()}
    return data
    

def load_sequences(kitti_config):
    """
        Load KITTI Sequences
    """
    print(kitti_config.calib_dir)
    dataset = OdometryDataset(
        calib_path=kitti_config.calib_dir,
        image_path=kitti_config.image_dir,
        lidar_path=kitti_config.lidar_dir,
        is_training=False
    )
    # Filter dataset for requested sequences
    dataset.file_map = [dataset.file_map[i] for i in kitti_config.sequences]
    dataset.len_list = [dataset.len_list[i] for i in kitti_config.sequences]

    # DataLoader for evaluation
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)


def main():
    config = BaseConfig()
    kitti_config = KittiConfig()
    logger = wandb.init(project=kitti_config.project_name)  

    # Load KITTI Sequences in DataLoader
    kitti_loader = load_sequences(kitti_config)

    # Initialize Model w/t Checkpt
    model = VOLNet(config)
    checkpoint = torch.load(kitti_config.vol_checkpoint)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)

    # # Evaluation
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("[INFO] KITTI Evaluation Started")
    with torch.no_grad():
        for data in tqdm(kitti_loader, desc="Processing", unit="batch"):
            data = preprocess_data(data, device, kitti_config)

            output = model(data)

            loss = compute_loss(
                                output['rotation'], 
                                data['rotations'], 
                                output['translation'], 
                                data['translations'],
                                alpha=config.loss_alpha)

            # Log Images
            if kitti_config.log_imgs:
                save_img1 = data["images"][0][0].cpu().numpy().transpose(1, 2, 0)
                save_img2 = data["images"][0][1].cpu().numpy().transpose(1, 2, 0)
                logger.log({
                            "Processed Input Images": [
                                wandb.Image(save_img1, caption="Image 1"),
                                wandb.Image(save_img2, caption="Image 2")], 
                            })
            
            # Log losses
            logger.log({
                        "total_loss": loss['total_loss'],
                        "translation_loss": loss['translation_loss'],
                        "rotation_loss": loss['rotation_loss']
                        })
            

            # print(f"Rotation Loss: {loss['rotation_loss']:.4f}, Translation Loss: {loss['translation_loss']:.4f}")

    print("[INFO] KITTI Evaluation Complete")

if __name__ == '__main__':
    main()
    



    
