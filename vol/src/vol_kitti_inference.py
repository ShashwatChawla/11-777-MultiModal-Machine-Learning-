import sys
sys.path.append('/ocean/projects/cis220039p/pkachana/projects/11-777-MultiModal-Machine-Learning-/vol/src')

import torch

from configs.base_config import BaseConfig
from dataloader.dataloader import get_dataloader, process_data
from model.vol_net import VOLNet
from training.trainer import Trainer
from training.losses import compute_loss

from kitti.kitti_pytorch import OdometryDataset

def evaluate_on_kitti_sequences(sequences):
    config = BaseConfig()

    dataset = OdometryDataset(
        calib_path='/ocean/projects/cis220039p/shared/datasets/KITTI_odometry/calib/dataset/sequences',
        image_path='/ocean/projects/cis220039p/shared/datasets/KITTI_odometry/color/dataset/sequences',
        lidar_path='/ocean/projects/cis220039p/shared/datasets/KITTI_odometry/velodyne/dataset/sequences',
        is_training=False
    )

    # Filter dataset for requested sequences
    dataset.file_map = [dataset.file_map[i] for i in sequences]
    dataset.len_list = [dataset.len_list[i] for i in sequences]

    # Create DataLoader for evaluation
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize model and load checkpoint
    model = VOLNet()
    checkpoint_path = 'path/to/your/checkpoint.pth'  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)

    # Set to evaluation
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = compute_loss

    with torch.no_grad():
        for data in val_loader:
            img2, img1, pos2, pos1, q_gt, t_gt = data

            # TODO@Shashwat: Transform data as per VOL input
            
            # Perform inference
            output = model()            

            # Compute loss or metrics if needed
            loss = loss_fn(
                output['rotation'],
                q_gt,
                output['translation'],
                t_gt,
                alpha=config.loss_alpha
            )

            print(f"Rotation Loss: {loss['rotation_loss']:.4f}, Translation Loss: {loss['translation_loss']:.4f}")

if __name__ == '__main__':
    # Example KITTI sequences  
    sequences_to_evaluate = [6, 7, 8]  
    evaluate_on_kitti_sequences(sequences_to_evaluate)



    
