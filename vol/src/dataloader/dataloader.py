import tartanair as ta
import numpy as np
import torch
import cv2

visualize = False

import open3d as o3d
import os
from os.path import join

_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

INTRINSICS = torch.tensor(
        [[320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]]

)


def vispcd(pc_np, vis_size=(1920, 480), o3d_cam=None):
    # pcd: numpy array
    w, h = (1920, 480)  # default o3d window size

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np)

    if o3d_cam:
        camerafile = o3d_cam
        w, h = camerafile['w'], camerafile['h']
        cam = o3d.camera.PinholeCameraParameters()

        intr_mat, ext_mat = camerafile['intrinsic'], camerafile['extrinsic']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h,
                                                      intr_mat[0, 0], intr_mat[1, 1],
                                                      intr_mat[0, -1], intr_mat[1, -1])
        intrinsic.intrinsic_matrix = intr_mat
        cam.intrinsic = intrinsic
        cam.extrinsic = ext_mat

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(pcd)

    if o3d_cam:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(cam)

    vis.poll_events()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    img = np.array(img)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, vis_size)

    return img

o3d_cam = join(_CURRENT_PATH, 'o3d_camera.npz')
lidarcam = np.load(o3d_cam)


# Initialize TartanAir.
tartanair_data_root = '/ocean/projects/cis220039p/shared/tartanair_v2'
ta.init(tartanair_data_root)

# Specify the environments, difficulties, and trajectory ids to load.
# envs = ['ArchVizTinyHouseDay']

envs = [
        # "ShoreCaves",
        # "AbandonedFactory",
        # "AbandonedSchool",
        # "AmericanDiner",
        # "AmusementPark",
        # "AncientTowns",
        # "Antiquity3D",
        # "Apocalyptic",
        # "ArchVizTinyHouseDay",
        # "ArchVizTinyHouseNight",
        # # "BrushifyMoon", # Seems to be very large and too easy for flow.
        # "CarWelding",
        # "CastleFortress",
        # "ConstructionSite",
        # "CountryHouse",
        # "CyberPunkDowntown",
        # "Cyberpunk",
        # "DesertGasStation",
        # "Downtown",
        # "EndofTheWorld",
        # "FactoryWeather",
        # "Fantasy",
        # "ForestEnv",
        # "Gascola",
        # "GothicIsland",
        # # "GreatMarsh",
        # "HQWesternSaloon",
        # "HongKong",
        # "Hospital",
        # "House",
        # "IndustrialHangar",
        # "JapaneseAlley",
        # "JapaneseCity",
        # "MiddleEast",
        # "ModUrbanCity",
        # "ModernCityDowntown",
        # "ModularNeighborhood",
        # "ModularNeighborhoodIntExt",
        # "NordicHarbor",
        # # "Ocean",
        # "Office",
        # "OldBrickHouseDay",
        # "OldBrickHouseNight",
        # "OldIndustrialCity",
        # "OldScandinavia",
        # "OldTownFall",
        # "OldTownNight",
        # "OldTownSummer",
        # "OldTownWinter",
        # # "PolarSciFi",
        # "Prison",
        # "Restaurant",
        # "RetroOffice",
        # "Rome",
        # "Ruins",
        # "SeasideTown",
        # # "SeasonalForestAutumn",
        # "SeasonalForestSpring",
        # # "SeasonalForestSummerNight",
        # "SeasonalForestWinter",
        # # "SeasonalForestWinterNight",
        # "Sewerage",
        # "Slaughter",
        # "SoulCity",
        # "Supermarket",
        # "TerrainBlending",
        # "UrbanConstruction",
        # "VictorianStreet",
        # "WaterMillDay",
        # "WaterMillNight",
        # "WesternDesertTown",
        # "AbandonedFactory2",
        "CoalMine"
        ]

difficulties = ['easy']
trajectory_ids = ['P005']

# Specify the modalities to load.
modalities = ['image', 'pose', 'lidar']
camnames = ['lcam_front']

# Specify the dataloader parameters.
new_image_shape_hw = None # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
subset_framenum = 10 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
seq_length = {'image': 2, 'pose': 2, 'lidar': 2} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
seq_stride = 1 # This is the stride between the data-sequences. For example, if the sequence length is 2 and the stride is 1, then the dataloader will load pairs of images [0,1], [1,2], [2,3], etc. If the stride is 2, then the dataloader will load pairs of images [0,1], [2,3], [4,5], etc.
frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
batch_size = 1 # This is the number of data-sequences in a mini-batch.
num_workers = 4 # This is the number of workers to use for loading the data.
shuffle = False # Whether to shuffle the data. Let's set this to False for now, so that we can see the data loading in a nice video. Yes it is nice don't argue with me please. Just look at it! So nice. :)

def get_dataloader():
    # Create a dataloader object.
    dataloader = ta.dataloader(env = envs,
                difficulty = difficulties,
                trajectory_id = trajectory_ids,
                modality = modalities,
                camera_name = camnames,
                # new_image_shape_hw = new_image_shape_hw,
                seq_length = seq_length,
                subset_framenum = subset_framenum,
                seq_stride = seq_stride,
                frame_skip = frame_skip,
                batch_size = batch_size,
                num_workers = num_workers,
                shuffle = shuffle,
                verbose = True)

    print("Dataloader created.")

    return dataloader

def main():
    # Create a dataloader object.
    dataloader = get_dataloader()

    # Iterate over the batches.
    for i in range(100):
        # Get the next batch.
        batch = dataloader.load_sample()
        # Visualize some images.
        # The shape of an image batch is (B, S, H, W, C), where B is the batch size, S is the sequence length, H is the height, W is the width, and C is the number of channels.

        print("Batch number: {}".format(i+1), "Loaded {} samples so far.".format((i+1) * batch_size))

        for b in range(batch_size):

            img0 = batch["rgb_lcam_front"][0][0]

            lidar0 = batch['lidar'][0][0]

            # breakpoint()
            # lidarvis = vispcd(lidar0, vis_size=(640, 640), o3d_cam=lidarcam)
            # breakpoint()

            # disp0 = cv2.vconcat((img0.numpy(), lidarvis))
            # if visualize:
            #     cv2.imshow('img', disp0)
            #     cv2.waitKey(1)

        # print("  Pose: ", pose[0][0])
        # print("  IMU: ", imu[0][0])

    dataloader.stop_cachers()


if __name__ == '__main__':
    main()