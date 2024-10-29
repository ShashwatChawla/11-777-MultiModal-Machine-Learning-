import tartanair as ta
import numpy as np
import cv2

# Initialize TartanAir.
tartanair_data_root = '/ocean/projects/cis220039p/shared/tartanair_v2'
ta.init(tartanair_data_root)

# Specify the environments, difficulties, and trajectory ids to load.
envs = ['ArchVizTinyHouseDay']
difficulties = ['easy']
trajectory_ids = ['P000', 'P001']

# Specify the modalities to load.
modalities = ['image', 'pose', 'lidar']
camnames = ['lcam_front', 'lcam_left', 'lcam_right', 'lcam_back', 'lcam_top', 'lcam_bottom']

# Specify the dataloader parameters.
new_image_shape_hw = [640, 640] # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
subset_framenum = 200 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
seq_length = {'image': 2, 'pose': 2, 'lidar': 2} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
seq_stride = 1 # This is the stride between the data-sequences. For example, if the sequence length is 2 and the stride is 1, then the dataloader will load pairs of images [0,1], [1,2], [2,3], etc. If the stride is 2, then the dataloader will load pairs of images [0,1], [2,3], [4,5], etc.
frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
batch_size = 8 # This is the number of data-sequences in a mini-batch.
num_workers = 4 # This is the number of workers to use for loading the data.
shuffle = True # Whether to shuffle the data. Let's set this to False for now, so that we can see the data loading in a nice video. Yes it is nice don't argue with me please. Just look at it! So nice. :)

# Create a dataloader object.
dataloader = ta.dataloader(env = envs,
            difficulty = difficulties,
            trajectory_id = trajectory_ids,
            modality = modalities,
            camera_name = camnames,
            new_image_shape_hw = new_image_shape_hw,
            seq_length = seq_length,
            subset_framenum = subset_framenum,
            seq_stride = seq_stride,
            frame_skip = frame_skip,
            batch_size = batch_size,
            num_workers = num_workers,
            shuffle = shuffle,
            verbose = True)

# Iterate over the batches.
for i in range(100):
    # Get the next batch.
    batch = dataloader.load_sample()
    # Visualize some images.
    # The shape of an image batch is (B, S, H, W, C), where B is the batch size, S is the sequence length, H is the height, W is the width, and C is the number of channels.

    print("Batch number: {}".format(i+1), "Loaded {} samples so far.".format((i+1) * batch_size))

    for b in range(batch_size):

        # Create image cross.
        left = batch['image_lcam_left'][b][0].numpy().transpose(1,2,0)
        front = batch['image_lcam_front'][b][0].numpy().transpose(1,2,0)
        right = batch['image_lcam_right'][b][0].numpy().transpose(1,2,0)
        back = batch['image_lcam_back'][b][0].numpy().transpose(1,2,0)
        top = batch['image_lcam_top'][b][0].numpy().transpose(1,2,0)
        bottom = batch['image_lcam_bottom'][b][0].numpy().transpose(1,2,0)
        cross_mid = np.concatenate([left, front, right, back], axis=1)
        cross_top = np.concatenate([np.zeros_like(top), top, np.zeros_like(top), np.zeros_like(top)], axis=1)
        cross_bottom = np.concatenate([np.zeros_like(bottom), bottom, np.zeros_like(bottom), np.zeros_like(bottom)], axis=1)
        cross = np.concatenate([cross_top, cross_mid, cross_bottom], axis=0)

        pose = batch['pose_lcam_front'].numpy()
        imu = batch['imu'].numpy()

        # Resize.
        cross = cv2.resize(cross, (cross.shape[1]//4, cross.shape[0]//4))

        # Show the image cross.
        cv2.imshow('cross', cross)
        cv2.waitKey(100)

    print("  Pose: ", pose[0][0])
    print("  IMU: ", imu[0][0])

dataloader.stop_cachers()