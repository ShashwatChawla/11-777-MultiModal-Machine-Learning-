class BaseConfig:
    def __init__(self):
        
        ##############################
        ######## DATA CONFIGS ########
        ##############################

        self.tartanair_data_root = '/ocean/projects/cis220039p/shared/tartanair_v2'

        # Training data configs
        self.train_envs = [
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
            "Office",
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
            # "CoalMine"
        ]

        self.train_difficulties = ['easy']
        self.train_trajectory_ids = ['P005']

        # Val data configs
        self.val_envs = [
            "CoalMine",
        ]

        self.val_difficulties = ['easy']
        self.val_trajectory_ids = ['P005']

        # Specify the modalities to load.
        self.modalities = ['image', 'pose', 'depth', 'flow']
        self.camnames = ['lcam_front']

        # Specify the dataloader parameters.
        self.new_image_shape_hw = None # If None, no resizing is performed. If a value is passed, then the image is resized to this shape.
        self.subset_framenum = 10 # This is the number of frames in a subset. Notice that this is an upper bound on the batch size. Ideally, make this number large to utilize your RAM efficiently. Information about the allocated memory will be provided in the console.
        self.seq_length = {'image': 2, 'pose': 2, 'depth': 2, 'flow': 2} # This is the length of the data-sequences. For example, if the sequence length is 2, then the dataloader will load pairs of images.
        self.seq_stride = 1 # This is the stride between the data-sequences. For example, if the sequence length is 2 and the stride is 1, then the dataloader will load pairs of images [0,1], [1,2], [2,3], etc. If the stride is 2, then the dataloader will load pairs of images [0,1], [2,3], [4,5], etc.
        self.frame_skip = 0 # This is the number of frames to skip between each frame. For example, if the frame skip is 2 and the sequence length is 3, then the dataloader will load frames [0, 3, 6], [1, 4, 7], [2, 5, 8], etc.
        self.batch_size = 1 # This is the number of data-sequences in a mini-batch.
        self.num_workers = 4 # This is the number of workers to use for loading the data.
        self.shuffle = False # Whether to shuffle the data. Let's set this to False for now, so that we can see the data loading in a nice video. Yes it is nice don't argue with me please. Just look at it! So nice. :)


        ##############################
        ####### MODEL CONFIGS ########
        ##############################
        self.load_pretrained_flow = True
        self.flow_checkpoint = "/ocean/projects/cis220039p/pkachana/projects/tartanvo-fisheye-old/src/tartanvo_fisheye/networks/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth" # GMFlow checkpoint


        ##############################
        ######## TRAIN CONFIGS #######
        ##############################
        self.num_steps = 10_000
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.val_freq = 1000

        self.loss_alpha = 1.0
        

        ##############################
        ####### LOGGING CONFIGS ######
        ##############################
        self.ckpt_save_dir = "/ocean/projects/cis220039p/pkachana/projects/11-777-MultiModal-Machine-Learning-/vol/src/checkpoints"
        self.project_name = 'VOL'
        self.log_freq = 10
        self.log_steps = 10