# a2rl-perception-experiments
REPOSITORY STRUCTURE:
├─-assets/
├─-data/
├─-models/
├─-utils
├─-.gitignore
├─-APPROACH.md
├─-TRAIN.md
└─-utils/

1. PLANNING
    Figure out data contracts:  determine training / inference data input and model inference output. Make it very 100% comprehensive and explicit - what data / values, what units if any, what format. Use isaac trajectory data as a sample to know where to cover our bases (beyond just DEFINITELY including rgb images, disjoint segmentation mask images meaning 1 mask per gate per frame, keypoint data, camera intrinsics, pose, gate ID, and a value that indicates if a gate visible frame is already passed, is the current target gate / being passed through, or a future gate (for vertical double gates, determine a convention to label the top and bottom gates, since they will share the same gate number but maybe label with A or B or .1 and .2)). The contract should have sufficient data so that I can later create a script that takes the model inference output and can use that to make a 3d reconstruction of the gate map that was in the input trajectory. Make sure values are nullable because real data will not have all the data. I am testing both rgb --> pose perception pipelines and rgb --> mask + mask --> pose pipelines (two joint models). The mask --> pose pipelines will either take in binary masks (1 mask per frame) or 1 mask per gate for frame (for the synthetic and isaac data, this will likely be instance ID masks, and for real data it will just be 1 for the biggest gate, 2 for the second biggest, etc. but the model should be prepared to have the data transformed into the 0 1 2 3 mask labeling set up since no real input will have instance masks, where the biggest mask might be labeled with ID 5, the next biggest mask might be labeled with ID 6). Set up the data contract / schema so that it can work with any of the pipelines, i.e. I will have one universal data folder that will be used for all the models I will be training.

    After determining the data formats --> there needs to be corresponding data splitting and validation script. The splitting script (inputs: seed, can be defaulted) should check for an existing valid data split (valid if all the sequences are assigned to one of validation, test, or training and none of them are repeated). Then it should check if the existing split seed if there is one matches the inputted one and if it does, it should print something like: split already exists / confirmed. If existing split doesnt match the potential new one, give  y/n prompt to ask if the user wants to replace the split. If there is no existing split, create the split. To easily configure the splits, I suggest making them by assigning the data sequences in a txt or json file rather than actually moving around the data files and keeping the original files untouched. Apart from checking the splits, the validation script should check the actual data if it's compatible with the data contract. By default the split should be 80-10-10 where 80 is for training. If there are minimal sequences to available, prioritize assigning sequences to training, then validation, then test.

    I will also need to training split script. (read below, but I will need splits of 10% and 30% of the data). When running this--first check for existing split of that ratio. If it already exists, do not rerun it. This split format should allow for multiple splits because I will need a 10% and a 30% split for the different training stages. If there are minimal sequences available, still make sure at least 1 sequence from the data is assigned to the 10% split.

    THESE SPLITS SHOULD SPLIT UP SEQUENCES AS A WHOLE -- THEY WILL NOT SPLIT UP INDIVIDUAL SEQUENCES.

    I need this data contract to be a file with all the information above specified -- I should be able to feed that in wherever (into LLMs) and it should be easy to understand by humans. Additionally at this point I also want to determine the metrics that will used to evaluate model performace. There are a few I certainly want--mask iou, and also prioritize gates further away, because if we just consider iou, if only the big masks in the front are detected, the iou will be high but won't account for the incorrect gate counts--and any more metrics that seem relevant.

2. DATA FORMATTING / GENERATION
    I will have 3 data formats. Real, Synthetic, and Sim.

    For Real images, i don't have them labeled - I want to create a labeling script where it opens a labeling UI. The script will take in 2 inputs: rgb directory / image path, and a processed data output directory, where the output will match the data contract specified in #1. The input can be a single image OR a directory of images (but either way the output will be a folder since the output will have rgb, masks, pose data, etc.). The labeling UI should have an opening screen where I input the known camera intrinsics (where there is a null option if I don't know them), and then it should iterate through the different images. This should be the workflow for each image:
    - START LABELING button (grey it out after it is clicked), and a RESTART LABELING (clears the current labels and ungreys out the START LABELING button, exactly one of START and RESTART should be greyed out at all times, or maybe make it a switch)
    - START GATE and a RESTART GATE (similar to the START LABELING and RESTART LABELING it should be a one or another situation / switch), where I will start labeling the keypoints for a specific gate. I should have the option to skip corners if they are not visible in the image. I should be able to 2 finger scroll zoom in and out of the image, but there should also be ZOOM IN and ZOOM OUT buttons and a slider to navigate the image. Click to select a keypoint. After selecting the 8 keypoints in order (4 outer TL TR BR BL and then 4 inner), show the mask created buy the keypoints and an adjustment menu. The selected keypoints should be fixed, but for any keypoints not selected there should be a movable corner i can used to fill out the mask. There also should be point I can adjust on the sides of the gate to line up the mask with gate. There should also be a brush add / erase to deal with any obstructions in the image. After selecting the 8 keypoints the COMPLETE MASK button should be ungreyed out (because I might not always need to adjust the mask). There should also be a NO GATES IN IMAGE button, to log 0 gates in the image.
    - Then there should be COMPLETE LABELING button. This should only appear after at least 1 gate has been logged or NO GATES IN IMAGE was selected. After this button is selected, the UI should proceed to the next image if there is one.
    The script should iterate the images in the folder that are UNLABELED. i.e. if the script finds existing labeling for an image, it should start from the first unlabeled image.
    Since I won't necessarily get through labeling all the images, rgb should be copied over to the outout directory, the output directory should be a separate directory that follows the data contract.

    For sim images, I will create a script that transforms the existing sim data into the format that follows the data contract-->first it will validate, and if it's validated, it leaves it as is and has a y/n prompt that asks if the user wants to rename the existing directory to match the output directory if one is provided. If it's not validated, it copies the data into the format that's data contract compatible. If there is no output directory provided just attach the prefix FORMATTED_ to the existing directory name since we want the output to be separate.

    For synthetic images, I will create a data generation script that takes in 3 inputs (these should NOT be hardcoded into the configs): a backgrounds folder, a gate skin image path, and an output directory. The synthetic image data generation script will be designed to create a diverse set of trajectories (they should mimic the files of a flight video, because some of my models will use temporal gate tracking), and should include all the information in the data contract. The diversity must come from random:
    - various gate paths (positioning of gates + flight failures)
    - varying speeds
    - different lighting affects - flares, random light patches, overall darking of image
    - noise - lower contrast, add random noise, add some blur
    - including double gates (one gate stacked upon the other)
    - using all the images provided in the backgrounds folder
    The config files should also take in and consider the camera intrinsics and the gate dimensions. The data generation script will lie in the utils/ directory.

3. DIRECTORY SET UP
    I will configure all the paths in the terminal I am working out of.

4. TRAINING
    The CLI commands should all be able to be run from the root directory. For each training stage I will need to train, evaluate, benchmark the latency, and run inference on three different sequences (1 real, 1 synthetic, 1 sim). All training will use real + sim + synthetic data.

    So each model will need a regular inference script and a pose covariance inference script. The inference videos should have a top and bottom panel. Top panel has translucent mask overlay with labeled and connected keypoints (to form the inner + outer gate shape), and the bottom panel uses black + white perfect seg mask representation with keypoints and inferred mask colorful overlay. For inference with rgb-->mask + mask-->pose pipelines, I want the inference video to be 4 panel where the left side uses inferred masks from the first model in the pipeline and the right side uses the perfect masks. I'm not sure if the evaluation and latency benchmark scripts have to be model specific, but if they are universal they will go in the utils/ folder.

    For each model, I will run:
    - A cheap training - 5 epochs with 10% of the data to smoke test
    - A baseline training - 15 epochs with 30% of the data to compare the efficacy of the model to compare all the models

    After picking certain models to follow through with based off the baseline performance, I will run:
    - Hyperparameter tuning stage (this might have multiple stages) - 10 epochs each with 30% of the data, number of trials and stages will likely vary for model to model. This will only be done with selected models. Should start out with X number of trials based off number of hyperparams and the the option to add more trials if it seems like there is room for improvement. Might need multiple stages for things like temporal window length tuning if needed.
    - Final training - 60 epochs with early training 
    - Post training threshold variable sweeps (model dependent, not necessary for all models)

utils/ --> will have scripts like the data validation + splitting scripts, an images to video script in addition to all the synthetic data generation code, and a script to make a 3d reconstruction of the gate map using the inference output data.