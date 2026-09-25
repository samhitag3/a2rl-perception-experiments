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
    Figure out data contracts:  determine training / inference data input and model inference output. Make it very 100% comprehensive and explicit - what data / values, what units if any, what format. Use isaac trajectory data as a sample to know where to cover our bases. Make sure values are nullable because real data will not have all the data. I am testing both rgb --> pose perception pipelines and rgb --> mask + mask --> pose pipelines (two joint models). The mask --> pose pipelines will either take in binary masks (1 mask per frame) or 1 mask per gate for frame (for the synthetic and isaac data, this will likely be instance ID masks, and for real data it will just be 1 for the biggest gate, 2 for the second biggest, etc. but the model should be prepared to have the data transformed into the 0 1 2 3 mask labeling set up since no real input will have instance masks (where the biggest mask might be labeled with ID 5, the next biggest mask might be labeled with ID 6). Set up the data contract / schema so that it can work with any of the pipelines, i.e. I will have one universal data folder that will be used for all the models I will be training.

    After determining the data formats --> there needs to be corresponding data splitting and validation script. The splitting script should check for an existing valid data split (valid if all the sequences)

- i want pose data - keypoint data,

- 
 domain randmizaion
seed generated split but dont overwrite


- i want to track current gate / passed gate and X+ gates in the future id gate instance an also gate state - passed current futrue bc when doubel gate ngate 6 there are past gates
reconstruct gate map

--> seapartae gaussain splat based off inference data
covaraince


^^decide which of these will be universal functions that are not model specific

determine model performance metrics - iou, dtectionin in fargates (find awway to create a value that helps me pick better gates based off this value)


2. figure out how to determine


3. training caommdns + configs - make them parameterized so i can j put the inputs in

for rgb to 