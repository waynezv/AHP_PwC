# AHP_PwC
Audio based human profiling for PwC project.

## Prerequisite
1. GPU capable computer
2. CUDA 9.0 + CuDNN 7.0
3. Packages: see [requirements](./requirements.txt)

## Usage
First go to the main directory

'''bash
cd ~/ProJEX/AHP_PwC
'''

### 1. Extract OpenSMILE features
1. '''bash
cd feat_extract
'''

2. '''bash
bash smile_feature_extractor.sh -c ${PATH_TO_OPENSMILE}/opensmile-2.3.0/config/IS13_ComParE.conf -f ./timit_test_wavlist.ctl -i ${PATH_TO_DATASET} -o ${OUTPUT_PATH} -t temp

3. The extracted features are already in *AHP_PwC/timit_opensmile_feat* and *AHP_PwC/interrogation_opensmile_feat*

----
### 2.1 Train model for age and height prediction
1. cd AHP_PwC/model

2. **age** '''bash
CUDA_VISIBLE_DEVICES=3 python predictor_nn.py --task age --nepoch 30 --cuda --manualSeed 1234 --outf age_prediction
'''

The trained model is saved in *AHP_PwC/model/age_prediction/checkpoints*.

**height** '''bash
CUDA_VISIBLE_DEVICES=3 python predictor_nn.py --task height --nepoch 30 --cuda --manualSeed 1234 --outf height_prediction
'''

The trained model is saved in *AHP_PwC/model/height_prediction/checkpoints*.

### 2.2. Evaluate model
1. cd AHP_PwC/model

2. **age** '''bash
CUDA_VISIBLE_DEVICES=3 python predictor_nn.py --task age --cuda --manualSeed 1234 --outf age_prediction --resume age_prediction/checkpoints/checkpoint_BEST.pth.tar --eval --testFn age_prediction/sa1.smile
'''

**height** '''bash
CUDA_VISIBLE_DEVICES=3 python predictor_nn.py --task height --cuda --manualSeed 1234 --outf height_prediction --resume height_prediction/checkpoints/checkpoint_BEST.pth.tar --eval --testFn height_prediction/sa1.smile
'''

----
### 3.1. Train model for lie detection
1. cd AHP_PwC/model

2. '''bash
CUDA_VISIBLE_DEVICES=3 python predictor_lie.py --task lie --cuda --manualSeed 1234 --nepoch 30 --outf lie_prediction
'''

The trained model is saved in *AHP_PwC/model/lie_prediction/checkpoints*.

### 3.2. Evaluate model
1. cd AHP_PwC/model

2. '''bash
CUDA_VISIBLE_DEVICES=3 python predictor_lie.py --task lie --cuda --manualSeed 1234 --outf lie_prediction --resume lie_prediction/checkpoints/checkpoint_BEST.pth.tar --eval --testFn lie_prediction/Amanda_Rogers_Bullhead_City_Police_Interview_1.smile
'''

'''bash
CUDA_VISIBLE_DEVICES=3 python predictor_lie.py --task lie --cuda --manualSeed 1234 --outf lie_prediction --resume lie_prediction/checkpoints/checkpoint_BEST.pth.tar --eval --testFn lie_prediction/sa1.smile
'''
