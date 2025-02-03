
Interpreter: Python 3.7

requirements: requirements.txt

DeepExplain reference: https://github.com/marcoancona/DeepExplain?tab=readme-ov-file

Sample data: data_preproc_ssvep: ch x time x trial (40 class SSVEP)
Example results: tmp_ssvep/Subject1.mat
Model: EEGnet_ssvep 
Epoch: 10
batch size: 256
Dropout  = 0.5


tmp_ssvep/Subject1.mat
  - 'deeplift' based attributions level
  - attributions_train (trial x ch x timesample): 
  - attributions_test (trial x ch x timesample)
