# GETTING STARTED
## Usage instructions for the COIN dataset and MTL & GLCU approchs

## Training
The training code is available in the `ssn_train.py` script. `ssn_models.py` defines the SSN model. `ssn_opts.py` defines the arguments to the model. 

### Model Parameters
```
dataset=coin
modality=RGB
arch=BNInception
num_aug_segments=default
num_body_segments=default
task_head=TRUE if doing MTL / GLCU architecture
glcu=TRUE if want to use hadamard GLCU after backbone merging into backbone
additive_glcu=TRUE if want to use + GLCU after backbone merging into backbone (glcu has to be TRUE)
glcu_skip=TRUE if want to use + GLCU after backbone merging into the next layer
use_task_target=Train GLCU while taking ground truth as input to DSC phase
data_root=path to the rawframes folder containing raw frames folders for all videos (not nested in task folders)
```

### Runtime parameters
See scripts in `relevant_scripts` directory. Next could see the scripts in `all_scripts` directory for older scripts.


## Testing
For faster testing the test TAG files have been divided into 6 parts, to be spawned off on 6 jobs and possibly run in parallel. These are in `data/tcoin/` directory. Consequently later in evaluation these are merged and then evaluated.

Test files:
```
ssn_test.py SSN test file
ssn_test_bp.py SSN test file for GLCU arch & backpropagation
ssn_test_skip.py SSN test file for skip-GLCU & corres. backpropagation
ssn_test_feedback.py SSN test file for skip-GLCU & feedback mechanism
```

Again see the script in the `relevant_scripts` directory. 

## Evaluation
For evaluation we use the `code/tc-ssn/eval_detection_results.py` script. First we combine the 6 scripts from before here. We use various helper scripts in `tools` directory. Each contains a docstring in the beginning for the instructions.

Listing the most common and frequently used:

### combining
```
python tools/combine_pkl_files.py 1 6 [0 5]
```
later two arguments in case of feedback.

### task consistency
```
python tools/task_consistency.py 2 data/coin/W.npy <RESULT_PKL> <PATH_2_TC_RESULT_PKL> data/coin/coin_tag_test_proposal_list.txt
```

### tag pruning
```
python tag_pruning.py <IN_PKL> <OUT_PKL>
```

### final evaluation
```
python code/tc-ssn/eval_detection_results.py coin <RESULT_PKL>
```

### final evaluation for TC
```
python code/tc-ssn/eval_detection_results.py coin <TC_RESULT_PKL> --tc
```
