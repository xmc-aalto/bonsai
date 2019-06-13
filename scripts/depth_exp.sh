#! /bin/zsh

dataset=$1
fanout=$2
max_depth=$3
run_id=$4

TRAIN_CMD="../shallow/bonsai_train"
TEST_CMD="../shallow/bonsai_predict"


DATA_ROOT="/scratch/cs/xml/wsdm/code/parabel-v2/sandbox/data"
RESULT_ROOT="/scratch/cs/xml/bonsai-result"
MODEL_ROOT="/scratch/cs/xml/bonsai-model"

data_dir="${DATA_ROOT}/${dataset}"
trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"


results_dir="${RESULT_ROOT}/${dataset}/d${max_depth}-rid${run_id}"
model_dir="${MODEL_ROOT}/${dataset}/d${max_depth}-rid${run_id}"

score_file="${results_dir}/score_mat.txt"
performance_file="${results_dir}/performance.txt"

echo "results_dir: $results_dir"
echo "model_dir: $model_dir"
echo "score_file: $score_file"
echo "performance_file: $performance_file"

mkdir -p $results_dir
mkdir -p $model_dir

# training
# Reads training features (in $trn_ft_file), training labels (in $trn_lbl_file), and writes FastXML model to $model_dir
$TRAIN_CMD $trn_ft_file $trn_lbl_file $model_dir \
    -T 1 \
    -s 0 \
    -t 1 \
    -w ${fanout} \
    -b 1.0 \
    -c 1.0 \
    -m ${max_depth} \
    -f 0.1 \
    -fcent 0 \
    -k 0.0001 \
    -siter 20 \
    -q 1

# testing
# Reads test features (in $tst_ft_file), FastXML model (in $model_dir), and writes test label scores to $score_file
$TEST_CMD $tst_ft_file $score_file $model_dir -t 1 -B 10

# performance evaluation 
matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('../tools')); trn_X_Y = read_text_mat('$trn_lbl_file'); tst_X_Y = read_text_mat('$tst_lbl_file'); wts = inv_propensity(trn_X_Y,0.55,1.5); score_mat = read_text_mat('$score_file'); get_all_metrics(score_mat, tst_X_Y, wts); exit;" > $performance_file

