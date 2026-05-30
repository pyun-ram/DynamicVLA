PROJECT_ROOT=$(pwd)
NUM_EPISODES=${1:-10}
DATASET_NAME=${2:-20260530_dynamicvla_data}
cd ${PROJECT_ROOT}
mkdir ${DATASET_NAME}
conda run -n isaaclab python simulations/simulate.py \
    --headless --enable_cameras --save \
    --task pick --robot franka \
    -n ${NUM_EPISODES} --seed 42 \
    --scene_dir ${PROJECT_ROOT}/scenes \
    --object_dir ${PROJECT_ROOT}/objects \
    -o ${PROJECT_ROOT}/${DATASET_NAME}/