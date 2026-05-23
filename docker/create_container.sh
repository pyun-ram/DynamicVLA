export CODE=/data_shared_fast1/Docker/pyun/DynamicVLA/Code/DynamicVLA
export DATA=/data_shared_fast1/Docker/pyun/DynamicVLA/Data

docker rm -f dynamicvla-dev 2>/dev/null || true

docker run --gpus all -d \
  --name dynamicvla-dev \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  -v "${DATA}:/workspace" \
  -v "${CODE}:/workspace/DynamicVLA" \
  -w /workspace/DynamicVLA \
  --shm-size=16g \
  dynamicvla:latest \
  sleep infinity