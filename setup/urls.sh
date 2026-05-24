# Asset URLs for DynamicVLA setup (sourced by download_assets.sh).
# Override any variable before running the script, e.g.:
#   export ISAACSIM_ZIP_URL="https://..."

# Required for Docker build (Isaac Sim 4.5.0 standalone, Linux x86_64)
ISAACSIM_ZIP_NAME="isaac-sim-standalone-4.5.0-linux-x86_64.zip"
ISAACSIM_ZIP_URL="${ISAACSIM_ZIP_URL:-https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip}"

# DOM assets — test / objects via Google Drive (gdown)
#   Test:   https://drive.google.com/file/d/1P-SwMipDjJOLgbCzXx2gZ-bGzpPOrWTw/view
#   Objects: https://drive.google.com/file/d/1leMP8T2WylpyiDGLuiIN3uInZ0H_jn7-/view
DOM_TEST_GDRIVE_ID="${DOM_TEST_GDRIVE_ID:-1P-SwMipDjJOLgbCzXx2gZ-bGzpPOrWTw}"
DOM_OBJECTS_GDRIVE_ID="${DOM_OBJECTS_GDRIVE_ID:-1leMP8T2WylpyiDGLuiIN3uInZ0H_jn7-}"
DOM_TEST_ARCHIVE_NAME="${DOM_TEST_ARCHIVE_NAME:-DOM-Test.zip}"
DOM_OBJECTS_ARCHIVE_NAME="${DOM_OBJECTS_ARCHIVE_NAME:-DOM-3D-Objects.zip}"

# Scenes still from infinitescript gateway (override DOM_SCENES_URL if needed)
GATEWAY_BASE="${GATEWAY_BASE:-https://gateway.infinitescript.com/?f=}"
DOM_SCENES_URL="${DOM_SCENES_URL:-${GATEWAY_BASE}DOM-3D-Scenes}"

# Hugging Face (training dataset)
HF_DOM_DATASET="${HF_DOM_DATASET:-hzxie/DOM}"
HF_DOM_MODEL="${HF_DOM_MODEL:-hzxie/dynamic-vla-DOM}"
