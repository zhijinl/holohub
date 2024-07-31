
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run -it --rm \
       --gpus all \
       --ipc=host \
       -p 8042:8042 \
       -p 4242:4242 \
       -p 8000:8000 \
       -v ${SCRIPT_DIR}/:/workspace \
       -w /workspace \
       holohub-cbct-vis
