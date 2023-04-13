MODEL_NAME=bert-base-uncased
GPU_ID=0
SEEDS=-1  # set the default seed(s) to be the same as GPU_ID

while getopts ":g:s:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG"
    ;;
    s) SEEDS="$OPTARG"
    ;;
  esac
done


IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"  # split SEEDS into an array


# if the seed is not specified, use the GPU ID as the seed
if [ ${SEED_ARRAY[0]} -eq -1 ]; then
  SEED_ARRAY[0]=$GPU_ID
fi

for seed
