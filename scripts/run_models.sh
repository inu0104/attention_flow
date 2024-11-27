#!/bin/bash

# Define common parameters
EPOCHS=100
BATCH_SIZE=50
LEARNING_RATE=0.001
DATA="burgers"  # Replace with your desired dataset (e.g., sine, ellipse)
EXPERIMENT="synthetic"  # Experiment type
SAVE_DIR="./results"  # Directory to save results

# Create results directory if not exists
mkdir -p $SAVE_DIR
run_model() {
    MODEL=$1
    FLOW_MODEL=$2
    LOG_FILE="${SAVE_DIR}/${MODEL}_${FLOW_MODEL}_log.txt"
    echo "Running $MODEL with $FLOW_MODEL..."
    python -m nfe.train \
        --seed 2\
        --experiment $EXPERIMENT \
        --data $DATA \
        --model $MODEL \
        --flow-model $FLOW_MODEL \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --hidden-dim 64 \
        --flow-layers 3 \
        --time-net TimeLinear \
        --nf $NF \
        --nu $NU > $LOG_FILE 2>&1
    echo "Results saved to $LOG_FILE"
}

delete_files() {
    # Define the path to the data directory
    DATA_DIR="./nfe/experiments/data/synth"

    # Check if the data directory exists
    if [ -d "$DATA_DIR" ]; then
        # Define the files to be deleted
        FILES_TO_DELETE=(
            "${DATA_DIR}/${DATA}_extrap_space.npz"
            "${DATA_DIR}/${DATA}_extrap_time.npz"
            "${DATA_DIR}/${DATA}.npz"
        )

        # Loop through the files and delete them if they exist
        for FILE in "${FILES_TO_DELETE[@]}"; do
            if [ -f "$FILE" ]; then
                echo "Deleting $FILE..."
                rm "$FILE"
            else
                echo "File $FILE does not exist."
            fi
        done
    else
        echo "Data directory $DATA_DIR does not exist."
    fi
}

# Loop over NF values and another parameter
for NF in 2000; do
    for NU in 20 40 60 70 80 100 200; do
        echo "Running models with NF=$NF and NU=$NU..."
        delete_files
        
        # Run ResNet Flow
        run_model "flow" "resnet"
        
        # Run GRU Flow
        run_model "flow" "gru"

        # Run Coupling Flow
        run_model "flow" "coupling"
    
        # Run Fourier Flow
        run_model "flow" "fourier"
  
        # # Run Attention Flow
        # run_model "flow" "attention"
    done
done