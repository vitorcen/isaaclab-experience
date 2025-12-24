#!/bin/bash

# NVIDIA Nucleus Checkpoint Downloader for Isaac Lab
# This script attempts to download pre-trained checkpoints for common tasks.

BASE_URL="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl"
DEST_DIR="IsaacLab/logs/rsl_rl"

# List of common tasks to try downloading
# Based on official IsaacLab source code and verified availability
TASKS=(
    # Direct Workflow Tasks (high performance)
    "Isaac-Cartpole-Direct-v0"
    "Isaac-Humanoid-Direct-v0"
    "Isaac-Ant-Direct-v0"
    "Isaac-Quadcopter-Direct-v0"
    "Isaac-Velocity-Flat-Anymal-C-Direct-v0"
    "Isaac-Velocity-Rough-Anymal-C-Direct-v0"
    "Isaac-Repose-Cube-Allegro-Direct-v0"

    # Manager-Based Tasks (modular design)
    "Isaac-Reach-Franka-v0"
    "Isaac-Cartpole-v0"
    "Isaac-Humanoid-v0"
    "Isaac-Ant-v0"
    "Isaac-Velocity-Flat-Anymal-C-v0"
    "Isaac-Velocity-Rough-Anymal-C-v0"
    "Isaac-Velocity-Flat-Anymal-D-v0"
    "Isaac-Velocity-Flat-Unitree-A1-v0"
    "Isaac-Velocity-Flat-Unitree-Go1-v0"
    "Isaac-Velocity-Flat-Unitree-Go2-v0"

    # Unitree Humanoid Robots
    "Isaac-Velocity-Flat-G1-v0"
    "Isaac-Velocity-Rough-G1-v0"
    "Isaac-Velocity-Flat-H1-v0"
)

echo "🤖 Isaac Lab Checkpoint Downloader"
echo "======================================"

for TASK in "${TASKS[@]}"; do
    echo "Checking for $TASK..."

    # Create destination directory
    # Note: Isaac Lab usually expects structure: logs/rsl_rl/{task_name}/...
    # But for 'play.py --use_pretrained_checkpoint', it downloads to a temp location or looks in specific places.
    # However, if we want to manually use a checkpoint, we can save it and use --checkpoint argument.

    TASK_DIR="$DEST_DIR/$TASK"
    mkdir -p "$TASK_DIR"

    # Attempt download
    CHECKPOINT_URL="$BASE_URL/$TASK/checkpoint.pt"
    DEST_FILE="$TASK_DIR/checkpoint.pt"

    if [ -f "$DEST_FILE" ]; then
        echo "  ℹ️  Checkpoint already exists. Skipping download."
    else
        echo "  Downloading from: $CHECKPOINT_URL"

        # Use curl to download, fail silently if not found (-f), show progress (-#), follow redirects (-L)
        if curl -f -L -# "$CHECKPOINT_URL" -o "$DEST_FILE"; then
            echo "  ✅ Success! Saved to $DEST_FILE"
        else
            echo "  ❌ Failed (Not found or unavailable)"
            rm -f "$DEST_FILE" # Cleanup empty file if created
        fi
    fi
    echo "--------------------------------------"
done

echo "Done."
echo ""
echo "Usage for downloaded checkpoints:"
echo "python scripts/reinforcement_learning/rsl_rl/play.py --task <TASK_NAME> --checkpoint logs/rsl_rl/<TASK_NAME>/checkpoint.pt"
