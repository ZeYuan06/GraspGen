#!/bin/bash

# Activate the Conda environment
source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
conda activate graspgen

# Define variables
OBJECT_NAME="clamp"
MESH_FILE="${PWD}/models/sample_data/meshes/${OBJECT_NAME}.obj"
MESH_SCALE="0.001"
GRIPPER_CONFIG="${PWD}/models/checkpoints/graspgen_robotiq_2f_140.yml"
SCENE_MESH_FILE="${PWD}/scene_mesh/table.glb"
GRIPPER_MESH_FILE="robotiq_2f_140"
OUTPUT_FILE="${PWD}/outputs/${OBJECT_NAME}_grasps.yml"
NUM_SAMPLE_POINTS="2000"
NUM_GRASPS="500"
TOPK_NUM_GRASPS="200"

# Run the Python script
python "${PWD}/scripts/demo_object_mesh.py" \
    --mesh_file "$MESH_FILE" \
    --mesh_scale "$MESH_SCALE" \
    --gripper_config "$GRIPPER_CONFIG" \
    --scene_mesh_file "$SCENE_MESH_FILE" \
    --gripper_mesh_file "$GRIPPER_MESH_FILE" \
    --filter_collisions \
    --output_file "$OUTPUT_FILE" \
    --num_sample_points "$NUM_SAMPLE_POINTS" \
    --num_grasps "$NUM_GRASPS" \
    --return_topk \
    --topk_num_grasps "$TOPK_NUM_GRASPS"