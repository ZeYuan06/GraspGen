#!/bin/bash

# Activate the Conda environment
source /home/zy3722/miniconda3/etc/profile.d/conda.sh
conda activate graspgen

# Define variables
MESH_FILE="${PWD}/models/sample_data/meshes/box.obj"
MESH_SCALE="1.0"
GRIPPER_CONFIG="${PWD}/models/checkpoints/graspgen_robotiq_2f_140.yml"
SCENE_MESH_FILE="${PWD}/scene_mesh/table.glb"
GRIPPER_MESH_FILE="robotiq_2f_140"
OUTPUT_FILE="${PWD}/outputs/output_grasps.yml"

# Run the Python script
python "${PWD}/scripts/demo_object_mesh.py" \
    --mesh_file "$MESH_FILE" \
    --mesh_scale "$MESH_SCALE" \
    --gripper_config "$GRIPPER_CONFIG" \
    --scene_mesh_file "$SCENE_MESH_FILE" \
    --gripper_mesh_file "$GRIPPER_MESH_FILE" \
    --filter_collisions \
    --output_file "$OUTPUT_FILE"