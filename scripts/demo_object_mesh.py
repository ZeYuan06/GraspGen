# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from pathlib import Path

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    get_normals_from_mesh,
    make_frame,
    visualize_grasp,
    visualize_mesh,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
from grasp_gen.dataset.dataset_utils import sample_points
from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format, save_to_maniskill_format, check_collision


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a single object mesh after GraspGen inference"
    )
    parser.add_argument(
        "--mesh_file",
        type=str,
        required=True,
        help="Path to the mesh file (obj, stl, or ply)",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="",
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=-1,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=400,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return when return_topk is True",
    )
    parser.add_argument(
        "--mesh_scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the mesh",
    )
    parser.add_argument(
        "--num_sample_points",
        type=int,
        default=2000,
        help="Number of points to sample from the mesh surface",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to save the output grasps. If empty, will save to outputs/YYYY-MM-DD/latest/output_grasps.yml",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable meshcat visualization",
    )
    parser.add_argument(
        "--scene_mesh_file",
        type=str,
        default="",
        help="Path to the scene mesh file for collision checking (obj, stl, ply, glb, or gltf)",
    )
    parser.add_argument(
        "--gripper_mesh_file", 
        type=str,
        default="",
        help="Path to the gripper mesh file for collision checking (obj, stl, or ply)",
    )
    parser.add_argument(
        "--filter_collisions",
        action="store_true",
        help="Filter out grasps that collide with the scene",
    )

    return parser.parse_args()


def place_object_on_scene(obj_mesh, scene_mesh, placement_height_offset=0.01):
    """Place object on top of the scene (e.g., on table surface)."""
    if obj_mesh is None or scene_mesh is None:
        return None
    
    # Get scene bounds
    scene_bounds = scene_mesh.bounds
    scene_min_z = scene_bounds[0][2]  # Bottom Z
    scene_max_z = scene_bounds[1][2]  # Top Z (table surface)
    
    # Get object bounds
    obj_bounds = obj_mesh.bounds
    obj_min_z = obj_bounds[0][2]
    obj_height = obj_bounds[1][2] - obj_bounds[0][2]
    
    # Calculate placement position
    # Place object on table surface with small offset
    target_z = scene_max_z + placement_height_offset
    
    # Move object so its bottom sits on the table
    z_offset = target_z - obj_min_z
    placement_transform = tra.translation_matrix([0, 0, z_offset])
    
    print(f"Scene Z range: {scene_min_z:.3f} to {scene_max_z:.3f}")
    print(f"Object height: {obj_height:.3f}, placing at Z: {target_z:.3f}")
    
    return placement_transform


def load_mesh_data(mesh_file, scale, num_sample_points):
    """Load mesh data and sample points from surface."""
    if mesh_file.endswith("ply"):
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(mesh_file)
        xyz = np.array(pcd.points).astype(np.float32)
        pt_idx = sample_points(xyz, num_sample_points)
        xyz = xyz[pt_idx]
        obj = None
    else:
        obj = trimesh.load(mesh_file)
        obj.apply_scale(scale)
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.array(xyz)

    # Center point cloud
    T_subtract_pc_mean = tra.translation_matrix(-xyz.mean(axis=0))
    xyz = tra.transform_points(xyz, T_subtract_pc_mean)
    if obj is not None:
        obj.apply_transform(T_subtract_pc_mean)

    # Create dummy RGB values (white)
    rgb = np.ones((len(xyz), 3)) * 255

    return xyz, rgb, obj, T_subtract_pc_mean


def load_mesh_data_with_scene_placement(mesh_file, scene_mesh, scale, num_sample_points, placement_offset=0.01):
    """Load mesh data and place it appropriately relative to scene."""
    if mesh_file.endswith("ply"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(mesh_file)
        xyz = np.array(pcd.points).astype(np.float32)
        pt_idx = sample_points(xyz, num_sample_points)
        xyz = xyz[pt_idx]
        obj = None
        T_subtract_pc_mean = np.eye(4)  # No centering for point cloud
    else:
        obj = trimesh.load(mesh_file)
        obj.apply_scale(scale)
        
        # Place object on scene if scene is provided
        if scene_mesh is not None:
            placement_transform = place_object_on_scene(obj, scene_mesh, placement_offset)
            if placement_transform is not None:
                obj.apply_transform(placement_transform)
                print(f"Placed object on scene surface")
        
        # Center object in XY but keep Z placement
        obj_center = obj.bounds.mean(axis=0)
        T_subtract_xy_mean = tra.translation_matrix([-obj_center[0], -obj_center[1], 0])
        obj.apply_transform(T_subtract_xy_mean)
        
        # Sample points from placed object
        xyz, _ = trimesh.sample.sample_surface(obj, num_sample_points)
        xyz = np.array(xyz)
        
        T_subtract_pc_mean = T_subtract_xy_mean

    # Create dummy RGB values (white)
    rgb = np.ones((len(xyz), 3)) * 255

    return xyz, rgb, obj, T_subtract_pc_mean


def load_scene_mesh(scene_mesh_file, scale=1.0):
    """Load scene mesh for collision checking, supports GLB files."""
    if not os.path.exists(scene_mesh_file):
        raise FileNotFoundError(f"Scene mesh file {scene_mesh_file} not found")
    
    try:
        # Load the mesh/scene file
        loaded_object = trimesh.load(scene_mesh_file)
        
        # Handle different types of loaded objects
        if isinstance(loaded_object, trimesh.Scene):
            print(f"Loaded GLB scene with {len(loaded_object.geometry)} geometries")
            
            # # Combine all geometries in the scene into a single mesh
            # meshes = []
            # for name, geometry in loaded_object.geometry.items():
            #     if isinstance(geometry, trimesh.Trimesh):
            #         # Apply the transform from the scene graph
            #         transform = loaded_object.graph.get(name)[0]
            #         geometry_copy = geometry.copy()
            #         geometry_copy.apply_transform(transform)
            #         meshes.append(geometry_copy)
            #         print(f"  - Added mesh '{name}' with {len(geometry.vertices)} vertices")
            
            # if meshes:
            #     # Combine all meshes into a single mesh
            #     scene_mesh = trimesh.util.concatenate(meshes)
            #     print(f"Combined scene mesh has {len(scene_mesh.vertices)} vertices")
            # else:
            #     raise ValueError("No valid meshes found in the scene")

            scene_mesh = loaded_object.dump(concatenate=True)
            print(f"Successfully dumped scene mesh with {len(scene_mesh.vertices)} vertices")
                
        elif isinstance(loaded_object, trimesh.Trimesh):
            # Single mesh file
            scene_mesh = loaded_object
            print(f"Loaded single mesh with {len(scene_mesh.vertices)} vertices")
        else:
            raise ValueError(f"Unsupported mesh type: {type(loaded_object)}")
        
        # Apply scaling
        scene_mesh.apply_scale(scale)
        
        return scene_mesh
        
    except Exception as e:
        print(f"Error loading scene mesh: {e}")
        raise


def load_gripper_mesh(gripper_mesh_file, gripper_name):
    """Load gripper mesh for collision checking."""
    if gripper_mesh_file and os.path.exists(gripper_mesh_file):
        gripper_mesh = trimesh.load(gripper_mesh_file)
        print(f"Loaded custom gripper mesh: {gripper_mesh_file}")
    else:
        # Use default gripper mesh paths
        default_gripper_paths = {
            "robotiq_2f_140": "assets/robotiq/robotiq_140_collision.obj",
            "franka_panda": "assets/franka/franka_panda.urdf",
            "suction": "assets/suction/suction_cup.obj"
        }
        
        if gripper_name in default_gripper_paths:
            gripper_path = default_gripper_paths[gripper_name]
            if os.path.exists(gripper_path) and gripper_path.endswith(('.obj', '.stl', '.ply')):
                gripper_mesh = trimesh.load(gripper_path)
                print(f"Loaded default gripper mesh for {gripper_name}")
            else:
                print(f"Warning: Default gripper mesh not found for {gripper_name}")
                gripper_mesh = None
        else:
            print(f"Warning: No gripper mesh available for {gripper_name}")
            gripper_mesh = None
    
    return gripper_mesh


def filter_collision_grasps(grasps, grasp_conf, scene_mesh, gripper_mesh):
    """Filter out grasps that collide with the scene."""
    print("Checking for collisions...")
    
    # Check collisions using the existing check_collision function
    collisions = check_collision(scene_mesh, gripper_mesh, grasps)
    
    # Filter out colliding grasps
    valid_mask = ~collisions  # Invert to keep non-colliding grasps
    valid_grasps = grasps[valid_mask]
    valid_conf = grasp_conf[valid_mask]
    
    print(f"Filtered {np.sum(collisions)} colliding grasps out of {len(grasps)}")
    print(f"Remaining valid grasps: {len(valid_grasps)}")
    
    return valid_grasps, valid_conf, valid_mask


if __name__ == "__main__":
    args = parse_args()

    if args.gripper_config == "":
        raise ValueError("Gripper config is required")

    if not os.path.exists(args.gripper_config):
        raise ValueError(f"Gripper config {args.gripper_config} does not exist")

    # Check if mesh file has valid extension
    if not args.mesh_file.endswith((".stl", ".obj")):
        raise ValueError("Mesh file must be a .stl or .obj file")

    # Handle return_topk logic
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # Create visualizer unless visualization is disabled
    vis = None if args.no_visualization else create_visualizer()

    # Load grasp configuration and initialize sampler
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # Load scene mesh first if collision filtering is enabled
    scene_mesh = None
    gripper_mesh = None
    if args.filter_collisions:
        if args.scene_mesh_file == "":
            raise ValueError("scene_mesh_file is required when filter_collisions is True")
        
        scene_mesh = load_scene_mesh(args.scene_mesh_file, args.mesh_scale)
        gripper_mesh = load_gripper_mesh(args.gripper_mesh_file, gripper_name)
        
        if gripper_mesh is None:
            print("Warning: No gripper mesh available, collision filtering disabled")
            args.filter_collisions = False

    # Load mesh data with scene-aware placement
    print(f"Processing mesh file: {args.mesh_file}")
    if args.filter_collisions and scene_mesh is not None:
        # Use scene-aware placement
        pc, pc_color, obj_mesh, T_subtract_pc_mean = load_mesh_data_with_scene_placement(
            args.mesh_file, scene_mesh, args.mesh_scale, args.num_sample_points
        )
    else:
        # Use original centering method
        pc, pc_color, obj_mesh, T_subtract_pc_mean = load_mesh_data(
            args.mesh_file, args.mesh_scale, args.num_sample_points
        )

    # Visualize meshes
    if not args.no_visualization:
        if obj_mesh is not None:
            visualize_mesh(vis, "object_mesh", obj_mesh, color=[169, 169, 169])
        if scene_mesh is not None:
            visualize_mesh(vis, "scene_mesh", scene_mesh, color=[200, 200, 200], alpha=0.3)
        visualize_pointcloud(vis, "pc", pc, pc_color, size=0.0025)

    # Run inference on point cloud
    grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
        pc,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
        remove_outliers=False,
    )

    if len(grasps_inferred) > 0:
        grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
        grasps_inferred = grasps_inferred.cpu().numpy()
        
        print(
            f"Inferred {len(grasps_inferred)} grasps, with scores ranging from {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}"
        )

        # Convert grasps back to original mesh frame BEFORE collision checking
        grasps_original_frame = np.array(
            [tra.inverse_matrix(T_subtract_pc_mean) @ g for g in grasps_inferred]
        )
        
        # Apply collision filtering if enabled  
        if args.filter_collisions and scene_mesh is not None and gripper_mesh is not None:
            # No need to transform scene_mesh since object is already placed relative to scene
            valid_grasps, valid_conf, valid_mask = filter_collision_grasps(
                grasps_original_frame, 
                grasp_conf_inferred,
                scene_mesh,  # Use original scene_mesh
                gripper_mesh
            )
            
            # Update data for visualization and saving
            grasps_inferred = valid_grasps
            grasp_conf_inferred = valid_conf
        else:
            grasps_inferred = grasps_original_frame

        scores_inferred = get_color_from_score(grasp_conf_inferred, use_255_scale=True)
        print(
            f"Final {len(grasps_inferred)} grasps, with scores ranging from {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}"
        )

        # Visualize inferred grasps
        if not args.no_visualization:
            for j, grasp in enumerate(grasps_inferred):
                visualize_grasp(
                    vis,
                    f"grasps_objectpc_filtered/{j:03d}/grasp",
                    grasp,
                    color=scores_inferred[j],
                    gripper_name=gripper_name,
                    linewidth=0.6,
                )

        # Save grasps to file only if output_file is not empty
        if args.output_file != "":
            print(f"Saving predicted grasps to {args.output_file}")
            
            # Save in Isaac format (YAML)
            save_to_isaac_grasp_format(
                grasps_inferred, grasp_conf_inferred, args.output_file
            )
            
            # Also save in ManiSkill format (NPZ) with .npz extension
            maniskill_output_path = args.output_file.replace('.yml', '.npz').replace('.yaml', '.npz')
            if not maniskill_output_path.endswith('.npz'):
                maniskill_output_path += '.npz'
            
            save_to_maniskill_format(
                grasps_inferred, grasp_conf_inferred, maniskill_output_path
            )
        else:
            print("No output file specified, skipping grasp saving")

    else:
        print("No grasps found from inference!")
