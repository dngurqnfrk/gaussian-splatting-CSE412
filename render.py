#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import json
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 통계 수집을 위한 리스트
    fps_list = []
    render_time_list = []
    memory_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_pkg["render"]
        
        # 성능 통계 수집
        fps_list.append(render_pkg.get("fps", 0.0))
        render_time_list.append(render_pkg.get("render_time_ms", 0.0))
        memory_list.append(render_pkg.get("memory_used_mb", 0.0))
        
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
    # 통계 계산
    if fps_list:
        stats = {
            "num_frames": len(fps_list),
            "fps": {
                "mean": float(np.mean(fps_list)),
                "std": float(np.std(fps_list)),
                "min": float(np.min(fps_list)),
                "max": float(np.max(fps_list)),
                "median": float(np.median(fps_list))
            },
            "render_time_ms": {
                "mean": float(np.mean(render_time_list)),
                "std": float(np.std(render_time_list)),
                "min": float(np.min(render_time_list)),
                "max": float(np.max(render_time_list)),
                "median": float(np.median(render_time_list))
            },
            "memory_mb": {
                "mean": float(np.mean(memory_list)),
                "std": float(np.std(memory_list)),
                "min": float(np.min(memory_list)),
                "max": float(np.max(memory_list)),
                "median": float(np.median(memory_list))
            }
        }
        
        # 콘솔 출력
        print(f"\n{'='*60}")
        print(f"Rendering Performance Statistics ({name} set)")
        print(f"{'='*60}")
        print(f"Frames rendered: {stats['num_frames']}")
        print(f"\nFPS:")
        print(f"  Mean:   {stats['fps']['mean']:.2f} fps")
        print(f"  Median: {stats['fps']['median']:.2f} fps")
        print(f"  Min:    {stats['fps']['min']:.2f} fps")
        print(f"  Max:    {stats['fps']['max']:.2f} fps")
        print(f"\nRender Time:")
        print(f"  Mean:   {stats['render_time_ms']['mean']:.2f} ms")
        print(f"  Median: {stats['render_time_ms']['median']:.2f} ms")
        print(f"\nMemory Usage:")
        print(f"  Mean:   {stats['memory_mb']['mean']:.2f} MB")
        print(f"  Max:    {stats['memory_mb']['max']:.2f} MB")
        print(f"{'='*60}\n")
        
        # JSON 파일로 저장
        stats_path = os.path.join(model_path, name, "ours_{}".format(iteration), "performance_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Performance statistics saved to: {stats_path}")
    
    return stats if fps_list else None

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)