#!/usr/bin/env python3
"""CLI for object detection using SIFT or SuperPoint+LightGlue."""

import os

import click

from detector.utils.timer import banchmark_time

DATA_DIR = "./data_example"

COMMON_OPTIONS = [
    click.argument("data_dir", metavar="DATA_DIR"),
    click.option("--output", default="./output", show_default=True, help="Output directory"),
    click.option("--min-matches", type=int, default=6, show_default=True, help="Minimum matches for homography"),
]


def common_options(fn):
    for option in reversed(COMMON_OPTIONS):
        fn = option(fn)
    return fn


def resolve_paths(data_dir):
    """Return (ref_path, scenes_dir) from a data directory with ref/ and scenes/ subdirs."""
    ref_dir = os.path.join(data_dir, "ref")
    scenes_dir = os.path.join(data_dir, "scenes")

    if not os.path.isdir(ref_dir):
        raise click.BadParameter(f"ref/ subdirectory not found in '{data_dir}'", param_hint="DATA_DIR")
    if not os.path.isdir(scenes_dir):
        raise click.BadParameter(f"scenes/ subdirectory not found in '{data_dir}'", param_hint="DATA_DIR")

    ref_files = [f for f in os.listdir(ref_dir) if os.path.isfile(os.path.join(ref_dir, f))]
    if not ref_files:
        raise click.BadParameter(f"No reference image found in '{ref_dir}'", param_hint="DATA_DIR")

    ref_path = os.path.join(ref_dir, ref_files[0])
    return ref_path, scenes_dir


def print_summary(results, output):
    print(f"{'Scene':<15} {'Found':<8} {'Confidence':<12} {'Matches'}")
    print("-" * 55)
    found_count = 0
    for name, r in results:
        status = "Yes" if r.found else "No"
        if r.found:
            found_count += 1
        print(f"{name:<15} {status:<8} {r.confidence:<12.3f} {r.num_inliers}/{r.num_matches} inliers")
    print(f"\nTotal: {found_count}/{len(results)} scenes with object detected")
    print(f"Output saved to: {os.path.abspath(output)}")


@click.group()
def cli():
    """Detect a reference object in scene images."""


@cli.command()
@common_options
@click.option("--ratio", type=float, default=0.80, show_default=True, help="Lowe's ratio test threshold")
def sift(data_dir, output, min_matches, ratio):
    """Detect using SIFT features.

    DATA_DIR must contain ref/ (reference image) and scenes/ (images to search) subdirectories.
    """
    from detector import detect_batch

    ref, scenes = resolve_paths(data_dir)
    os.makedirs(output, exist_ok=True)

    print(f"\nMethod: SIFT")
    print(f"Reference: {ref}")
    print(f"Scenes: {scenes}\n")

    results = detect_batch(ref, scenes, output, ratio=ratio, min_matches=min_matches)
    print_summary(results, output)


@cli.command()
@common_options
@click.option("--max-keypoints", type=int, default=2048, show_default=True, help="Max keypoints")
def superpoint(data_dir, output, min_matches, max_keypoints):
    """Detect using SuperPoint+LightGlue features.

    DATA_DIR must contain ref/ (reference image) and scenes/ (images to search) subdirectories.
    """
    from detector import detect_batch_sp

    ref, scenes = resolve_paths(data_dir)
    os.makedirs(output, exist_ok=True)

    print(f"\nMethod: SuperPoint+LightGlue")
    print(f"Reference: {ref}")
    print(f"Scenes: {scenes}\n")

    results = detect_batch_sp(ref, scenes, output, max_keypoints=max_keypoints, min_matches=min_matches)
    print_summary(results, output)


if __name__ == "__main__":
    cli()
