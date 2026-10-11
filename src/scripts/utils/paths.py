import os
import time

from .common import PROJECT_ROOT, project_path


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def stage_root(cfg, stage):
    return os.path.join(cfg.output_root, stage)


def find_latest_run(cfg, stage):
    root = stage_root(cfg, stage)
    runs = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    return max(runs, key=os.path.getmtime)


def resolve_run_dir(cfg, stage, run="latest"):
    if run in (None, "", "latest"):
        return find_latest_run(cfg, stage)
    if os.path.isabs(run):
        return project_path(run)
    if "/" in run or run.startswith(("~", "$", ".")):
        return project_path(run)
    return os.path.join(stage_root(cfg, stage), run)


def prepare_run_dir(cfg, stage, run_name=None):
    name = run_name or timestamp()
    root = stage_root(cfg, stage)
    os.makedirs(root, exist_ok=True)
    run_dir = os.path.join(root, name)
    if os.path.exists(run_dir):
        suffix = 1
        while os.path.exists(f"{run_dir}-{suffix:02d}"):
            suffix += 1
        run_dir = f"{run_dir}-{suffix:02d}"
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def default_config_path():
    return str(PROJECT_ROOT / "src" / "configs" / "physiodit.yaml")
