import argparse
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from box import Box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datapath import Datapath
from src.data.dataset import UniRigDataset
from src.data.extract import get_files
from src.data.raw_data import RawData
from src.data.transform import TransformConfig
from src.inference.download import download
from src.model.parse import get_model
from src.tokenizer.parse import get_tokenizer
from src.tokenizer.spec import TokenizerConfig


def load_yaml(path: str) -> Box:
    if not path.endswith(".yaml"):
        path = f"{path}.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        return Box(yaml.safe_load(handle))


def nullable_string(val):
    if not val:
        return None
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native UniRig AR inference for XPU/CUDA/CPU")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input", type=nullable_string, default=None)
    parser.add_argument("--input_dir", type=nullable_string, default=None)
    parser.add_argument("--output", type=nullable_string, default=None)
    parser.add_argument("--output_dir", type=nullable_string, default=None)
    parser.add_argument("--npz_dir", type=nullable_string, default="tmp")
    parser.add_argument("--cls", type=nullable_string, default=None)
    parser.add_argument("--data_name", type=nullable_string, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_files(task: Box, args: argparse.Namespace) -> Datapath:
    if args.input is None and args.input_dir is None:
        raise ValueError("input or input_dir must be specified")
    if args.output is None and args.output_dir is None:
        raise ValueError("output or output_dir must be specified")
    files = get_files(
        data_name=task.components.data_name,
        inputs=args.input,
        input_dataset_dir=args.input_dir,
        output_dataset_dir=args.npz_dir,
        force_override=True,
        warning=False,
    )
    return Datapath(files=[out_dir for _, out_dir in files], cls=args.cls)


def build_model(task: Box):
    tokenizer_config = load_yaml(os.path.join("configs/tokenizer", task.components.tokenizer))
    tokenizer = get_tokenizer(config=TokenizerConfig.parse(config=tokenizer_config))
    model_config = load_yaml(os.path.join("configs/model", task.components.model))
    return get_model(tokenizer=tokenizer, **model_config)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned[key.removeprefix("model.")] = value
        elif not key.startswith(("optimizer", "scheduler", "callbacks", "loops")):
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"warning: missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected keys while loading checkpoint: {len(unexpected)}")


def iter_dataset(datapath: Datapath, model, data_name: str, transform_config: TransformConfig) -> Iterable[dict]:
    dataset = UniRigDataset(
        process_fn=model._process_fn,
        data=datapath.get_data(),
        name="predict-native",
        tokenizer=None,
        transform_config=transform_config,
        debug=False,
        data_name=data_name,
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = dataset.collate_fn([sample])
        yield batch


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def export_prediction(batch: dict, prediction, output_dir: str | None, output_name: str | None, npz_dir: str) -> None:
    origin_vertices = batch["origin_vertices"]
    origin_vertex_normals = batch["origin_vertex_normals"]
    origin_faces = batch["origin_faces"]
    origin_face_normals = batch["origin_face_normals"]
    num_points = batch["num_points"]
    num_faces = batch["num_faces"]
    paths = batch["path"]

    if isinstance(origin_vertices, torch.Tensor):
        origin_vertices = origin_vertices.detach().cpu().numpy()
    if isinstance(origin_vertex_normals, torch.Tensor):
        origin_vertex_normals = origin_vertex_normals.detach().cpu().numpy()
    if isinstance(origin_faces, torch.Tensor):
        origin_faces = origin_faces.detach().cpu().numpy()
    if isinstance(origin_face_normals, torch.Tensor):
        origin_face_normals = origin_face_normals.detach().cpu().numpy()
    if isinstance(num_points, torch.Tensor):
        num_points = num_points.detach().cpu().numpy()
    if isinstance(num_faces, torch.Tensor):
        num_faces = num_faces.detach().cpu().numpy()

    rel_path = os.path.relpath(paths[0], npz_dir)
    stem = Path(rel_path)
    if output_name is not None:
        export_path = output_name
    else:
        base_dir = output_dir if output_dir is not None else paths[0]
        export_path = os.path.join(base_dir, str(stem), "skeleton.fbx")

    raw_data = RawData(
        vertices=origin_vertices[0, : num_points[0]],
        vertex_normals=origin_vertex_normals[0, : num_points[0]],
        faces=origin_faces[0, : num_faces[0]],
        face_normals=origin_face_normals[0, : num_faces[0]],
        joints=prediction.joints,
        tails=prediction.tails,
        parents=prediction.parents,
        skin=None,
        no_skin=prediction.no_skin,
        names=prediction.names,
        matrix_local=None,
        path=None,
        cls=prediction.cls,
    )
    raw_data.export_fbx(path=export_path)
    print(export_path)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    task = load_yaml(args.task)
    if task.mode != "predict":
        raise ValueError(f"predict_ar_xpu.py only supports predict mode, found {task.mode}")

    device = choose_device(args.device)
    print(f"using device: {device}")

    datapath = resolve_files(task, args)
    transform_config = TransformConfig.parse(
        config=load_yaml(os.path.join("configs/transform", task.components.transform)).predict_transform_config
    )
    data_name = task.components.get("data_name", "raw_data.npz")
    if args.data_name is not None:
        data_name = args.data_name

    model = build_model(task)
    checkpoint_path = download(task.get("resume_from_checkpoint", None))
    if checkpoint_path is None:
        raise ValueError("missing resume_from_checkpoint in task config")
    load_checkpoint(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    system_config = load_yaml(os.path.join("configs/system", task.components.system))
    generate_kwargs = dict(system_config.get("generate_kwargs", {}))

    with torch.no_grad():
        for batch in iter_dataset(datapath, model, data_name, transform_config):
            batch = move_batch_to_device(batch, device)
            batch["generate_kwargs"] = generate_kwargs
            predictions = model.predict_step(batch)
            if not predictions:
                raise RuntimeError(f"prediction failed for {batch['path'][0]}")
            export_prediction(
                batch=batch,
                prediction=predictions[0],
                output_dir=args.output_dir,
                output_name=args.output,
                npz_dir=args.npz_dir,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
