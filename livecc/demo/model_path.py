import os
from pathlib import Path


DEFAULT_MODEL_ID = "chenjoya/LiveCC-7B-Instruct"
LIVECC_MODEL_PATH_ENV = "LIVECC_MODEL_PATH"


def resolve_model_path(model_path: str | None = None) -> str:
    candidate = model_path or os.environ.get(LIVECC_MODEL_PATH_ENV) or DEFAULT_MODEL_ID
    path = Path(candidate).expanduser()
    if not path.exists():
        return candidate
    if not path.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {path}")
    if (path / "config.json").exists():
        return str(path)

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find config.json or snapshots/ under model path: {path}"
        )

    ref_path = path / "refs" / "main"
    if ref_path.exists():
        snapshot = snapshots_dir / ref_path.read_text().strip()
        if (snapshot / "config.json").exists():
            return str(snapshot)

    snapshots = sorted(
        snapshot for snapshot in snapshots_dir.iterdir() if (snapshot / "config.json").exists()
    )
    if snapshots:
        return str(snapshots[-1])

    raise FileNotFoundError(f"No valid model snapshot found under: {path}")
