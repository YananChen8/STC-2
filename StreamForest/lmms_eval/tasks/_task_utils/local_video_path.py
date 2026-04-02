import json
import os
import re
from functools import lru_cache

import yaml


_REMOTE_PREFIX_RE = re.compile(r"^[A-Za-z0-9_+\-]+:s3://")


def is_remote_path(path):
    return bool(path) and ("s3://" in path or _REMOTE_PREFIX_RE.match(path) is not None)


def _normalize_env_key(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()


def _join_path(root, relative_path):
    if not root:
        return relative_path
    if is_remote_path(root):
        return root.rstrip("/") + "/" + relative_path.lstrip("/")
    return os.path.join(root, relative_path)


def _strip_remote_prefix(path):
    if not path:
        return ""
    if "s3://" not in path:
        return path.strip("/")
    return path.split("s3://", 1)[1].strip("/")


def _resolve_local_value(value, base_dir):
    if not isinstance(value, str) or not value:
        return value
    if is_remote_path(value) or os.path.isabs(value):
        return os.path.expanduser(os.path.expandvars(value))
    return os.path.abspath(os.path.join(base_dir, os.path.expanduser(os.path.expandvars(value))))


def _resolve_config_paths(node, base_dir):
    if isinstance(node, dict):
        return {key: _resolve_config_paths(value, base_dir) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_config_paths(item, base_dir) for item in node]
    return _resolve_local_value(node, base_dir)


@lru_cache(maxsize=1)
def _load_roots_config():
    config_path = os.getenv("STREAMFOREST_DATA_ROOTS_FILE", "").strip()
    if not config_path:
        return {}

    config_path = os.path.abspath(os.path.expanduser(os.path.expandvars(config_path)))
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"STREAMFOREST_DATA_ROOTS_FILE does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        if config_path.endswith(".json"):
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"STREAMFOREST_DATA_ROOTS_FILE must contain a mapping, got: {type(data).__name__}")

    return _resolve_config_paths(data, os.path.dirname(config_path))


def get_root_override(task_name, sub_task=None):
    env_keys = []
    if sub_task:
        env_keys.append(f"STREAMFOREST_{_normalize_env_key(task_name)}_{_normalize_env_key(sub_task)}_ROOT")
    env_keys.append(f"STREAMFOREST_{_normalize_env_key(task_name)}_ROOT")

    for env_key in env_keys:
        value = os.getenv(env_key, "").strip()
        if value:
            value = os.path.expanduser(os.path.expandvars(value))
            return value if is_remote_path(value) else os.path.abspath(value)

    config = _load_roots_config()
    task_config = config.get(task_name)
    if task_config is None:
        return None

    if isinstance(task_config, str):
        return task_config

    if isinstance(task_config, dict):
        if sub_task and sub_task in task_config:
            return task_config[sub_task]
        if "_default" in task_config:
            return task_config["_default"]

    return None


def resolve_video_path(task_name, sub_task, relative_path, default_root, logger=None):
    override_root = get_root_override(task_name, sub_task=sub_task)
    candidate_roots = []

    if override_root:
        candidate_roots.append(override_root)
        if default_root and not is_remote_path(override_root):
            remote_suffix = _strip_remote_prefix(default_root)
            if remote_suffix:
                candidate_roots.append(os.path.join(override_root, remote_suffix))
            default_basename = os.path.basename(default_root.rstrip("/"))
            if default_basename:
                candidate_roots.append(os.path.join(override_root, default_basename))
    elif default_root:
        candidate_roots.append(default_root)

    candidate_paths = []
    for root in candidate_roots:
        candidate = _join_path(root, relative_path)
        candidate_paths.append(candidate)

        if not is_remote_path(candidate):
            dataset_name = os.path.basename(str(root).rstrip("/"))
            if dataset_name in {"clevrer", "star"}:
                candidate_paths.append(os.path.join(os.path.dirname(str(root)), "data0613", dataset_name, relative_path))

    seen = set()
    ordered_candidates = []
    for candidate in candidate_paths:
        if candidate not in seen:
            seen.add(candidate)
            ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if is_remote_path(candidate) or os.path.exists(candidate):
            return candidate

    if logger is not None and ordered_candidates:
        logger.error(f"Video path does not exist. Tried: {ordered_candidates}")

    if ordered_candidates:
        return ordered_candidates[0]
    return relative_path
