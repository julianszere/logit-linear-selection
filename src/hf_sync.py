import os
from pathlib import Path


DEFAULT_HF_SYNC = {
    "enabled": True,
    "repo_id": "julianszere/logit-linear-selection",
    "repo_type": "dataset",
    "paths": ["data", "experiments"],
    "auto_pull": True,
    "auto_push": True,
    "fail_on_error": False,
}


def hf_sync_config(cfg):
    sync_cfg = dict(DEFAULT_HF_SYNC)
    sync_cfg.update((cfg or {}).get("hf_sync") or {})
    return sync_cfg


def hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _handle_sync_error(sync_cfg, action, error):
    message = f"HF sync {action} failed: {error}"
    if sync_cfg.get("fail_on_error", False):
        raise RuntimeError(message) from error
    print(f"WARNING: {message}")


def _sync_paths(sync_cfg):
    return [Path(path) for path in sync_cfg.get("paths", [])]


def pull_hf_artifacts(cfg, reason=None):
    sync_cfg = hf_sync_config(cfg)
    if not sync_cfg.get("enabled", True) or not sync_cfg.get("auto_pull", True):
        return

    try:
        from huggingface_hub import snapshot_download

        repo_id = sync_cfg["repo_id"]
        allow_patterns = [
            f"{path.as_posix()}/**"
            for path in _sync_paths(sync_cfg)
        ]
        token = hf_token()
        print(
            "Pulling data/ and experiments/ from "
            f"https://huggingface.co/datasets/{repo_id}"
            + (f" ({reason})" if reason else "")
        )
        snapshot_download(
            repo_id=repo_id,
            repo_type=sync_cfg.get("repo_type", "dataset"),
            local_dir=Path("."),
            allow_patterns=allow_patterns,
            token=token,
        )
    except Exception as error:
        _handle_sync_error(sync_cfg, "pull", error)


def push_hf_artifacts(cfg, commit_message=None):
    sync_cfg = hf_sync_config(cfg)
    if not sync_cfg.get("enabled", True) or not sync_cfg.get("auto_push", True):
        return

    try:
        from huggingface_hub import HfApi

        token = hf_token()
        if not token:
            raise RuntimeError(
                "No Hugging Face token found. Set HF_TOKEN or "
                "HUGGINGFACE_HUB_TOKEN before running the script."
            )

        repo_id = sync_cfg["repo_id"]
        api = HfApi(token=token)
        for folder_path in _sync_paths(sync_cfg):
            if not folder_path.exists():
                continue
            path_in_repo = folder_path.as_posix()
            print(
                f"Pushing {path_in_repo}/ to "
                f"https://huggingface.co/datasets/{repo_id}"
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type=sync_cfg.get("repo_type", "dataset"),
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                commit_message=commit_message
                or f"Sync {path_in_repo} from logit-linear-selection",
                delete_patterns=[f"{path_in_repo}/**"],
                token=token,
            )
    except Exception as error:
        _handle_sync_error(sync_cfg, "push", error)
