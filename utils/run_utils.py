import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf


RUN_ID_ENV_VAR = "EQF_RUN_ID"


def _find_repo_root(start: Optional[Path] = None) -> Path:
    path = (start or Path.cwd()).resolve()
    while not (path / ".git").exists():
        if path == path.parent:
            return Path.cwd()
        path = path.parent
    return path


def _get_cli_value(argv: List[str], key: str) -> Optional[str]:
    for arg in argv:
        if "=" not in arg:
            continue
        arg_key, value = arg.split("=", 1)
        if arg_key.lstrip("+") == key:
            return value
    return None


def _normalize_optional_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value.lower() in {"", "null", "none"}:
        return None
    return value


def _normalize_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def _set_run_id(run_id: str) -> str:
    os.environ[RUN_ID_ENV_VAR] = run_id
    return run_id


def _infer_requeue_run_id(repo_root: Path) -> Optional[str]:
    job_id = os.environ.get("SLURM_JOB_ID", "manual_requeue_location")
    link_path = repo_root / "outputs/.requeue_links" / job_id
    if not link_path.exists():
        return None

    requeue_meta_path = link_path.resolve() / "requeue_meta.yaml"
    if not requeue_meta_path.exists():
        return None

    meta = OmegaConf.to_container(OmegaConf.load(str(requeue_meta_path)), resolve=True)
    if not isinstance(meta, dict) or "resume" not in meta:
        return None
    return _normalize_optional_string(meta["resume"])


def _infer_shortcode_run_id(argv: List[str], repo_root: Path) -> Optional[str]:
    shortcode = _normalize_optional_string(_get_cli_value(argv, "shortcode"))
    if shortcode is None:
        return None

    shortcode_path = repo_root / "configurations" / "shortcode" / f"{shortcode}.yaml"
    if not shortcode_path.exists():
        shortcode_path = repo_root / "configurations" / "shortcode" / shortcode / "base.yaml"
        if not shortcode_path.exists():
            return None

    shortcode_cfg = OmegaConf.to_container(OmegaConf.load(str(shortcode_path)), resolve=False)
    if not isinstance(shortcode_cfg, dict) or "resume" not in shortcode_cfg:
        return None
    return _normalize_optional_string(shortcode_cfg["resume"])


def generate_new_run_id() -> str:
    """Generate a short W&B-style run id."""
    from wandb.util import generate_id

    return _set_run_id(generate_id())


def choose_run_id(argv: List[str], repo_root: Optional[Path] = None) -> str:
    existing = _normalize_optional_string(os.environ.get(RUN_ID_ENV_VAR))
    if existing is not None:
        return existing

    repo_root = _find_repo_root(repo_root)

    if _normalize_bool(_get_cli_value(argv, "requeue")):
        requeue_run_id = _infer_requeue_run_id(repo_root)
        if requeue_run_id is not None:
            return _set_run_id(requeue_run_id)

    resume_id = _normalize_optional_string(_get_cli_value(argv, "resume"))
    if resume_id is not None:
        return _set_run_id(resume_id)

    shortcode_run_id = _infer_shortcode_run_id(argv, repo_root)
    if shortcode_run_id is not None:
        return _set_run_id(shortcode_run_id)

    return generate_new_run_id()


def get_current_run_id() -> str:
    run_id = _normalize_optional_string(os.environ.get(RUN_ID_ENV_VAR))
    if run_id is not None:
        return run_id
    return choose_run_id(sys.argv)


def make_timestamped_name(base_name: str, run_id: Optional[str] = None) -> str:
    """Return a readable, collision-resistant name for snapshot/log directories."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    return f"{timestamp}-{base_name}-{run_id or get_current_run_id()}"


def replace_symlink(link_path: Path, target: Path, target_is_directory: bool = True) -> None:
    """Atomically replace a symlink without the unlink/create race window."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    temp_link = link_path.with_name(
        f".{link_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temp_link.unlink(missing_ok=True)
    temp_link.symlink_to(target, target_is_directory=target_is_directory)
    os.replace(temp_link, link_path)
