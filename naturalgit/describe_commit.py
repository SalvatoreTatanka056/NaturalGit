"""
Module for generating detailed, metadata-rich Git commit messages.
Includes user information (Git/OS/Host) and detailed file metrics.
"""
from pathlib import Path
from typing import Optional, Tuple
from .git_context import GitContext


def generate_detailed_commit(
    ctx: GitContext,
    title: Optional[str] = None,
    add_all: bool = True
) -> Tuple[str, str, str]:
    """
    Generates a descriptive commit with user and files metadata.
    Returns:
        (command_str, explanation, full_message)
    """
    ctx._refresh()
    user_info = ctx.get_user_info()
    commit_title, commit_body, added, deleted = ctx.build_detailed_commit_parts(title=title, add_all=add_all)

    full_message = f"{commit_title}\n\n{commit_body}\n"

    # Save to .git/NATURALGIT_COMMIT_MSG or ~/.naturalgit/COMMIT_MSG
    repo_root = ctx.cwd or "."
    git_dir = Path(repo_root) / ".git"
    if git_dir.is_dir():
        msg_file = git_dir / "NATURALGIT_COMMIT_MSG"
    else:
        cfg_dir = Path.home() / ".naturalgit"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        msg_file = cfg_dir / "COMMIT_MSG"

    try:
        with open(msg_file, "w", encoding="utf-8") as f:
            f.write(full_message)
        msg_path_str = str(msg_file.resolve()).replace("\\", "/")
    except Exception:
        # Fallback to local file in case of path issues
        msg_file = Path("NATURALGIT_COMMIT_MSG")
        with open(msg_file, "w", encoding="utf-8") as f:
            f.write(full_message)
        msg_path_str = "NATURALGIT_COMMIT_MSG"

    # Check if git user name/email are configured in Git, if not include inline config
    git_cfg_flags = ""
    if not user_info.get("git_configured"):
        git_cfg_flags = f'-c user.name="{user_info["name"]}" -c user.email="{user_info["email"]}" '

    add_prefix = "git add -A && " if add_all else ""
    command = f'{add_prefix}git {git_cfg_flags}commit -F "{msg_path_str}"'

    file_count = len(ctx.staged_files) if not add_all else (ctx.staged_count + ctx.unstaged_count + ctx.untracked_count)
    explanation = (
        f"Crea un commit descrittivo con metadati utente ({user_info['name']} su {user_info['hostname']}), "
        f"dettaglio di {file_count} file e statistiche (+{added}/-{deleted})."
    )

    return command, explanation, full_message
