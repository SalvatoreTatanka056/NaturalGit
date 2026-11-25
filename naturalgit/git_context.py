"""
Git repository inspection and contextual information.
"""
import datetime
import os
import platform
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def run_git(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Execute a git command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return -1, "", "Git non trovato nel sistema. Assicurati che 'git' sia installato e nel PATH."
    except Exception as e:
        return -1, "", str(e)


class GitContext:
    def __init__(self, cwd: Optional[str] = None):
        self.cwd = cwd
        self._refresh()

    def _refresh(self):
        code, out, _ = run_git(["rev-parse", "--is-inside-work-tree"], cwd=self.cwd)
        self.is_repo = (code == 0 and out.strip() == "true")
        
        if not self.is_repo:
            self.branch = None
            self.clean = False
            self.staged_count = 0
            self.unstaged_count = 0
            self.untracked_count = 0
            self.staged_files = []
            self.unstaged_files = []
            self.untracked_files = []
            self.remotes = []
            return

        # Current branch
        code, branch_out, _ = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=self.cwd)
        self.branch = branch_out if code == 0 else "HEAD (detached)"

        # Status porcelain
        code, status_out, _ = run_git(["status", "--porcelain"], cwd=self.cwd)
        self.staged_files = []
        self.unstaged_files = []
        self.untracked_files = []
        self.status_entries = []

        if code == 0 and status_out:
            for line in status_out.splitlines():
                if len(line) < 3:
                    continue
                index_status = line[0]
                worktree_status = line[1]
                filename = line[3:].strip()
                # Handle possible rename "old -> new"
                if " -> " in filename:
                    filename = filename.split(" -> ")[1].strip()

                self.status_entries.append({
                    "filename": filename,
                    "index": index_status,
                    "worktree": worktree_status
                })

                if index_status in ["M", "A", "D", "R", "C"]:
                    self.staged_files.append(filename)
                if worktree_status in ["M", "D"]:
                    self.unstaged_files.append(filename)
                if index_status == "?" and worktree_status == "?":
                    self.untracked_files.append(filename)

        self.staged_count = len(self.staged_files)
        self.unstaged_count = len(self.unstaged_files)
        self.untracked_count = len(self.untracked_files)
        self.clean = (self.staged_count == 0 and self.unstaged_count == 0 and self.untracked_count == 0)

        # Remotes
        code, remotes_out, _ = run_git(["remote"], cwd=self.cwd)
        self.remotes = remotes_out.splitlines() if code == 0 and remotes_out else []

    def get_user_info(self) -> Dict[str, str]:
        """Gathers Git and system user metadata."""
        code, git_name, _ = run_git(["config", "user.name"], cwd=self.cwd)
        code2, git_email, _ = run_git(["config", "user.email"], cwd=self.cwd)

        os_user = os.environ.get("USERNAME") or os.environ.get("USER") or "Utente"
        hostname = os.environ.get("COMPUTERNAME") or platform.node() or "LocalHost"
        domain = os.environ.get("USERDOMAIN", "")

        name = git_name if (code == 0 and git_name) else os_user
        email = git_email if (code2 == 0 and git_email) else f"{os_user}@{hostname.lower()}.local"

        return {
            "name": name,
            "email": email,
            "os_user": os_user,
            "hostname": hostname,
            "domain": domain,
            "git_configured": bool(git_name and git_email),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_file_change_details(self, staged_only: bool = False) -> List[Dict[str, Any]]:
        """Returns details for each modified file including status and line additions/deletions."""
        if not self.is_repo:
            return []

        # Parse numstats
        numstats: Dict[str, Dict[str, int]] = {}

        # 1. Staged stats
        code, out_cached, _ = run_git(["diff", "--cached", "--numstat"], cwd=self.cwd)
        if code == 0 and out_cached:
            for line in out_cached.splitlines():
                parts = line.split("\t")
                if len(parts) >= 3:
                    fn = parts[2].strip()
                    added = int(parts[0]) if parts[0].isdigit() else 0
                    deleted = int(parts[1]) if parts[1].isdigit() else 0
                    numstats[fn] = {"added": added, "deleted": deleted}

        # 2. Unstaged stats if not staged_only
        if not staged_only:
            code, out_unstaged, _ = run_git(["diff", "--numstat"], cwd=self.cwd)
            if code == 0 and out_unstaged:
                for line in out_unstaged.splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        fn = parts[2].strip()
                        added = int(parts[0]) if parts[0].isdigit() else 0
                        deleted = int(parts[1]) if parts[1].isdigit() else 0
                        if fn in numstats:
                            numstats[fn]["added"] += added
                            numstats[fn]["deleted"] += deleted
                        else:
                            numstats[fn] = {"added": added, "deleted": deleted}

        details = []
        for entry in self.status_entries:
            fn = entry["filename"]
            idx = entry["index"]
            wt = entry["worktree"]

            if staged_only and idx not in ["M", "A", "D", "R", "C"]:
                continue

            # Determine human-readable status
            if idx == "A" or (idx == "?" and wt == "?"):
                st_desc = "Nuovo file"
            elif idx == "D" or wt == "D":
                st_desc = "Eliminato"
            elif idx == "R":
                st_desc = "Ridenominato"
            else:
                st_desc = "Modificato"

            st_info = numstats.get(fn, {"added": 0, "deleted": 0})
            details.append({
                "filename": fn,
                "status": st_desc,
                "added": st_info["added"],
                "deleted": st_info["deleted"],
                "is_staged": idx in ["M", "A", "D", "R", "C"]
            })

        return details

    def build_detailed_commit_parts(
        self,
        title: Optional[str] = None,
        add_all: bool = True
    ) -> Tuple[str, str, int, int]:
        """
        Builds title and detailed body for a rich commit.
        Returns (commit_title, commit_body, total_added, total_deleted).
        """
        user_info = self.get_user_info()
        file_details = self.get_file_change_details(staged_only=not add_all)

        total_files = len(file_details)
        total_added = sum(f["added"] for f in file_details)
        total_deleted = sum(f["deleted"] for f in file_details)

        # Generate a descriptive title if not provided
        if not title:
            if total_files == 0:
                commit_title = "Aggiornamento modifiche"
            elif total_files == 1:
                clean_name = file_details[0]['filename'].rstrip("/\\").split("/")[-1].split("\\")[-1]
                commit_title = f"{file_details[0]['status']}: {clean_name}"
            else:
                names = [f["filename"].rstrip("/\\").split("/")[-1].split("\\")[-1] for f in file_details[:3]]
                first_few = ", ".join([n for n in names if n])
                if total_files > 3:
                    first_few += f" e altri {total_files - 3} file"
                commit_title = f"Modifiche a {total_files} file ({first_few})"

        else:
            commit_title = title

        # Build detailed body
        lines = []
        lines.append(f"Autore: {user_info['name']} <{user_info['email']}>")
        host_str = f"{user_info['os_user']}@{user_info['hostname']}"
        if user_info["domain"]:
            host_str += f" ({user_info['domain']})"
        lines.append(f"Host/Sistema: {host_str}")
        lines.append(f"Data: {user_info['timestamp']}")
        lines.append(f"Branch: {self.branch or 'main'}")
        lines.append("")

        if file_details:
            lines.append(f"File coinvolti ({total_files}):")
            for f in file_details:
                stat_parts = []
                if f["added"] > 0:
                    stat_parts.append(f"+{f['added']}")
                if f["deleted"] > 0:
                    stat_parts.append(f"-{f['deleted']}")
                stat_str = f" ({', '.join(stat_parts)})" if stat_parts else ""
                lines.append(f"  - {f['filename']} [{f['status']}]{stat_str}")
            lines.append("")

        lines.append("Riepilogo:")
        lines.append(f"  {total_files} file modificati, {total_added} inserzioni(+), {total_deleted} cancellazioni(-)")

        commit_body = "\n".join(lines)
        return commit_title, commit_body, total_added, total_deleted

    def get_diff_summary(self, max_chars: int = 1500) -> str:
        """Returns a truncated diff of staged or unstaged changes."""
        if not self.is_repo:
            return ""
        code, diff_out, _ = run_git(["diff", "--stat"], cwd=self.cwd)
        if code == 0 and diff_out:
            return diff_out[:max_chars]
        return ""

    def get_recent_commits(self, limit: int = 3) -> List[str]:
        """Returns short oneline recent commits."""
        if not self.is_repo:
            return []
        code, out, _ = run_git(["log", f"-n{limit}", "--oneline"], cwd=self.cwd)
        if code == 0 and out:
            return out.splitlines()
        return []

    def to_dict(self) -> Dict:
        """Dictionary representation of the git context."""
        return {
            "is_repo": self.is_repo,
            "branch": self.branch,
            "clean": self.clean,
            "staged_count": self.staged_count,
            "unstaged_count": self.unstaged_count,
            "untracked_count": self.untracked_count,
            "staged_files": self.staged_files[:10],
            "unstaged_files": self.unstaged_files[:10],
            "untracked_files": self.untracked_files[:10],
            "remotes": self.remotes,
            "user_info": self.get_user_info(),
            "recent_commits": self.get_recent_commits(3)
        }
