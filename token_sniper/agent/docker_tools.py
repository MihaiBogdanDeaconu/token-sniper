from __future__ import annotations

import json
import logging
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_PULL_MAX_ATTEMPTS = 5
DEFAULT_PULL_BASE_DELAY_S = 60
DEFAULT_PULL_MAX_DELAY_S = 120
DEFAULT_RUN_RETRY_DELAY_S = 30

_docker_pull_semaphore = threading.Semaphore(1)

logger = logging.getLogger(__name__)


class DockerStartError(RuntimeError):
    def __init__(self, message: str, *, stderr: str = "", stdout: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr
        self.stdout = stdout


_DOCKER_EXEC_TIMEOUT = 180
_OUTPUT_LIMIT = 8_000
_CONDA_ACTIVATE = "source activate testbed 2>/dev/null; "


class DockerToolExecutor:
    def __init__(
        self,
        container_name: str,
        testbed: str = "/testbed",
    ) -> None:
        self._container = container_name
        self._testbed = testbed
        self._completed = False
        self._complete_message: str | None = None

    @property
    def completed(self) -> bool:
        return self._completed

    @completed.setter
    def completed(self, value: bool) -> None:
        self._completed = value

    @property
    def complete_message(self) -> str | None:
        return self._complete_message

    @complete_message.setter
    def complete_message(self, value: str | None) -> None:
        self._complete_message = value

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "bash":
            return self._bash(command=str(arguments["command"]))
        if name == "str_replace_editor":
            return self._str_replace_editor(arguments)
        if name == "complete":
            self._completed = True
            self._complete_message = str(arguments.get("message", ""))
            return {"status": "completed", "message": self._complete_message}
        raise ValueError(f"Unsupported tool: {name}")

    def generate_patch(self) -> str | None:
        self._exec("find . -name '*.bak' -delete", timeout=10)
        tracked = self._exec("git diff", timeout=30)
        if tracked is None:
            return None
        untracked_raw = self._exec(
            "git ls-files --others --exclude-standard", timeout=10,
        )
        parts: list[str] = []
        if tracked.strip():
            parts.append(tracked.rstrip("\r\n"))
        if untracked_raw:
            for new_file in untracked_raw.strip().splitlines():
                new_file = new_file.strip()
                if not new_file or new_file.endswith(".bak"):
                    continue
                content = self._exec(f"cat -- {_shell_quote(new_file)}", timeout=10)
                if content is None:
                    continue
                lines = content.splitlines()
                parts.append(
                    f"diff --git a/{new_file} b/{new_file}\n"
                    f"new file mode 100644\n"
                    f"--- /dev/null\n"
                    f"+++ b/{new_file}\n"
                    f"@@ -0,0 +1,{len(lines)} @@\n"
                    + "\n".join("+" + line for line in lines)
                )
        diff = "\n".join(parts)
        return diff if diff else None

    def _bash(self, *, command: str) -> dict[str, Any]:
        try:
            result = subprocess.run(
                [
                    "docker", "exec", "-w", self._testbed,
                    self._container, "bash", "-c",
                    _CONDA_ACTIVATE + command,
                ],
                capture_output=True,
                text=True,
                timeout=_DOCKER_EXEC_TIMEOUT,
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if len(output) > _OUTPUT_LIMIT:
                half = _OUTPUT_LIMIT // 2
                output = output[:half] + "\n\n<response clipped>\n\n" + output[-half:]
            return {"tool": "bash", "exit_code": result.returncode, "output": output}
        except subprocess.TimeoutExpired:
            return {"tool": "bash", "exit_code": -1, "output": f"Command timed out ({_DOCKER_EXEC_TIMEOUT}s)"}
        except Exception as e:
            return {"tool": "bash", "exit_code": -1, "output": f"Error: {e}"}

    def _str_replace_editor(self, args: dict[str, Any]) -> dict[str, Any]:
        cmd = args["command"]
        path_str = str(args["path"])

        if cmd == "view":
            return self._editor_view(path_str, args.get("view_range"))
        if cmd == "create":
            return self._editor_create(path_str, str(args.get("file_text", "")))
        if cmd == "str_replace":
            return self._editor_str_replace(
                path_str,
                str(args.get("old_str", "")),
                str(args.get("new_str", "")),
            )
        if cmd == "insert":
            return self._editor_insert(
                path_str,
                int(args.get("insert_line", 0)),
                str(args.get("new_str", "")),
            )
        if cmd == "undo_edit":
            return self._editor_undo(path_str)
        return {"error": f"Unknown editor command: {cmd}"}

    def _editor_view(self, path: str, view_range: list[int] | None) -> dict[str, Any]:
        file_type = self._exec(
            f"test -d {_shell_quote(path)} && echo dir || (test -f {_shell_quote(path)} && echo file || echo missing)",
            timeout=5,
        )
        if file_type is None:
            return {"error": f"Cannot access path: {path}"}
        file_type = file_type.strip()

        if file_type == "dir":
            listing = self._exec(
                f"find {_shell_quote(path)} -maxdepth 2 -not -name '.*' | head -200",
                timeout=10,
            )
            return {
                "tool": "str_replace_editor",
                "command": "view",
                "content": listing or "",
            }
        if file_type == "missing":
            return {"error": f"File not found: {path}"}

        if view_range:
            start = max(view_range[0], 1)
            end = view_range[1] if len(view_range) > 1 and view_range[1] != -1 else ""
            content = self._exec(
                f"sed -n '{start},{end}p' {_shell_quote(path)} | cat -n | sed 's/^ *\\([0-9]*\\)/\\1/' | awk -v s={start - 1} '{{printf \"%6d\\t%s\\n\", NR+s, $0}}'",
                timeout=10,
            )
            if content is None:
                content = self._exec(f"sed -n '{start},{end}p' {_shell_quote(path)}", timeout=10) or ""
        else:
            content = self._exec(f"cat -n {_shell_quote(path)}", timeout=10) or ""

        if len(content) > 12_000:
            content = content[:6_000] + "\n\n<response clipped>\n\n" + content[-6_000:]
        return {
            "tool": "str_replace_editor",
            "command": "view",
            "path": path,
            "content": content,
        }

    def _editor_create(self, path: str, content: str) -> dict[str, Any]:
        exists = self._exec(f"test -e {_shell_quote(path)} && echo yes || echo no", timeout=5)
        if exists and exists.strip() == "yes":
            return {"error": f"File already exists: {path}. Use str_replace to edit it."}
        self._exec(f"mkdir -p $(dirname {_shell_quote(path)})", timeout=5)
        self._write_file(path, content)
        return {"tool": "str_replace_editor", "command": "create", "path": path}

    def _editor_str_replace(self, path: str, old_str: str, new_str: str) -> dict[str, Any]:
        exists = self._exec(f"test -f {_shell_quote(path)} && echo yes || echo no", timeout=5)
        if not exists or exists.strip() != "yes":
            return {"error": f"File not found: {path}"}

        content = self._read_file(path)
        if content is None:
            return {"error": f"Could not read {path}"}

        count = content.count(old_str)
        if count == 0:
            return {
                "error": (
                    f"old_str not found in {path}. "
                    "Make sure it matches the file content exactly, including whitespace."
                ),
            }
        if count > 1:
            return {
                "error": (
                    f"old_str found {count} times in {path}. "
                    "Include more surrounding context to make it unique."
                ),
            }

        self._exec(f"cp {_shell_quote(path)} {_shell_quote(path + '.bak')}", timeout=5)
        new_content = content.replace(old_str, new_str, 1)
        self._write_file(path, new_content)
        return {
            "tool": "str_replace_editor",
            "command": "str_replace",
            "path": path,
            "status": "ok",
        }

    def _editor_insert(self, path: str, line_num: int, new_str: str) -> dict[str, Any]:
        exists = self._exec(f"test -f {_shell_quote(path)} && echo yes || echo no", timeout=5)
        if not exists or exists.strip() != "yes":
            return {"error": f"File not found: {path}"}

        content = self._read_file(path)
        if content is None:
            return {"error": f"Could not read {path}"}

        self._exec(f"cp {_shell_quote(path)} {_shell_quote(path + '.bak')}", timeout=5)
        lines = content.splitlines(keepends=True)
        insert_idx = min(line_num, len(lines))
        new_lines = new_str.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        lines[insert_idx:insert_idx] = new_lines
        self._write_file(path, "".join(lines))
        return {
            "tool": "str_replace_editor",
            "command": "insert",
            "path": path,
            "status": "ok",
        }

    def _editor_undo(self, path: str) -> dict[str, Any]:
        bak = path + ".bak"
        exists = self._exec(f"test -f {_shell_quote(bak)} && echo yes || echo no", timeout=5)
        if not exists or exists.strip() != "yes":
            return {"error": f"No edit to undo for {path}"}
        self._exec(f"mv {_shell_quote(bak)} {_shell_quote(path)}", timeout=5)
        return {
            "tool": "str_replace_editor",
            "command": "undo_edit",
            "path": path,
            "status": "ok",
        }

    def _exec(self, command: str, *, timeout: int = _DOCKER_EXEC_TIMEOUT) -> str | None:
        try:
            result = subprocess.run(
                [
                    "docker", "exec", "-w", self._testbed,
                    self._container, "bash", "-c",
                    _CONDA_ACTIVATE + command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout
        except Exception:
            return None

    def _read_file(self, path: str) -> str | None:
        return self._exec(f"cat {_shell_quote(path)}", timeout=15)

    def _write_file(self, path: str, content: str) -> bool:
        try:
            subprocess.run(
                [
                    "docker", "exec", "-i", "-w", self._testbed,
                    self._container, "bash", "-c",
                    f"cat > {_shell_quote(path)}",
                ],
                input=content,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return True
        except Exception:
            return False


def _docker_pull(
    image_tag: str,
    *,
    pull_timeout: int = 600,
    max_attempts: int = DEFAULT_PULL_MAX_ATTEMPTS,
    base_delay_s: float = DEFAULT_PULL_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_PULL_MAX_DELAY_S,
) -> None:
    with _docker_pull_semaphore:
        _docker_pull_locked(
            image_tag,
            pull_timeout=pull_timeout,
            max_attempts=max_attempts,
            base_delay_s=base_delay_s,
            max_delay_s=max_delay_s,
        )


def _docker_pull_locked(
    image_tag: str,
    *,
    pull_timeout: int = 600,
    max_attempts: int = DEFAULT_PULL_MAX_ATTEMPTS,
    base_delay_s: float = DEFAULT_PULL_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_PULL_MAX_DELAY_S,
) -> None:
    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, max_attempts + 1):
        logger.info(
            "Pulling image %s (attempt %d/%d)...",
            image_tag, attempt, max_attempts,
        )
        result = subprocess.run(
            ["docker", "pull", image_tag],
            capture_output=True,
            text=True,
            check=False,
            timeout=pull_timeout,
        )
        if result.returncode == 0:
            return

        last_stdout = result.stdout or ""
        last_stderr = result.stderr or ""
        detail = (last_stderr or last_stdout).strip()
        logger.warning(
            "Docker pull failed for %s (attempt %d/%d): %s",
            image_tag, attempt, max_attempts, detail or f"exit {result.returncode}",
        )
        if attempt >= max_attempts:
            break

        delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
        delay += random.uniform(0, delay * 0.1)
        logger.info("Retrying docker pull for %s in %.1fs", image_tag, delay)
        time.sleep(delay)

    raise DockerStartError(
        f"Failed to pull image {image_tag} after {max_attempts} attempts",
        stderr=last_stderr,
        stdout=last_stdout,
    )


def _docker_run_task_container(
    *,
    container_name: str,
    image_tag: str,
    timeout: int,
    max_attempts: int = 2,
) -> None:
    command = [
        "docker", "run", "-d",
        "--name", container_name,
        "--network", "none",
        image_tag,
        "sleep", "infinity",
    ]
    last_stdout = ""
    last_stderr = ""
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode == 0:
            return

        last_stdout = result.stdout or ""
        last_stderr = result.stderr or ""
        detail = (last_stderr or last_stdout).strip()
        logger.warning(
            "Docker run failed for %s (attempt %d/%d): %s",
            container_name, attempt, max_attempts, detail or f"exit {result.returncode}",
        )
        _remove_container(container_name)
        if attempt >= max_attempts:
            break
        time.sleep(DEFAULT_RUN_RETRY_DELAY_S)

    raise DockerStartError(
        f"Failed to start container {container_name} from {image_tag}",
        stderr=last_stderr,
        stdout=last_stdout,
    )


def start_task_container(
    instance_id: str,
    *,
    image_tag: str | None = None,
    timeout: int = 30,
    pull_timeout: int = 600,
    pull_max_attempts: int = DEFAULT_PULL_MAX_ATTEMPTS,
    pull_base_delay_s: float = DEFAULT_PULL_BASE_DELAY_S,
) -> str:
    container_name = f"swebench-{instance_id.replace('/', '__').replace(':', '_')}"
    if image_tag is None:
        safe_id = instance_id.replace("/", "__").replace(":", "_")
        image_tag = f"sweb.eval.x86_64.{safe_id}:latest"

    _remove_container(container_name)

    inspect_result = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        capture_output=True, timeout=10,
    )
    if inspect_result.returncode != 0:
        _docker_pull(
            image_tag,
            pull_timeout=pull_timeout,
            max_attempts=pull_max_attempts,
            base_delay_s=pull_base_delay_s,
        )

    logger.info("Starting container %s from image %s", container_name, image_tag)
    _docker_run_task_container(
        container_name=container_name,
        image_tag=image_tag,
        timeout=timeout,
    )
    return container_name


def stop_task_container(container_name: str) -> None:
    _remove_container(container_name)


def _patch_swebench_adduser() -> None:
    try:
        import swebench.harness.dockerfiles.python as _pydf

        _ADDUSER = "RUN adduser --disabled-password --gecos 'dog' nonroot"
        _USERADD = "RUN useradd -m -s /bin/bash nonroot"

        for attr in ("_DOCKERFILE_BASE_PY",):
            original = getattr(_pydf, attr, "")
            if _ADDUSER in original:
                setattr(_pydf, attr, original.replace(_ADDUSER, _USERADD))
                logger.info("Patched %s.%s: adduser -> useradd", _pydf.__name__, attr)

        for lang in ("rust", "ruby", "php", "javascript", "go", "java", "c"):
            try:
                mod = __import__(f"swebench.harness.dockerfiles.{lang}", fromlist=["_"])
                for name in dir(mod):
                    val = getattr(mod, name, "")
                    if isinstance(val, str) and _ADDUSER in val:
                        setattr(mod, name, val.replace(_ADDUSER, _USERADD))
                        logger.info("Patched %s.%s: adduser -> useradd", mod.__name__, name)
            except ImportError:
                pass
    except Exception as exc:
        logger.warning("Failed to patch swebench adduser: %s", exc)


def build_swebench_images(
    dataset: list[dict[str, Any]],
    *,
    max_workers: int = 4,
    force_rebuild: bool = False,
) -> tuple[list[str], list[str]]:
    import docker
    from swebench.harness.docker_build import build_env_images, build_instance_images
    from swebench.harness.test_spec.test_spec import make_test_spec

    _patch_swebench_adduser()

    client = docker.from_env()
    test_specs = [make_test_spec(inst) for inst in dataset]

    logger.info("Building env images...")
    try:
        _, env_failed = build_env_images(
            client=client,
            dataset=test_specs,
            force_rebuild=force_rebuild,
            max_workers=max_workers,
        )
        if env_failed:
            logger.warning("Failed env images: %d", len(env_failed))
    except Exception as e:
        logger.error("Env/base image build failed: %s — retrying once...", e)
        try:
            _, env_failed = build_env_images(
                client=client,
                dataset=test_specs,
                force_rebuild=True,
                max_workers=1,
            )
        except Exception as e2:
            logger.error("Env image build retry also failed: %s", e2)
            all_ids = [spec.instance_id for spec in test_specs]
            return [], all_ids

    try:
        successful, failed = build_instance_images(
            client=client,
            dataset=test_specs,
            force_rebuild=force_rebuild,
            max_workers=max_workers,
        )
        successful_ids = [s.instance_id for s in successful]
        failed_ids = [f.instance_id for f in failed]
    except Exception as e:
        logger.warning("Instance image build batch failed: %s", e)
        successful_ids = []
        failed_ids = []
        for spec in test_specs:
            try:
                client.images.get(spec.instance_image_key)
                successful_ids.append(spec.instance_id)
            except Exception:
                failed_ids.append(spec.instance_id)

    if failed_ids:
        logger.warning(
            "Failed to build images for %d instances: %s",
            len(failed_ids), failed_ids[:10],
        )
    logger.info(
        "Image build complete: %d succeeded, %d failed",
        len(successful_ids), len(failed_ids),
    )
    return successful_ids, failed_ids


def _remove_container(name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True, check=False, timeout=15,
    )


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
