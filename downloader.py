import hashlib
import html
import json
import os
import queue
import re
import shutil
import stat
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from PySide6.QtCore import Qt, QTimer, QStandardPaths, QUrl
from PySide6.QtGui import QDesktopServices, QColor
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
                               QComboBox, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QLineEdit, QFileDialog, QProgressBar, QCheckBox,
                               QMessageBox, QHeaderView)

USER_AGENT = "TorchCUDAChecker"
PYTORCH_INDEX = "https://download.pytorch.org/whl"
PYPI_JSON = "https://pypi.org/pypi"
NVIDIA_REDIST = "https://developer.download.nvidia.com/compute"
GITHUB_API = "https://api.github.com/repos"

CUDA_COMPONENT_KEYS = {
    "cuda-nvrtc": "cuda_nvrtc", "cuda-runtime": "cuda_cudart", "cuda-nvcc": "cuda_nvcc",
    "cuda-cupti": "cuda_cupti", "cublas": "libcublas", "cufft": "libcufft",
    "curand": "libcurand", "cusolver": "libcusolver", "cusparse": "libcusparse",
    "nvtx": "cuda_nvtx", "nvjitlink": "libnvjitlink",
}

NVIDIA_PRODUCTS = {"CUDA Toolkit": "cuda", "cuDNN": "cudnn"}


def nvidia_platform(platform):
    return "windows-x86_64" if platform == "windows" else "linux-x86_64"


def human_size(size):
    if not size:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def _open(url, method="GET", timeout=30):
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def _describe(error):
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code} from {urllib.parse.urlsplit(error.url or '').netloc}"
    if isinstance(error, urllib.error.URLError):
        return f"Network error: {error.reason}"
    return str(error) or error.__class__.__name__


def _version_key(version):
    return [(1, int(t)) if t.isdigit() else (0, t) for t in re.findall(r"\d+|[a-z]+", version.lower())]


def _wheel_tags(filename):
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6):
        return None
    return parts[0], parts[1], parts[-3], parts[-2], parts[-1]


def _python_ok(python_tag, abi_tag, python):
    minor = int(python.split(".")[1])
    for tag in python_tag.split("."):
        impl, digits = tag[:2], tag[2:]
        if impl not in ("cp", "py") or not digits.startswith("3"):
            continue
        tag_minor = int(digits[1:]) if len(digits) > 1 else None
        if impl == "cp" and abi_tag == tag and tag_minor == minor:
            return True
        if abi_tag == "abi3" and tag_minor is not None and tag_minor <= minor:
            return True
        if impl == "py" and abi_tag == "none" and (tag_minor is None or tag_minor <= minor):
            return True
    return False


def _platform_ok(platform_tag, platform):
    tags = platform_tag.split(".")
    if "any" in tags:
        return True
    if platform == "windows":
        return "win_amd64" in tags
    return any(t.endswith("_x86_64") and t.startswith(("manylinux", "linux")) for t in tags)


class Artifact:
    def __init__(self, filename, url, sha256=None, size=None, version="", note="", unpack_into=None, description=""):
        self.filename = filename
        self.url = url
        self.sha256 = sha256.lower() if sha256 else None
        self.size = size
        self.version = version
        self.note = note
        self.unpack_into = unpack_into
        self.description = description
        self.instructions = ""


class Sources:
    def __init__(self):
        self._cache = {}
        self._locks = {}
        self._guard = threading.Lock()

    def fetch(self, url, method="GET"):
        key = (method, url)
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self._cache:
                with _open(url, method) as response:
                    self._cache[key] = response.headers if method == "HEAD" else response.read()
            return self._cache[key]

    def _json(self, url):
        return json.loads(self.fetch(url))

    def content_length(self, url):
        try:
            return int(self.fetch(url, "HEAD").get("Content-Length") or 0) or None
        except (urllib.error.URLError, ValueError, OSError):
            return None

    def pytorch_wheel(self, package, version, moniker, python, platform):
        index = f"{PYTORCH_INDEX}/{moniker}/{package}/"
        page = self.fetch(index).decode("utf-8", "replace")
        wanted = (f"{version}+{moniker}", version)
        best = None
        for href, text in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', page):
            name = html.unescape(text).strip()
            tags = _wheel_tags(name)
            if not tags or tags[0].lower().replace("_", "-") != package or tags[1] not in wanted:
                continue
            if not (_python_ok(tags[2], tags[3], python) and _platform_ok(tags[4], platform)):
                continue
            rank = (wanted.index(tags[1]), 0 if "manylinux" in tags[4] or "win" in tags[4] else 1)
            if best is None or rank < best[0]:
                best = (rank, html.unescape(href), name)
        if best is None:
            raise LookupError(f"No {package} {version} wheel for Python {python} on the {moniker} index")
        url, _, fragment = urllib.parse.urljoin(index, best[1]).partition("#")
        sha256 = fragment[len("sha256="):] if fragment.startswith("sha256=") else None
        return Artifact(best[2], url, sha256, self.content_length(url), version)

    def pypi_wheel(self, project, version, python, platform, prefix=False):
        releases = self._json(f"{PYPI_JSON}/{project}/json")["releases"]
        best = None
        for release, files in releases.items():
            if not (release == version or (prefix and release.startswith(version + "."))):
                continue
            if prefix and re.search(r"(a|b|rc|dev)\d*$", release):
                continue
            for f in files:
                tags = _wheel_tags(f["filename"])
                if f.get("yanked") or not tags:
                    continue
                if _python_ok(tags[2], tags[3], python) and _platform_ok(tags[4], platform):
                    if best is None or _version_key(release) > _version_key(best[0]):
                        best = (release, f)
                    break
        if best is None:
            spec = f"{version}.*" if prefix else version
            raise LookupError(f"No {project} {spec} wheel for Python {python} on PyPI")
        release, f = best
        return Artifact(f["filename"], f["url"], f["digests"].get("sha256"), f.get("size"), release)

    def pypi_sdist(self, project, version, note=""):
        files = self._json(f"{PYPI_JSON}/{project}/json")["releases"].get(version, [])
        for f in files:
            if f["packagetype"] == "sdist" and not f.get("yanked"):
                return Artifact(f["filename"], f["url"], f["digests"].get("sha256"), f.get("size"), version, note)
        raise LookupError(f"No {project} {version} source package on PyPI")

    def github_asset(self, url, version=""):
        parts = urllib.parse.urlsplit(url).path.strip("/").split("/")
        owner, repo, tag, quoted = parts[0], parts[1], parts[4], parts[5]
        filename = urllib.parse.unquote(quoted)
        try:
            assets = self._json(f"{GITHUB_API}/{owner}/{repo}/releases/tags/{tag}").get("assets", [])
        except urllib.error.URLError as error:
            return Artifact(filename, url, version=version, note=f"No checksum: GitHub API unavailable ({_describe(error)})")
        for asset in assets:
            if asset.get("name") == filename:
                digest = asset.get("digest") or ""
                sha256 = digest.split(":", 1)[1] if digest.startswith("sha256:") else None
                return Artifact(filename, url, sha256, asset.get("size"), version)
        raise LookupError(f"{filename} is not in {owner}/{repo} {tag}")

    def nvidia_releases(self, product):
        page = self.fetch(f"{NVIDIA_REDIST}/{product}/redist/").decode("utf-8", "replace")
        releases = set(re.findall(r"redistrib_(\d+\.\d+\.\d+)\.json", page))
        return sorted(releases, key=lambda v: [int(p) for p in v.split(".")], reverse=True)

    def nvidia_manifest(self, product, release):
        return self._json(f"{NVIDIA_REDIST}/{product}/redist/redistrib_{release}.json")

    def nvidia_artifacts(self, product, release, platform, variant=None, keys=None):
        return manifest_artifacts(self.nvidia_manifest(product, release), product, platform, variant, keys)

    def nvidia_artifact(self, product, release, platform, key, variant=None):
        found = self.nvidia_artifacts(product, release, platform, variant, {key})
        if key not in found:
            where = f"{platform} {variant}" if variant else platform
            raise LookupError(f"NVIDIA {product} {release} has no {key} for {where}")
        return found[key]

    def cudnn_archive(self, version, platform, variant):
        release = ".".join(version.split(".")[:3])
        artifact = self.nvidia_artifact("cudnn", release, platform, "cudnn", variant)
        if artifact.version != version:
            raise LookupError(f"NVIDIA cuDNN {release} is {artifact.version}, not {version}")
        return artifact



def manifest_artifacts(manifest, product, platform, variant=None, keys=None):
    artifacts = {}
    for key, component in manifest.items():
        if not isinstance(component, dict) or "name" not in component:
            continue
        if keys is not None and key not in keys:
            continue
        target = platform if platform in component else ("source" if "source" in component else None)
        if target is None:
            continue
        entry, into = component[target], target
        if "relative_path" not in entry:
            if not variant or variant not in entry:
                continue
            entry, into = entry[variant], f"{target}/{variant}"
        artifacts[key] = Artifact(
            os.path.basename(entry["relative_path"]),
            f"{NVIDIA_REDIST}/{product}/redist/{entry['relative_path']}",
            entry.get("sha256"), int(entry["size"]) if entry.get("size") else None,
            component.get("version", ""), unpack_into=into, description=component.get("name", ""))
    return artifacts


def manifest_platforms(manifest):
    platforms = set()
    for component in manifest.values():
        if isinstance(component, dict) and "name" in component:
            platforms.update(k for k in component if "-" in k)
    return sorted(platforms)


def manifest_variants(manifest, platform):
    variants = set()
    for component in manifest.values():
        if isinstance(component, dict) and isinstance(component.get(platform), dict):
            entry = component[platform]
            if "relative_path" not in entry:
                variants.update(entry)
    return sorted(variants, key=_version_key, reverse=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_writable(root):
    for base, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            if not os.path.islink(path):
                os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)


def unpack_archive(path, folder, into):
    target = os.path.join(folder, *into.split("/"))
    scratch = tempfile.mkdtemp(prefix=".unpack-", dir=folder)
    try:
        if path.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(scratch)
        else:
            with tarfile.open(path) as archive:
                archive.extractall(scratch, filter="data")
        _make_writable(scratch)
        entries = os.listdir(scratch)
        if not entries:
            raise ValueError("Archive is empty")
        single = len(entries) == 1 and os.path.isdir(os.path.join(scratch, entries[0]))
        shutil.copytree(os.path.join(scratch, entries[0]) if single else scratch, target,
                        symlinks=True, dirs_exist_ok=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return target


class _Cancelled(Exception):
    pass


def download_artifact(artifact, folder, cancel, report):
    path = os.path.join(folder, artifact.filename)
    if os.path.exists(path):
        if artifact.sha256 and _sha256(path) == artifact.sha256:
            return path, "Already downloaded, verified"
        if not artifact.sha256 and artifact.size and os.path.getsize(path) == artifact.size:
            return path, "Already downloaded"
    partial = path + ".part"
    digest = hashlib.sha256()
    done = 0
    try:
        with _open(artifact.url, timeout=60) as response, open(partial, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0) or artifact.size or 0
            while not cancel.is_set():
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                report(done, total)
        if cancel.is_set():
            raise _Cancelled()
        if artifact.sha256 and digest.hexdigest() != artifact.sha256:
            raise ValueError("Checksum mismatch, file deleted")
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, path)
    return path, "Done, verified" if artifact.sha256 else "Done (no checksum published)"


class _Entry:
    def __init__(self, item, versions, resolver):
        self.item = item
        self.versions = versions
        self.version = versions[0] if versions else ""
        self.resolver = resolver
        self.artifact = None
        self.seq = 0
        self.combo = None


class _DownloadDialog(QDialog):
    COL_NAME, COL_VERSION, COL_FILE, COL_SIZE, COL_STATUS = range(5)

    def __init__(self, parent, settings, title):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.settings = settings
        self.sources = Sources()
        self.entries = []
        self._inbox = queue.Queue()
        self._cancel = threading.Event()
        self._downloading = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain)
        self._timer.start(50)

        layout = QVBoxLayout(self)
        self.header = QLabel()
        self.header.setWordWrap(True)
        self.header.setVisible(False)
        layout.addWidget(self.header)
        self.top_layout = QVBoxLayout()
        layout.addLayout(self.top_layout)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Package", "Version", "File", "Size", "Status"])
        self.tree.setAlternatingRowColors(True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        for col in range(5):
            header.setSectionResizeMode(col, QHeaderView.Stretch if col == self.COL_FILE else QHeaderView.ResizeToContents)
        self.tree.itemChanged.connect(self._update_summary)
        layout.addWidget(self.tree, 1)

        select_layout = QHBoxLayout()
        for text, state in (("Select All", Qt.Checked), ("Select None", Qt.Unchecked)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, s=state: self._check_all(s))
            select_layout.addWidget(button)
        self.summary_label = QLabel()
        select_layout.addWidget(self.summary_label, 1)
        layout.addLayout(select_layout)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Save to:"))
        default_folder = os.path.normpath(QStandardPaths.writableLocation(QStandardPaths.DownloadLocation))
        self.folder_edit = QLineEdit(str(settings.value("downloads/folder", default_folder)))
        folder_layout.addWidget(self.folder_edit, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        folder_layout.addWidget(browse)
        layout.addLayout(folder_layout)

        self.unpack_check = QCheckBox("Unpack NVIDIA archives into one folder (per platform)")
        self.unpack_check.setToolTip(
            "After downloading, each CUDA or cuDNN archive is extracted and merged into\n"
            "<Save to>\\<platform>, so bin, include and lib end up in one place.\n"
            "cuDNN goes into a subfolder named after its CUDA variant. The archives are kept.")
        self.unpack_check.setChecked(str(settings.value("downloads/unpack", "true")).lower() == "true")
        self.unpack_check.setVisible(False)
        layout.addWidget(self.unpack_check)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.open_button = QPushButton("Open Folder")
        self.open_button.clicked.connect(self._open_folder)
        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self._download_clicked)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        for button in (self.open_button, self.download_button, close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.resize(1050, 560)

    def _post(self, fn):
        self._inbox.put(fn)

    def _drain(self):
        while True:
            try:
                fn = self._inbox.get_nowait()
            except queue.Empty:
                return
            fn()

    def _run(self, work, done):
        def target():
            try:
                result, error = work(), None
            except Exception as e:
                result, error = None, e
            self._post(lambda: done(result, error))
        threading.Thread(target=target, daemon=True).start()

    def add_entry(self, name, versions, resolver, parent=None, checked=False, tooltip="", resolved=None):
        item = QTreeWidgetItem([name, versions[0] if len(versions) == 1 else "", "", "", ""])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(self.COL_NAME, Qt.Checked if checked else Qt.Unchecked)
        if tooltip:
            item.setToolTip(self.COL_NAME, tooltip)
        (parent or self.tree.invisibleRootItem()).addChild(item)
        entry = _Entry(item, versions, resolver)
        if len(versions) > 1:
            entry.combo = QComboBox()
            entry.combo.addItems(versions)
            entry.combo.currentTextChanged.connect(lambda v, e=entry: self._version_changed(e, v))
            self.tree.setItemWidget(item, self.COL_VERSION, entry.combo)
        self.entries.append(entry)
        if resolved is None:
            self.resolve(entry)
        else:
            self._resolved(entry, entry.seq, resolved, None)
        return entry

    def add_group(self, name, checked=False):
        item = QTreeWidgetItem([name, "", "", "", ""])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        item.setCheckState(self.COL_NAME, Qt.Checked if checked else Qt.Unchecked)
        self.tree.addTopLevelItem(item)
        item.setExpanded(True)
        return item

    def clear_entries(self):
        self.entries = []
        self.tree.clear()
        self._update_summary()

    def _version_changed(self, entry, version):
        entry.version = version
        self.resolve(entry)

    def resolve(self, entry):
        entry.seq += 1
        seq, version = entry.seq, entry.version
        entry.artifact = None
        for col in (self.COL_FILE, self.COL_SIZE):
            entry.item.setText(col, "")
        self._set_status(entry, "Looking up…")
        self._run(lambda: entry.resolver(version), lambda result, error: self._resolved(entry, seq, result, error))

    def _resolved(self, entry, seq, artifact, error):
        if seq != entry.seq or self.entries.count(entry) == 0:
            return
        if error is not None:
            entry.item.setCheckState(self.COL_NAME, Qt.Unchecked)
            entry.item.setFlags(entry.item.flags() & ~Qt.ItemIsUserCheckable)
            self._set_status(entry, f"Not available: {_describe(error)}", QColor(190, 40, 40))
            return
        entry.artifact = artifact
        entry.item.setFlags(entry.item.flags() | Qt.ItemIsUserCheckable)
        if entry.combo is None and artifact.version:
            entry.item.setText(self.COL_VERSION, artifact.version)
        entry.item.setText(self.COL_FILE, artifact.filename)
        entry.item.setToolTip(self.COL_FILE, artifact.url)
        entry.item.setText(self.COL_SIZE, human_size(artifact.size))
        if artifact.description and not entry.item.toolTip(self.COL_NAME):
            entry.item.setToolTip(self.COL_NAME, artifact.description)
        self._set_status(entry, artifact.note or "Ready")
        self._update_summary()

    def _set_status(self, entry, text, color=None):
        entry.item.setText(self.COL_STATUS, text)
        steps = entry.artifact.instructions if entry.artifact else ""
        entry.item.setToolTip(self.COL_STATUS, f"{text}\n\nTo install it:\n{steps}" if steps else text)
        entry.item.setForeground(self.COL_STATUS, color or self.palette().text().color())

    def _checked_entries(self):
        return [e for e in self.entries if e.artifact and e.item.checkState(self.COL_NAME) == Qt.Checked]

    def _update_summary(self, *args):
        checked = self._checked_entries()
        total = sum(e.artifact.size or 0 for e in checked)
        unknown = any(not e.artifact.size for e in checked)
        size = human_size(total) + ("+" if unknown and total else "") if total else ("size unknown" if checked else "")
        self.summary_label.setText(f"{len(checked)} selected" + (f", {size}" if size else ""))
        self.unpack_check.setVisible(any(e.artifact and e.artifact.unpack_into for e in self.entries))
        self.download_button.setEnabled(self._downloading or bool(checked))

    def _check_all(self, state):
        for e in self.entries:
            if e.artifact:
                e.item.setCheckState(self.COL_NAME, state)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Save downloads to", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(os.path.normpath(folder))

    def _open_folder(self):
        folder = self.folder_edit.text().strip()
        if os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _download_clicked(self):
        if self._downloading:
            self._cancel.set()
            self.download_button.setEnabled(False)
            self.status_label.setText("Cancelling…")
            return
        jobs = self._checked_entries()
        folder = self.folder_edit.text().strip()
        if not jobs or not folder:
            return
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as error:
            QMessageBox.warning(self, "Download", f"Cannot use {folder}:\n{error}")
            return
        self.settings.setValue("downloads/folder", folder)
        self.settings.setValue("downloads/unpack", "true" if self.unpack_check.isChecked() else "false")
        for e in jobs:
            self._set_status(e, "Queued")
        self._cancel.clear()
        self._set_downloading(True)
        unpack = self.unpack_check.isChecked()
        threading.Thread(target=self._download_all, args=(jobs, folder, unpack), daemon=True).start()

    def _set_downloading(self, active):
        self._downloading = active
        self.download_button.setText("Cancel" if active else "Download")
        self.download_button.setEnabled(True)
        self.tree.setEnabled(not active)
        self.progress.setVisible(active)
        self.progress.setValue(0)

    def _download_all(self, jobs, folder, unpack):
        ok = failed = 0
        for index, entry in enumerate(jobs):
            if self._cancel.is_set():
                self._post(lambda e=entry: self._set_status(e, "Cancelled"))
                continue
            artifact = entry.artifact
            label = f"{index + 1} of {len(jobs)}: {artifact.filename}"
            self._post(lambda e=entry, t=label: (self._set_status(e, "Downloading…"), self.status_label.setText(t)))
            last = [0.0]

            def report(done, total, e=entry, t=label):
                now = time.monotonic()
                if now - last[0] < 0.1 and done != total:
                    return
                last[0] = now
                self._post(lambda: self._show_progress(e, t, done, total))

            try:
                path, status = download_artifact(artifact, folder, self._cancel, report)
                if unpack and artifact.unpack_into:
                    self._post(lambda e=entry: self._set_status(e, "Unpacking…"))
                    target = unpack_archive(path, folder, artifact.unpack_into)
                    status += f", unpacked to {os.path.relpath(target, folder)}"
                ok += 1
                self._post(lambda e=entry, s=status: self._set_status(e, s, QColor(30, 130, 60)))
            except _Cancelled:
                self._post(lambda e=entry: self._set_status(e, "Cancelled"))
            except Exception as error:
                failed += 1
                message = f"Failed: {_describe(error)}"
                self._post(lambda e=entry, m=message: self._set_status(e, m, QColor(190, 40, 40)))
        cancelled = self._cancel.is_set()
        self._post(lambda: self._finished(ok, failed, cancelled, folder))

    def _show_progress(self, entry, label, done, total):
        if total:
            self.progress.setValue(int(done * 1000 / total))
            self._set_status(entry, f"Downloading {done * 100 // total}%")
            self.status_label.setText(f"{label}  ({human_size(done)} of {human_size(total)})")
        else:
            self.status_label.setText(f"{label}  ({human_size(done)})")

    def _finished(self, ok, failed, cancelled, folder):
        self._set_downloading(False)
        parts = [f"{ok} file{'s' if ok != 1 else ''} saved to {folder}"]
        if failed:
            parts.append(f"{failed} failed")
        if cancelled:
            parts.append("cancelled")
        self.status_label.setText("Finished: " + ", ".join(parts) + ".")
        self._update_summary()

    def is_downloading(self):
        return self._downloading

    def shutdown(self):
        self._cancel.set()
        self._downloading = False
        self.close()

    def reject(self):
        if self._downloading:
            answer = QMessageBox.question(self, "Download in progress", "Cancel the download and close this window?")
            if answer != QMessageBox.Yes:
                return
            self._cancel.set()
        self._timer.stop()
        super().reject()


class RowDownloadDialog(_DownloadDialog):
    def __init__(self, parent, settings, spec, focus=None):
        super().__init__(parent, settings, f"Download: {spec['title']}")
        self.spec = spec
        self.header.setText(f"<b>{spec['title']}</b>")
        self.header.setVisible(True)
        self.tree.blockSignals(True)
        platform, python, moniker = spec["platform"], spec["python"], spec["moniker"]
        for item in spec["items"]:
            key = item["key"]
            checked = focus == key or (focus in (None, "wheels") and item.get("wheel", False))
            if key == "cuda":
                group = self.add_group(item["name"], checked)
                for name, version, json_key in item["children"]:
                    self.add_entry(name, [version], lambda v, k=json_key, r=item["release"]: self.sources.nvidia_artifact(
                        "cuda", r, nvidia_platform(platform), k), parent=group, checked=checked, tooltip=json_key)
                continue
            self.add_entry(item["name"], item["versions"], self._resolver(item, platform, python, moniker), checked=checked)
        self.tree.blockSignals(False)
        self._update_summary()

    def _resolver(self, item, platform, python, moniker):
        kind = item["source"]
        if kind == "pytorch":
            return lambda v: self.sources.pytorch_wheel(item["package"], v, moniker, python, platform)
        if kind == "pypi":
            return lambda v: self.sources.pypi_wheel(item["package"], v, python, platform, item.get("prefix", False))
        if kind == "fa2":
            def resolve(v):
                url = item["urls"].get(v)
                if url:
                    artifact = self.sources.github_asset(url, v)
                    artifact.note = item.get("notes", {}).get(v, artifact.note)
                    return artifact
                if platform == "linux":
                    artifact = self.sources.pypi_sdist(
                        "flash-attn", v, "Ready: source package, needs building (hover for the steps)")
                    artifact.instructions = (
                        "pip install packaging psutil ninja wheel\n"
                        f"FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 pip install {artifact.filename} --no-build-isolation")
                    return artifact
                raise LookupError(f"No pre-built Windows wheel for flash-attn {v} with this combination")
            return resolve
        if kind == "cudnn":
            return lambda v: self.sources.cudnn_archive(v, nvidia_platform(platform), item["variant"])
        raise ValueError(kind)


class NvidiaDownloadDialog(_DownloadDialog):
    def __init__(self, parent, settings, product="cuda", release=None, platform=None, variant=None, checked=()):
        super().__init__(parent, settings, "Download CUDA / cuDNN from NVIDIA")
        self.tree.setRootIsDecorated(False)
        self.tree.headerItem().setText(self.COL_NAME, "Component")
        self._preset = {"release": release, "platform": platform, "variant": variant, "checked": set(checked)}
        self._seq = 0
        self._manifest = None

        form = QFormLayout()
        self.product_combo = QComboBox()
        self.product_combo.addItems(list(NVIDIA_PRODUCTS))
        self.release_combo = QComboBox()
        self.platform_combo = QComboBox()
        self.variant_combo = QComboBox()
        for combo in (self.product_combo, self.release_combo, self.platform_combo, self.variant_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.variant_label = QLabel("CUDA variant:")
        row = QHBoxLayout()
        for label, combo in (("Product:", self.product_combo), ("Release:", self.release_combo),
                             ("Platform:", self.platform_combo)):
            row.addWidget(QLabel(label))
            row.addWidget(combo)
            row.addSpacing(12)
        row.addWidget(self.variant_label)
        row.addWidget(self.variant_combo)
        row.addStretch()
        form.addRow(row)
        self.top_layout.addLayout(form)

        self.product_combo.setCurrentText(next(k for k, v in NVIDIA_PRODUCTS.items() if v == product))
        self.product_combo.currentTextChanged.connect(self._load_releases)
        self.release_combo.currentTextChanged.connect(self._load_manifest)
        self.platform_combo.currentTextChanged.connect(self._platform_changed)
        self.variant_combo.currentTextChanged.connect(self._populate)
        self._load_releases()

    def _product(self):
        return NVIDIA_PRODUCTS[self.product_combo.currentText()]

    def _set_loading(self, text):
        self.clear_entries()
        self.status_label.setText(text)
        for combo in (self.product_combo, self.release_combo, self.platform_combo, self.variant_combo):
            combo.setEnabled(False)

    def _ready(self):
        for combo in (self.product_combo, self.release_combo, self.platform_combo, self.variant_combo):
            combo.setEnabled(True)

    def _set_downloading(self, active):
        super()._set_downloading(active)
        for combo in (self.product_combo, self.release_combo, self.platform_combo, self.variant_combo):
            combo.setEnabled(not active)

    def _latest(self, callback):
        self._seq += 1
        seq = self._seq
        return lambda result, error: callback(result, error) if seq == self._seq else None

    def _fill(self, combo, values, preferred):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(values)
        for value in preferred:
            if value in values:
                combo.setCurrentText(value)
                break
        combo.blockSignals(False)

    def _load_releases(self):
        product = self._product()
        self._set_loading(f"Reading NVIDIA's {self.product_combo.currentText()} release list…")
        self._run(lambda: self.sources.nvidia_releases(product), self._latest(self._releases_loaded))

    def _releases_loaded(self, releases, error):
        self._ready()
        if error is not None:
            self.status_label.setText(f"Could not read NVIDIA's release list: {_describe(error)}")
            return
        preset = self._preset.pop("release", None)
        self._fill(self.release_combo, releases, [preset] if preset else [])
        self._load_manifest()

    def _load_manifest(self):
        product, release = self._product(), self.release_combo.currentText()
        if not release:
            return
        self._manifest = None
        self._set_loading(f"Reading the {release} manifest…")
        self._run(lambda: self.sources.nvidia_manifest(product, release), self._latest(self._manifest_loaded))

    def _manifest_loaded(self, manifest, error):
        self._ready()
        if error is not None:
            self.status_label.setText(f"Could not read the manifest: {_describe(error)}")
            return
        self._manifest = manifest
        preferred = [self._preset.pop("platform", None) or self.platform_combo.currentText(), "windows-x86_64", "linux-x86_64"]
        self._fill(self.platform_combo, manifest_platforms(manifest), [p for p in preferred if p])
        self._platform_changed()

    def _platform_changed(self):
        platform = self.platform_combo.currentText()
        if not platform or self._manifest is None:
            return
        variants = manifest_variants(self._manifest, platform) if self._product() == "cudnn" else []
        preferred = [self._preset.pop("variant", None) or self.variant_combo.currentText()]
        self._fill(self.variant_combo, variants, [p for p in preferred if p])
        self.variant_combo.setVisible(bool(variants))
        self.variant_label.setVisible(bool(variants))
        self._populate()

    def _populate(self):
        if self._manifest is None:
            return
        platform, variant = self.platform_combo.currentText(), self.variant_combo.currentText() or None
        checked = self._preset.pop("checked", set())
        self.clear_entries()
        self.status_label.setText("")
        self.tree.blockSignals(True)
        artifacts = manifest_artifacts(self._manifest, self._product(), platform, variant)
        for key, artifact in sorted(artifacts.items(), key=lambda kv: kv[1].description.lower()):
            entry = self.add_entry(artifact.description or key, [artifact.version], lambda v, a=artifact: a,
                                   checked=key in checked, tooltip=key, resolved=artifact)
            entry.item.setData(self.COL_NAME, Qt.UserRole, key)
        self.tree.blockSignals(False)
        self._update_summary()
