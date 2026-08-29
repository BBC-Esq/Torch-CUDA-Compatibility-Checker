r"""
****************************
GROUND TRUTH SOURCES
****************************
# This section lists, per library, the EXACT files and URL patterns used to extract
# compatibility data. When a new release comes out, fetch these files at the new tag
# and update accordingly. ALWAYS fetch raw files with curl/wget (no AI summarization)
# so the data goes straight from the source into the program.
#
# ---- PyTorch ----
#   Latest version (PyPI):
#     curl -s https://pypi.org/pypi/torch/json | jq -r .info.version
#   Per-release build matrix (CUDA archs, cuDNN pins, Python versions):
#     https://raw.githubusercontent.com/pytorch/pytorch/v{VER}/.github/scripts/generate_binary_build_matrix.py
#     Look for: CUDA_ARCHES, CUDA_STABLE, CUDA_ARCHES_FULL_VERSION,
#               PYTORCH_EXTRA_INSTALL_REQUIREMENTS (cuDNN pins inside),
#               FULL_PYTHON_VERSIONS
#     ALSO REQUIRED: the `elif os == "windows":` branch, which drops arches from the
#     Windows build. See P1/P2/P3 in the pitfalls section — three separate mistakes
#     have been made in this one file. Then HEAD-check the wheel index.
#   Per-wheel Windows/Linux/Python availability (authoritative over the build script):
#     https://download.pytorch.org/whl/{MONIKER}/torch/      (403 = wheel absent)
#   Triton pin for that torch release:
#     https://raw.githubusercontent.com/pytorch/pytorch/v{VER}/.ci/docker/triton_version.txt
#   Sympy: setup.py is what users actually get (install_requires, e.g. sympy>=1.13.3);
#   requirements-ci.txt is the CI-pinned exact version (sympy==1.13.3). The program
#   stores the setup.py spec because it feeds a `pip install` command.
#     https://raw.githubusercontent.com/pytorch/pytorch/v{VER}/setup.py
#     https://raw.githubusercontent.com/pytorch/pytorch/v{VER}/.ci/docker/requirements-ci.txt
#
# ---- Torchvision / Torchaudio ----
#   PyPI JSON:
#     https://pypi.org/pypi/torchvision/json
#     https://pypi.org/pypi/torchaudio/json
#   NOTE: torchaudio entered maintenance mode after 2.11.0. There may be no
#         matching torchaudio release for newer torch versions.
#
# ---- CUDA metapackages ----
#   Per-version redistribution JSON:
#     https://developer.download.nvidia.com/compute/cuda/redist/redistrib_{X.Y.Z}.json
#   ENUMERATE the index — do not probe candidate URLs (see P17):
#     curl -s https://developer.download.nvidia.com/compute/cuda/redist/ \
#       | grep -o 'redistrib_[0-9]\+\.[0-9]\+\.[0-9]\+\.json' | sort -u
#     One request returns every release that exists. Diff that against
#     cuda_metapackages.keys() to find BOTH new releases and historical holes.
#   Components to extract (each has a .version field):
#     cuda_nvrtc, cuda_cudart, cuda_nvcc, cuda_cupti, libcublas, libcufft,
#     libcurand, libcusolver, libcusparse, cuda_nvtx, libnvjitlink
#   Not every release has all 11 — 11.8.0 has no libnvjitlink (nvJitLink is CUDA 12.0+).
#   Absent components are simply omitted from that version's dict and render as "-".
#
# ---- cuDNN ----
#   Support matrix (Linux/Windows, CUDA compat, driver minimums):
#     https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
#     HTML-only. curl it and grep the raw markup for the literal values; the page is
#     partly JS-rendered but the table text is present in the served HTML.
#   Latest cuDNN release (PyPI metapackage):
#     https://pypi.org/pypi/nvidia-cudnn-cu12/json
#     https://pypi.org/pypi/nvidia-cudnn-cu13/json
#     MUST filter yanked files and prereleases — see P4 (9.25.0.15 is fully yanked).
#   CAVEAT: nvidia-cudnn-cu13 does publish a win_amd64 wheel (since 9.12.0.46, Aug 2025),
#   which superficially contradicts "CUDA 13.x cuDNN is Linux-only". NVIDIA's support
#   matrix still lists Windows driver support as N/A for the CUDA 13.x build, so the
#   program's claim is correct as stated. Expect users to ask about this.
#
# ---- Triton (Linux, upstream) ----
#   Repo: https://github.com/triton-lang/triton
#   PyPI versions:
#     https://pypi.org/pypi/triton/json
#   Per-torch pin: same file as PyTorch's triton_version.txt (above).
#   Linux Triton is just `pip install triton==X.Y.Z` — no `-windows.postN` suffix.
#
# ---- Triton (Windows, fork) ----
#   Repo: https://github.com/triton-lang/triton-windows (v3.6.0-windows.post26+)
#         https://github.com/woct0rdho/triton-windows (up to v3.6.0-windows.post25)
#   Compatibility table (torch ↔ triton, bundled CUDA tools):
#     https://raw.githubusercontent.com/triton-lang/triton-windows/readme/README.md
#   PyPI versions:
#     https://pypi.org/pypi/triton-windows/json
#   Bundled CUDA tool versions in each wheel:
#     cmake/nvidia-toolchain-version.json on the release/{X.Y}.x-windows branch
#
# ---- Flash Attention 2 (Linux) ----
#   Repo: https://github.com/Dao-AILab/flash-attention
#   CI build matrix (torch versions, python versions, CUDA version):
#     https://raw.githubusercontent.com/Dao-AILab/flash-attention/v{VER}/.github/workflows/publish.yml
#     Look in `build_wheels.strategy.matrix`: python-version, torch-version, cuda-version
#   Release assets — AUTHORITATIVE over the CI matrix (see P7). Enumerate real names:
#     curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v{VER} \
#       | jq -r '.assets[].name'
#     Parse each as: flash_attn-{VER}+cu{CU}torch{T_MM}cxx11abi{ABI}-cp{PY}-cp{PY}-linux_{ARCH}.whl
#     Watch for P6 (filename version != tag) and P8 (newer release, less coverage).
#   PyPI:
#     https://pypi.org/pypi/flash-attn/json
#   NOTE: FA2 has no hard torch pin (setup.py: install_requires=["torch"]).
#         The CI matrix defines what was built/tested.
#   NOTE: upstream now publishes FA4 betas (fa4-v4.0.0.betaNN) on a weekly cadence while
#         FA2 sits idle. FA2 2.x may stop receiving releases; betas are not tracked.
#
# ---- Flash Attention 2 (Windows) ----
#   Repo: https://github.com/kingbri1/flash-attention
#   Build workflow (workflow_dispatch — manually triggered):
#     https://raw.githubusercontent.com/kingbri1/flash-attention/main/.github/workflows/build-wheels.yml
#   Actual wheel availability comes from release assets, not the matrix:
#     https://github.com/kingbri1/flash-attention/releases
#
# ---- Xformers ----
#   Repo: https://github.com/facebookresearch/xformers
#   Per-release torch version + CUDA build matrix:
#     https://raw.githubusercontent.com/facebookresearch/xformers/v{VER}/.github/workflows/wheels.yml
#     Look for: torch_version, CU_VERSIONS, os (ubuntu vs windows)
#   FA2 version range supported:
#     https://raw.githubusercontent.com/facebookresearch/xformers/v{VER}/xformers/ops/fmha/flash.py
#     Look for: FLASH_VER_MIN, FLASH_VER_LAST
#   Build toolkit CUDA versions (may differ from torch CUDA monikers):
#     https://raw.githubusercontent.com/facebookresearch/xformers/v{VER}/.github/actions/setup-build-cuda/action.yml
#   PyPI metadata (for torch pin in published wheel — may differ from pyproject.toml):
#     https://pypi.org/pypi/xformers/json
#
# ---- Bitsandbytes ----
#   Repo: https://github.com/bitsandbytes-foundation/bitsandbytes
#   CI build matrix (CUDA versions, Linux + Windows + ARM):
#     https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/{VER}/.github/workflows/python-package.yml
#     Look in `build-cuda` job for: cuda_version matrix
#   Python support (wheels are py3 / version-agnostic):
#     https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/{VER}/pyproject.toml
#     Look for: requires-python, classifiers (Programming Language :: Python :: ...)
#   PyPI:
#     https://pypi.org/pypi/bitsandbytes/json
#
# ---- General update procedure ----
#   1. For each library, hit the "latest version" endpoint (PyPI JSON or releases page).
#   2. If a newer version exists than what we have, fetch its tagged ground truth file.
#   3. Parse the values (curl + python/jq, NOT WebFetch).
#   4. Update test_compatibility.py (data structures) and this file.
#   5. Verify by re-fetching the same file and re-parsing.
#   6. Run the EMPIRICAL CHECKS in the next section. A build script says what CI
#      *intended* to build; only the wheel index proves what actually shipped.



*********************************************************
PITFALLS AND INSTITUTIONAL KNOWLEDGE  (READ BEFORE UPDATING)
*********************************************************
# Every item below is a mistake that was actually made on this project, by either the
# maintainer, an AI assistant, or an outside reviewer. They are subtle, they recur, and
# several of them produced WRONG compatibility data that shipped. Read this list before
# trusting any parse.
#
# The cardinal rule: THIS PROGRAM'S PURPOSE IS TO NOT LIE TO USERS. A false positive
# ("this combination works" when no wheel exists) is the worst possible defect. When a
# build script and the actual published wheels disagree, THE PUBLISHED WHEELS WIN.
#
#
# --- P1. A build matrix constant is NOT proof a wheel shipped. Verify empirically. ---
#   PyTorch's generate_binary_build_matrix.py lists CUDA_ARCHES, but individual arches
#   are excluded per-OS further down the file. Always confirm with a real HTTP request:
#
#     curl -s -o /dev/null -w "%{http_code}" -I \
#       "https://download.pytorch.org/whl/cu129/torch-2.13.0%2Bcu129-cp312-cp312-win_amd64.whl"
#
#   IMPORTANT: download.pytorch.org is S3-backed and returns **403** (not 404) for a
#   nonexistent wheel. Treat 403 as "does not exist". 200 means it shipped.
#   Note the "%2B" — the "+" in the local version must be URL-encoded.
#
#
# --- P2. The Windows arch-exclusion idiom CHANGED between releases. ---
#   This one bit two independent reviewers. Searching for only one idiom gives a false
#   "no exclusion found" and produces bogus windows=True data.
#     torch 2.9.1 - 2.11.0 use:  windows_cuda_arches = CUDA_ARCHES.copy()
#                                windows_cuda_arches.remove("12.9")
#     torch 2.12.1 - 2.13.0 use: windows_cuda_arches = list_without(CUDA_ARCHES, ["12.9"])
#     torch 2.14.0+      use:    CUDA_ARCHES_NO_WINDOWS: list[str] = []      (module level)
#                                arches += list_without(CUDA_ARCHES, CUDA_ARCHES_NO_WINDOWS)
#   The 2.14 form is the dangerous one: it moved the excluded versions OFF the exclusion
#   site and into a named module-level constant. A grep for the 2.12 literal
#   `list_without(CUDA_ARCHES, ["` returns nothing on 2.14 and reads as "no exclusion."
#   For 2.14 that conclusion is accidentally correct (the constant is empty, so all of
#   12.6/13.0/13.2 build for Windows) — but it is correct by luck. Add one version to
#   that constant and the same grep silently produces bogus windows=True data.
#   ALWAYS resolve the CONSTANT'S VALUE, not just the shape of the exclusion call.
#   Grep for ALL THREE, plus any future variant, by searching the `elif os == "windows":`
#   branch directly rather than for a specific helper name:
#     grep -A 8 'elif os == "windows"' generate_binary_build_matrix.py
#   then grep the constant that branch names, e.g.:
#     grep -n 'CUDA_ARCHES_NO_WINDOWS' generate_binary_build_matrix.py
#   Then still do P1.
#
#   Known history of the cu129 Windows wheel (verified by HEAD request):
#     2.8.0  -> Windows cu129 EXISTS (200)
#     2.9.0  -> Windows cu129 EXISTS (200)   [see P7 - not even in that tag's matrix]
#     2.9.1  -> ABSENT (403)   first release to exclude 12.9 from Windows
#     2.10.0 -> ABSENT (403)
#     2.11.0 -> ABSENT (403)
#     2.12.0 -> n/a (12.9 not in CUDA_ARCHES at all)
#     2.12.1 -> ABSENT (403)
#     2.13.0 -> ABSENT (403)
#     2.14.0 -> n/a (12.9 not in CUDA_ARCHES at all; RC-only as of 2026-08-29)
#
#
# --- P3. cuDNN pins are per-CUDA-arch and live in a DIFFERENT dict. ---
#   Do not regex forward from a CUDA_ARCHES_FULL_VERSION key; you will capture the wrong
#   block and report one cuDNN version for every arch (this happened). The pins are inside
#   PYTORCH_EXTRA_INSTALL_REQUIREMENTS, keyed by arch, and DIFFER between arches:
#     torch 2.13.0:  12.6 -> nvidia-cudnn-cu12==9.10.2.21
#                    12.9 -> nvidia-cudnn-cu12==9.20.0.48   <-- different from 12.6
#                    13.0 -> nvidia-cudnn-cu13==9.20.0.48
#                    13.2 -> nvidia-cudnn-cu13==9.20.0.48
#   Note the package NAME also changes (cu12 vs cu13). Parse each arch's block separately.
#
#
# --- P4. PyPI "latest" can be YANKED or a prerelease. Filter both. ---
#   info.version is not always safe. Two real cases:
#     - nvidia-cudnn-cu12/cu13 9.25.0.15 exists but EVERY file is yanked. When that was
#       the newest version the correct "latest usable" was 9.24.0.43; as of 2026-08-29
#       it is 9.25.1.1, which superseded the yanked build. The yank is permanent — a
#       yanked version never becomes valid again, so keep filtering it.
#     - xformers publishes many 0.0.35.devNNNN builds; flash-attention publishes
#       fa4-v4.0.0.betaNN prereleases. Neither is a stable release.
#   Filter with: skip if Version(v).is_prerelease, and skip if all(f["yanked"] for f in files).
#
#
# --- P5. Prose/README summary tables go stale. Prefer machine-readable ground truth. ---
#   The triton-windows README says "3.3 .. 3.7 -> CUDA 12.8", and this file once repeated
#   that. It is WRONG for 3.7.x. The authoritative source is
#   cmake/nvidia-toolchain-version.json on the release branch, which shows 3.7.x ships a
#   CUDA 13.1 runtime. When a repo offers both a hand-written table and a JSON/YAML file,
#   the JSON/YAML is ground truth and the table is a convenience that may lag.
#
#
# --- P6. An asset's FILENAME version may differ from its RELEASE TAG. ---
#   flash-attention v2.8.3.post1 ships 50 assets: 48 are named "flash_attn-2.8.3.post1+...",
#   but the two cu13 assets are named "flash_attn-2.8.3+cu13torch2.9...". Any URL builder
#   that interpolates the release version into the filename will 404 on those. This is why
#   flash_attention_linux entries support an optional "wheel_ver" override (see SCHEMA below).
#   ALWAYS enumerate real asset names; never assume the tag is in the filename:
#     curl -s https://api.github.com/repos/OWNER/REPO/releases/tags/TAG | jq -r '.assets[].name'
#
#
# --- P7. A tagged CI matrix and the published release assets can disagree. ---
#   Wheels get added after a CI run (manual upload, re-run, or a later branch build).
#   Known instances:
#     - flash-attention v2.8.3's publish.yml matrix stops at torch 2.8.0, yet the release
#       carries torch 2.9 and torch 2.10 wheels.
#     - torch 2.9.0's CUDA_ARCHES has no "12.9", yet torch-2.9.0+cu129 wheels exist for
#       BOTH Linux and Windows. RESOLVED — see P16 for the mechanism and the outcome.
#   Policy: use the CI matrix to understand intent; use the release assets / wheel index
#   to decide what the program claims is installable. When they disagree and the wheel
#   verifiably exists, the program SHOWS the combination and marks it with the
#   out_of_matrix flag (rendered as "†") rather than hiding it. Hiding a working
#   combination is a false negative, which is still the program being wrong.
#
#
# --- P16. The TAG and the RELEASE BRANCH are different files. Check both. ---
#   This is the mechanism behind the torch 2.9.0/cu129 puzzle, and it will recur.
#   A release branch keeps moving after its tag is cut, and wheels can be published
#   from the newer branch state under the older version number:
#     v2.9.0 (tag)        CUDA_ARCHES = ["12.6","12.8","13.0"]        <- no 12.9 at all
#     release/2.9 (branch) CUDA_ARCHES = ["12.6","12.8","12.9","13.0"] + cuDNN
#                          nvidia-cudnn-cu12==9.10.2.21, and a Windows 12.9 exclusion
#   Reconstructed timeline:
#     1. v2.9.0 tagged — 12.9 absent entirely
#     2. 12.9 added to release/2.9 for BOTH OSes -> 2.9.0+cu129 wheels published for
#        Linux AND Windows (all of cp310-cp314, incl. cp313t/cp314t)
#     3. The "Only build CUDA 12.9 for Linux" exclusion lands on release/2.9
#     4. v2.9.1 cut from that later state -> cu129 becomes Linux-only from 2.9.1 on
#   So the tag under-reports, and the CURRENT branch over-reports for the older tag
#   (it shows the Windows exclusion, which did not apply when 2.9.0's wheels were built).
#   Neither file alone is sufficient. The wheel index is the tiebreaker.
#     https://raw.githubusercontent.com/pytorch/pytorch/release/{X.Y}/.github/scripts/generate_binary_build_matrix.py
#   Use the branch to recover values the tag lacks (e.g. the cu129 cuDNN pin above),
#   then confirm each wheel with a HEAD request per P1.
#
#
# --- P8. A NEWER release can have LESS coverage than an older one. ---
#   flash-attention 2.8.3.post1 DROPPED the cu12torch2.9 and cu13torch2.10 wheels that
#   v2.8.3 has. "Latest" is not a superset. Diff the asset sets between adjacent releases
#   rather than assuming forward progress, and keep the older release in the data.
#
#
# --- P9. PEP 440: "==X.Y.Z" does NOT match "X.Y.Z.postN". ---
#   Every triton-windows release is a .postN (e.g. 3.7.1.post27); no bare version exists.
#   `pip install triton-windows==3.7.1` fails with "No matching distribution found".
#   The generated command must use prefix matching: triton-windows==3.7.1.*
#   Verify any spec you emit:
#     python -c "from packaging.specifiers import SpecifierSet; from packaging.version import Version; \
#       print(Version('3.7.1.post27') in SpecifierSet('==3.7.1.*'))"
#
#
# --- P10. Published wheel METADATA can differ from the repo's pyproject.toml. ---
#   xformers v0.0.34's pyproject.toml says torch>=2.10, but the wheel published to PyPI
#   pins torch==2.10.0 exactly (the build process rewrites it). Only v0.0.35+ truly allows
#   torch>=2.10. For dependency pins, trust the PyPI JSON's requires_dist over the repo.
#
#
# --- P11. Linux C++11 ABI depends on the TORCH version, not the FA2 version. ---
#   PyTorch's Linux wheels moved from manylinux1 (old ABI) to manylinux_2_28 (new ABI) at
#   torch 2.7. So FA2 Linux wheels need cxx11abiTRUE for torch >= 2.7 and cxx11abiFALSE
#   for torch 2.6.x. Picking wrong either 404s or imports and dies with undefined symbols.
#   Check a torch wheel's platform tag to confirm which ABI a torch release uses.
#
#
# --- P12. Some packages ship PER-CUDA builds on the PyTorch index, not just PyPI. ---
#   PyPI has only ONE CUDA variant of xformers. The per-CUDA builds live at
#   download.pytorch.org/whl/cuXXX/xformers/. A bare `pip install xformers==X` can install
#   a build compiled against a different CUDA than the row advertises, so generated
#   commands must pass --index-url for the row's moniker.
#
#
# --- P13. Version strings must be sorted as VERSIONS, never as strings. ---
#   Lexical sort puts "3.9" above "3.14" and "2.9.1" above "2.12.0". The UI uses a
#   version-aware sort helper. It must also tolerate non-numeric segments such as
#   "2.8.3.post1" - a plain int() cast on each dotted segment will raise ValueError and
#   crash the app at startup.
#
#
# --- P14. Python version support can be PLATFORM-SPECIFIC. ---
#   torch 2.13.0 builds cp315 (Python 3.15) wheels for Linux ONLY; Windows stops at cp314.
#   PyPI itself ships only cp310-cp314. So FULL_PYTHON_VERSIONS in the build matrix is a
#   superset of what any single platform gets. Verify per-index and per-platform:
#     curl -s https://download.pytorch.org/whl/cu130/torch/   # then filter by platform tag
#   Free-threaded builds (cp313t/cp314t/cp315t) also appear in FULL_PYTHON_VERSIONS as
#   "3.14t" etc. The program does NOT currently model free-threaded variants.
#
#
# --- P15. Never use an AI-summarizing fetch for ground truth. ---
#   Use curl + python/jq and read raw bytes. A summarization layer between NVIDIA's JSON
#   and the data file can silently alter version numbers. (For HTML-only sources like the
#   cuDNN support matrix, curl the page and grep the raw markup for the exact strings.)
#
#
# --- P17. ENUMERATE upstream versions and DIFF. Never probe forward from the newest. ---
#   For a long time CUDA versions were found by probing candidate URLs above the newest
#   known version. That structurally CANNOT find a hole below it, and it left five:
#   12.6.0, 12.6.1, 12.6.2, 12.9.0, 13.0.1 were all absent while 13.3.1 was present.
#   The user noticed 12.9.0 missing from the CUDA dropdown; nothing in the process would
#   ever have caught it. This file previously ENDORSED the bad method, which is why it
#   persisted — the guidance itself was the bug.
#   Correct method for every library: get the full upstream list in one shot, filter to
#   >= the floor in VERSION SCOPE POLICY, and set-difference against what is tracked.
#     PyPI:   curl -s https://pypi.org/pypi/{pkg}/json  -> .releases keys
#             (drop prereleases and fully-yanked versions — see P4)
#     GitHub: curl -s https://api.github.com/repos/{owner}/{repo}/releases?per_page=100
#     NVIDIA: enumerate the redist index (see the CUDA entry above)
#   Audited this way on 2026-08-29: torch, torchvision, torchaudio, triton, xformers,
#   bitsandbytes and flash-attn were all already COMPLETE from their floors; only CUDA
#   had holes. Re-run the diff every time rather than trusting that result.
#
#
# --- P18. A row can be present in the data yet UNREACHABLE in the UI. ---
#   torch_cuda and torch_python_triton must agree, because a torch_cuda row only ever
#   displays if its CUDA major.minor appears in that torch version's cuda_versions list.
#   Three cu118 rows (torch 2.6.0 / 2.7.0 / 2.7.1) sat in torch_cuda for months and never
#   rendered, because no cuda_versions list contained "11.8" — even though upstream
#   CUDA_ARCHES for all three DOES include 11.8 and the wheels exist (HTTP 200, Windows
#   and Linux, cuDNN pin nvidia-cudnn-cu11==9.1.0.70). Fifteen valid combinations were
#   invisible. Consistency check to run after ANY edit to either structure:
#     for each row in torch_cuda:
#         assert major_minor(row.cuda) in cuda_versions[row.torch]
#     for each cv in cuda_versions[torch]:
#         assert some torch_cuda row for that torch has major_minor == cv
#   Both directions matter: the first catches invisible rows, the second catches a
#   cuda_versions entry with no wheel behind it.
#
#
# --- P19. cuDNN package NAME tracks the CUDA major: cu11 / cu12 / cu13. ---
#   P3 covers per-arch pins differing within one release. Note there are THREE package
#   names, not two: torch 2.6.0/2.7.x cu118 pin nvidia-cudnn-cu11==9.1.0.70. A regex
#   looking only for nvidia-cudnn-cu12/cu13 silently misses the cu118 arch.
#
#
# --- P20. A CUDA version in a CI matrix is not proof of a RELEASED toolkit. ---
#   Grepping a workflow for cuda_version and adding whatever turns up will eventually
#   pull in a prerelease. bitsandbytes 0.50.1 added:
#       include:
#         - os: windows-11-vs2026-arm
#           cuda_version: "13.4.0"
#   13.4.0 has no redistrib JSON (404) and is not in the redistribution index, because
#   install-cuda-woa.ps1 fetches it from packages.nvidia.com/PRERELEASE/cuda/13.4.0/
#   for a Windows-on-ARM build. There is nothing to source component versions from.
#   Rules that follow from this:
#     - The redistribution index (P17) is the authority on which CUDA versions EXIST.
#       If a version is not enumerable there, it does not go in cuda_metapackages,
#       no matter what a CI file says.
#     - Read the matrix STRUCTURE, not just the values. A bare list is the tracked
#       matrix; an `include:` block bolts on extra one-off combinations that may
#       target a different OS/arch entirely.
#     - Check which runner/arch a value belongs to. This program's bitsandbytes CUDA
#       column tracks the Windows x64 matrix; the 13.4.0 entry is win_arm64 only.
#   Same discipline as P4 (filter prereleases on PyPI) applied to CUDA itself.



*********************************************************
PROGRAM DATA SCHEMA NOTES  (test_compatibility.py)
*********************************************************
# Non-obvious fields, so a future maintainer does not have to reverse-engineer them.
#
# torch_cuda[] :
#   windows        True  = a Windows wheel exists AND has cuDNN.
#                  False = not usable on Windows; the REASON is in no_win_reason.
#   no_win_reason  "cudnn"   = Windows wheel IS built, but no Windows cuDNN exists for
#                              this CUDA major (cuDNN 9.x for CUDA 13.x is Linux-only).
#                  "nowheel" = NO Windows wheel is built at all (cu129 from torch 2.9.1 on).
#                  These are different failure modes and must not be labeled identically;
#                  telling a Windows user "no cuDNN" when the wheel does not exist is wrong.
#   out_of_matrix  OPTIONAL, True = the wheel is confirmed installable but does NOT appear
#                  in that release's TAGGED build matrix (see P7/P16). Rendered as a "†"
#                  on the "CUDA (torch-tested)" cell, purple, with an explanatory tooltip,
#                  and listed in the legend. Currently set on exactly one row:
#                  torch 2.9.0 / cu129.
#                  DO NOT delete such a row because the tagged matrix omits it — that has
#                  already happened once. Re-verify with a HEAD request instead.
#                  The "†" is presentation only: the export's metapackage column filter
#                  and the install-command builder both strip it before matching.
#
# torch_python_triton[] :
#   cuda_versions      major.minor strings, matched against torch_cuda's full versions.
#   triton_compat      all triton minors known-compatible (README-level), vs "triton"
#                      which is torch's exact hard pin.
#   python_linux_only  Python versions present ONLY in Linux wheels (see P14). Filtered
#                      out when the platform toggle is set to Windows.
#
# flash_attention_linux[] :
#   cuda       CUDA MAJOR ("12"/"13") of the torch wheel this FA2 wheel pairs with.
#   wheel_ver  OPTIONAL. The version string as it appears in the ASSET FILENAME, when it
#              differs from the release version (see P6). Defaults to the "fa2" value.
#
# cuda_metapackages{} :
#   Not every version carries all 11 components (11.8.0 has no nvjitlink). The "Any"
#   metapackage view therefore builds its row list from the UNION of all versions' keys,
#   NOT from the first entry — it previously used next(iter(...)) which would have
#   dropped the nvjitlink row entirely once 11.8.0 became the first key. Missing
#   components render as "-". Column order uses the version-aware sort, not string sort.



****************************
Torch and CUDA Compatibility
****************************

# Torch wheels and the CUDA/cuDNN versions they were tested with.
# NOTE: This shows which cuDNN version torch pins for each wheel — it does NOT mean
# that a particular CUDA version is only compatible with that specific cuDNN version.
# See the cuDNN & CUDA section below for actual CUDA/cuDNN/platform compatibility.
+--------+---------+--------+------------+
| Torch  | Moniker | CUDA   | cuDNN      |
+--------+---------+--------+------------+
|        | cu132   | 13.2.1 | 9.20.0.48  |
| 2.13.0 | cu130   | 13.0.3 | 9.20.0.48  |
|        | cu129   | 12.9.1 | 9.20.0.48  | <-- Linux-only: no Windows wheel built
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu132   | 13.2.1 | 9.20.0.48  |
| 2.12.1 | cu130   | 13.0.2 | 9.20.0.48  |
|        | cu129   | 12.9.1 | 9.20.0.48  | <-- Linux-only: no Windows wheel built
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu132   | 13.2.1 | 9.20.0.48  |
| 2.12.0 | cu130   | 13.0.2 | 9.20.0.48  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu130   | 13.0.2 | 9.19.0.56  |
| 2.11.0 | cu129   | 12.9.1 | 9.17.1.4   |
|        | cu128   | 12.8.1 | 9.19.0.56  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu130   | 13.0.0 | 9.15.1.9   |
| 2.10.0 | cu129   | 12.9.1 | 9.10.2.21  |
|        | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu130   | 13.0.0 | 9.13.0.50  |
| 2.9.1  | cu129   | 12.9.1 | 9.10.2.21  |
|        | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu130   | 13.0.0 | 9.13.0.50  |
| 2.9.0  | cu129   | 12.9.1 | 9.10.2.21  | <-- † not in the v2.9.0 tagged matrix (P16)
|        | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu129   | 12.9.1 | 9.10.2.21  |
| 2.8.0  | cu128   | 12.8.1 | 9.10.2.21  |
|        | cu126   | 12.6.3 | 9.10.2.21  |
+--------+---------+--------+------------+
|        | cu128   | 12.8.0 | 9.7.1.26   |
| 2.7.1  | cu126   | 12.6.3 | 9.5.1.17   |
|        | cu118   | 11.8.0 | 9.1.0.70   | <-- nvidia-cudnn-cu11 (P19)
+--------+---------+--------+------------+
|        | cu128   | 12.8.0 | 9.7.1.26   |
| 2.7.0  | cu126   | 12.6.3 | 9.5.1.17   |
|        | cu118   | 11.8.0 | 9.1.0.70   | <-- nvidia-cudnn-cu11 (P19)
+--------+---------+--------+------------+
|        | cu126   | 12.6.3 | 9.5.1.17   |
| 2.6.0  | cu124   | 12.4.1 | 9.1.0.70   |
|        | cu118   | 11.8.0 | 9.1.0.70   | <-- nvidia-cudnn-cu11 (P19)
+--------+---------+--------+------------+
* Obtained from: https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py
  (check the tagged release, e.g. v2.11.0, for each torch version)
* The cu118 rows were unreachable in the UI until 2026-08-03 — see P18. All three have
  Windows AND Linux wheels (verified HTTP 200) and are NOT excluded from the Windows build.
* CUDA 11.8.0 is the scope floor. torch 2.6.0/2.7.x also build for older CUDA in some
  channels; anything below 11.8 is deliberately out of scope (see VERSION SCOPE POLICY).

# GPU architecture notes (torch 2.11.0+):
#   - Volta (SM 7.0, e.g. V100) support was removed from cu128 and cu129 builds
#     starting with torch 2.11.0 due to the update to cuDNN 9.15.1+, which is
#     incompatible with Volta. Volta users should use cu126, which retains support.
#   - CUDA 13.0 (cu130) only supports Turing (SM 7.5) and newer on Linux x86_64.
#     Maxwell and Pascal GPUs are not supported under CUDA 13.0.
#   - PyPI default: starting with torch 2.11.0, `pip install torch` installs the
#     cu130 wheel by default (Linux x86_64 and aarch64). Users with CUDA 12.x-only
#     drivers must specify an index URL for cu126/cu128/cu129.
#   Source: https://github.com/pytorch/pytorch/releases/tag/v2.11.0
#
# Torch 2.12.0 changes:
#   - cu128 and cu129 wheels were DROPPED. New wheel matrix is cu126, cu130, cu132.
#   - cu132 (CUDA 13.2.1) is new. Linux-only (cuDNN 9.x for CUDA 13 = Linux only).
#   - PyPI default remains cu130 (CUDA_STABLE = "13.0" in the build matrix).
#   - Python 3.13 free-threaded (3.13t) and 3.14 free-threaded (3.14t) wheels added.
#   - torchaudio entered maintenance mode after 2.11.0 — no 2.12.0 release of
#     torchaudio exists. torchvision still pairs (0.27.0 ↔ torch 2.12.0).
#   Source: https://github.com/pytorch/pytorch/releases/tag/v2.12.0
#
# Torch 2.12.1 changes (patch release; NOT a copy of the 2.12.0 row):
#   - cu129 RESTORED to the matrix, but Linux-only (no Windows wheel — see below).
#     Wheel matrix: cu126, cu129, cu130, cu132.
#   - Triton pin bumped 3.7.0 -> 3.7.1.
#   - Python 3.13 free-threaded (3.13t) dropped; list is 3.10-3.14 + 3.14t.
#   - torchvision 0.27.1 pins torch==2.12.1.
#
# Torch 2.13.0 changes:
#   - Wheel matrix: cu126, cu129 (Linux-only), cu130, cu132.
#   - cu130 moved to CUDA 13.0.3 (2.12.x used 13.0.2).
#   - Python 3.15 / 3.15t wheels added, LINUX ONLY (Windows stops at cp314; PyPI
#     ships cp310-cp314 only). See P14.
#   - cu129's cuDNN pin is 9.20.0.48, differing from cu126's 9.10.2.21 in the same
#     release. See P3.
#   - Triton pin 3.7.1; sympy>=1.13.3; torchvision 0.28.0 pins torch==2.13.0.
#   - PyPI default remains cu130 (CUDA_STABLE = "13.0").
#   - No torchaudio release.
#   Source: https://github.com/pytorch/pytorch/releases/tag/v2.13.0
#
# THE cu129-ON-WINDOWS HISTORY (verified by HEAD request against the wheel index).
# This is a recurring source of wrong data — the build matrix lists 12.9 in
# CUDA_ARCHES, but the Windows build explicitly removes it. See P1/P2.
#   torch 2.8.0  -> Windows cu129 wheel EXISTS
#   torch 2.9.0  -> Windows cu129 wheel EXISTS, even though 12.9 is absent from that
#                   tag's CUDA_ARCHES entirely. The wheels came from the release/2.9
#                   branch after tagging — see P16. The program DOES list this row,
#                   flagged out_of_matrix ("†"). Verified: cp310/cp312/cp314 all HTTP 200.
#   torch 2.9.1  -> NO Windows cu129 wheel  (first release to exclude it)
#   torch 2.10.0 -> NO Windows cu129 wheel
#   torch 2.11.0 -> NO Windows cu129 wheel
#   torch 2.12.0 -> n/a (12.9 not in CUDA_ARCHES)
#   torch 2.12.1 -> NO Windows cu129 wheel
#   torch 2.13.0 -> NO Windows cu129 wheel
# The program marked 2.9.1/2.10.0/2.11.0 cu129 as Windows-supported until 2026-08-03;
# that was a false positive and has been corrected.



# "Metapackage" component versions per CUDA release version.
# Row-per-version (was column-per-version; transposed once the list grew past ~12).
# COMPLETE for every CUDA release from 11.8.0 onward (37 versions as of 2026-08-29).
# Every value was generated directly from the redistribution JSONs and re-verified
# (406 values, 0 mismatches). Regenerate rather than hand-edit — see the enumerate
# command in GROUND TRUTH SOURCES.
# There is no 12.7.x — NVIDIA never released it. 11.8.0 shows "-" for nvjitlink
# because nvJitLink did not exist until CUDA 12.0.
+--------+----------+----------+----------+----------+------------+------------+------------+------------+------------+----------+-----------+
| CUDA   | nvrtc    | runtime  | nvcc     | cupti    | cublas     | cufft      | curand     | cusolver   | cusparse   | nvtx     | nvjitlink |
+--------+----------+----------+----------+----------+------------+------------+------------+------------+------------+----------+-----------+
| 11.8.0 | 11.8.89  | 11.8.89  | 11.8.89  | 11.8.87  | 11.11.3.6  | 10.9.0.58  | 10.3.0.86  | 11.4.1.48  | 11.7.5.86  | 11.8.86  | -         |
| 12.0.0 | 12.0.76  | 12.0.107 | 12.0.76  | 12.0.90  | 12.0.1.189 | 11.0.0.21  | 10.3.1.50  | 11.4.2.57  | 12.0.0.76  | 12.0.76  | 12.0.76   |
| 12.0.1 | 12.0.140 | 12.0.146 | 12.0.140 | 12.0.146 | 12.0.2.224 | 11.0.1.95  | 10.3.1.124 | 11.4.3.1   | 12.0.1.140 | 12.0.140 | 12.0.140  |
| 12.1.0 | 12.1.55  | 12.1.55  | 12.1.66  | 12.1.62  | 12.1.0.26  | 11.0.2.4   | 10.3.2.56  | 11.4.4.55  | 12.0.2.55  | 12.1.66  | 12.1.55   |
| 12.1.1 | 12.1.105 | 12.1.105 | 12.1.105 | 12.1.105 | 12.1.3.1   | 11.0.2.54  | 10.3.2.106 | 11.4.5.107 | 12.1.0.106 | 12.1.105 | 12.1.105  |
| 12.2.0 | 12.2.91  | 12.2.53  | 12.2.91  | 12.2.60  | 12.2.1.16  | 11.0.8.15  | 10.3.3.53  | 11.5.0.53  | 12.1.1.53  | 12.2.53  | 12.2.91   |
| 12.2.1 | 12.2.128 | 12.2.128 | 12.2.128 | 12.2.131 | 12.2.4.5   | 11.0.8.91  | 10.3.3.129 | 11.5.1.129 | 12.1.2.129 | 12.2.128 | 12.2.128  |
| 12.2.2 | 12.2.140 | 12.2.140 | 12.2.140 | 12.2.142 | 12.2.5.6   | 11.0.8.103 | 10.3.3.141 | 11.5.2.141 | 12.1.2.141 | 12.2.140 | 12.2.140  |
| 12.3.0 | 12.3.52  | 12.3.52  | 12.3.52  | 12.3.52  | 12.3.2.9   | 11.0.11.19 | 10.3.4.52  | 11.5.3.52  | 12.1.3.153 | 12.3.52  | 12.3.52   |
| 12.3.1 | 12.3.103 | 12.3.101 | 12.3.103 | 12.3.101 | 12.3.4.1   | 11.0.12.1  | 10.3.4.101 | 11.5.4.101 | 12.2.0.103 | 12.3.101 | 12.3.101  |
| 12.3.2 | 12.3.107 | 12.3.101 | 12.3.107 | 12.3.101 | 12.3.4.1   | 11.0.12.1  | 10.3.4.107 | 11.5.4.101 | 12.2.0.103 | 12.3.101 | 12.3.101  |
| 12.4.0 | 12.4.99  | 12.4.99  | 12.4.99  | 12.4.99  | 12.4.2.65  | 11.2.0.44  | 10.3.5.119 | 11.6.0.99  | 12.3.0.142 | 12.4.99  | 12.4.99   |
| 12.4.1 | 12.4.127 | 12.4.127 | 12.4.131 | 12.4.127 | 12.4.5.8   | 11.2.1.3   | 10.3.5.147 | 11.6.1.9   | 12.3.1.170 | 12.4.127 | 12.4.127  |
| 12.5.0 | 12.5.40  | 12.5.39  | 12.5.40  | 12.5.39  | 12.5.2.13  | 11.2.3.18  | 10.3.6.39  | 11.6.2.40  | 12.4.1.24  | 12.5.39  | 12.5.40   |
| 12.5.1 | 12.5.82  | 12.5.82  | 12.5.82  | 12.5.82  | 12.5.3.2   | 11.2.3.61  | 10.3.6.82  | 11.6.3.83  | 12.5.1.3   | 12.5.82  | 12.5.82   |
| 12.6.0 | 12.6.20  | 12.6.37  | 12.6.20  | 12.6.37  | 12.6.0.22  | 11.2.6.28  | 10.3.7.37  | 11.6.4.38  | 12.5.2.23  | 12.6.37  | 12.6.20   |
| 12.6.1 | 12.6.68  | 12.6.68  | 12.6.68  | 12.6.68  | 12.6.1.4   | 11.2.6.59  | 10.3.7.68  | 11.6.4.69  | 12.5.3.3   | 12.6.68  | 12.6.68   |
| 12.6.2 | 12.6.77  | 12.6.77  | 12.6.77  | 12.6.80  | 12.6.3.3   | 11.3.0.4   | 10.3.7.77  | 11.7.1.2   | 12.5.4.2   | 12.6.77  | 12.6.77   |
| 12.6.3 | 12.6.85  | 12.6.77  | 12.6.85  | 12.6.80  | 12.6.4.1   | 11.3.0.4   | 10.3.7.77  | 11.7.1.2   | 12.5.4.2   | 12.6.77  | 12.6.85   |
| 12.8.0 | 12.8.61  | 12.8.57  | 12.8.61  | 12.8.57  | 12.8.3.14  | 11.3.3.41  | 10.3.9.55  | 11.7.2.55  | 12.5.7.53  | 12.8.55  | 12.8.61   |
| 12.8.1 | 12.8.93  | 12.8.90  | 12.8.93  | 12.8.90  | 12.8.4.1   | 11.3.3.83  | 10.3.9.90  | 11.7.3.90  | 12.5.8.93  | 12.8.90  | 12.8.93   |
| 12.8.2 | 12.8.93  | 12.8.90  | 12.8.93  | 12.8.90  | 12.8.5.5   | 11.3.3.83  | 10.3.9.90  | 11.7.3.90  | 12.5.8.93  | 12.8.90  | 12.8.93   |
| 12.9.0 | 12.9.41  | 12.9.37  | 12.9.41  | 12.9.19  | 12.9.0.13  | 11.4.0.6   | 10.3.10.19 | 11.7.4.40  | 12.5.9.5   | 12.9.19  | 12.9.41   |
| 12.9.1 | 12.9.86  | 12.9.79  | 12.9.86  | 12.9.79  | 12.9.1.4   | 11.4.1.4   | 10.3.10.19 | 11.7.5.82  | 12.5.10.65 | 12.9.79  | 12.9.86   |
| 12.9.2 | 12.9.86  | 12.9.79  | 12.9.86  | 12.9.79  | 12.9.2.10  | 11.4.1.4   | 10.3.10.19 | 11.7.5.82  | 12.5.10.65 | 12.9.79  | 12.9.86   |
| 13.0.0 | 13.0.48  | 13.0.48  | 13.0.48  | 13.0.48  | 13.0.0.19  | 12.0.0.15  | 10.4.0.35  | 12.0.3.29  | 12.6.2.49  | 13.0.39  | 13.0.39   |
| 13.0.1 | 13.0.88  | 13.0.88  | 13.0.88  | 13.0.85  | 13.0.2.14  | 12.0.0.61  | 10.4.0.35  | 12.0.4.66  | 12.6.3.3   | 13.0.85  | 13.0.88   |
| 13.0.2 | 13.0.88  | 13.0.96  | 13.0.88  | 13.0.85  | 13.1.0.3   | 12.0.0.61  | 10.4.0.35  | 12.0.4.66  | 12.6.3.3   | 13.0.85  | 13.0.88   |
| 13.0.3 | 13.0.88  | 13.0.96  | 13.0.88  | 13.0.85  | 13.1.1.3   | 12.0.0.61  | 10.4.0.35  | 12.0.4.66  | 12.6.3.3   | 13.0.85  | 13.0.88   |
| 13.1.0 | 13.1.80  | 13.1.80  | 13.1.80  | 13.1.75  | 13.2.0.9   | 12.1.0.31  | 10.4.1.34  | 12.0.7.41  | 12.7.2.19  | 13.1.68  | 13.1.80   |
| 13.1.1 | 13.1.115 | 13.1.80  | 13.1.115 | 13.1.115 | 13.2.1.1   | 12.1.0.78  | 10.4.1.81  | 12.0.9.81  | 12.7.3.1   | 13.1.115 | 13.1.115  |
| 13.1.2 | 13.1.115 | 13.1.80  | 13.1.115 | 13.1.115 | 13.2.2.2   | 12.1.0.78  | 10.4.1.81  | 12.0.9.81  | 12.7.3.1   | 13.1.115 | 13.1.115  |
| 13.2.0 | 13.2.51  | 13.2.51  | 13.2.51  | 13.2.23  | 13.3.0.5   | 12.2.0.37  | 10.4.2.51  | 12.1.0.51  | 12.7.9.17  | 13.2.20  | 13.2.51   |
| 13.2.1 | 13.2.78  | 13.2.75  | 13.2.78  | 13.2.75  | 13.4.0.1   | 12.2.0.46  | 10.4.2.55  | 12.2.0.1   | 12.7.10.1  | 13.2.75  | 13.2.78   |
| 13.2.2 | 13.2.86  | 13.2.86  | 13.2.86  | 13.2.86  | 13.4.1.3   | 12.2.0.57  | 10.4.2.66  | 12.2.0.11  | 12.7.10.12 | 13.2.86  | 13.2.86   |
| 13.3.0 | 13.3.33  | 13.3.29  | 13.3.33  | 13.3.35  | 13.5.1.27  | 12.3.0.29  | 10.4.3.29  | 12.2.2.18  | 12.8.1.7   | 13.3.29  | 13.3.33   |
| 13.3.1 | 13.3.33  | 13.3.29  | 13.3.73  | 13.3.75  | 13.6.0.2   | 12.3.0.29  | 10.4.3.29  | 12.2.6.9   | 12.8.2.51  | 13.3.29  | 13.3.33   |
+--------+----------+----------+----------+----------+------------+------------+------------+------------+------------+----------+-----------+
* Obtained from: https://developer.download.nvidia.com/compute/cuda/redist/redistrib_{X.Y.Z}.json
* Human-readable cross-check: https://docs.nvidia.com/cuda/archive/{X.Y.Z}/cuda-toolkit-release-notes/index.html
* Column names map to JSON keys as: nvrtc=cuda_nvrtc, runtime=cuda_cudart, nvcc=cuda_nvcc,
  cupti=cuda_cupti, cublas=libcublas, cufft=libcufft, curand=libcurand,
  cusolver=libcusolver, cusparse=libcusparse, nvtx=cuda_nvtx, nvjitlink=libnvjitlink
* Patch releases often differ in only ONE component (13.0.2 -> 13.0.3 changes only
  cublas 13.1.0.3 -> 13.1.1.3; 12.8.1 -> 12.8.2 changes only cublas 12.8.4.1 -> 12.8.5.5).
  Do not assume a new patch version is a no-op, and do not assume it changed everything.
* Torch wheel refs (the subset that strictly must be present): 11.8.0, 12.4.1, 12.6.3,
  12.8.0, 12.8.1, 12.9.1, 13.0.0, 13.0.2, 13.0.3, 13.2.1. Everything else is tracked
  because the table is intentionally COMPLETE from the floor onward.



*********************************************************
VERSION SCOPE POLICY  (what belongs in the program at all)
*********************************************************
# Set by the maintainer on 2026-08-03. Apply this when deciding whether a version
# you just discovered should be added.
#
# CUDA:  include EVERY release from 11.8.0 onward. 11.8.0 is the floor because it is
#        the oldest CUDA referenced anywhere in the program (torch 2.6.0/2.7.x cu118).
#        Do NOT add anything older than 11.8.0, even if a tracked torch version
#        supports it. If a torch release supports a pre-11.8 CUDA (e.g. 11.7), that
#        combination is deliberately OUT OF SCOPE and must not appear.
#
# ALL OTHER LIBRARIES: include every release from the oldest version currently
#        tracked, onward. Do not go back further than the existing floor; do not
#        leave holes above it. Current floors:
#            torch          2.6.0
#            torchvision    0.21.0
#            torchaudio     2.6.0   (maintenance; no release after 2.11.0)
#            triton         3.2.0
#            xformers       0.0.29.post2
#            bitsandbytes   0.47.0
#            flash-attn     2.8.2   (both Linux and Windows tables)
#
# The failure mode this policy exists to prevent: for a long time versions were added
# only when a scan happened to surface them, which silently left holes ABOVE the floor
# (CUDA 12.6.0/12.6.1/12.6.2, 12.9.0, 13.0.1 were all missing while 13.3.1 was present).
# Completeness is now checkable mechanically — see P17.



*********************************************************
DEFERRED — EXISTS UPSTREAM, DELIBERATELY NOT TRACKED YET
*********************************************************
# VERSION SCOPE POLICY says what belongs in the program. This register is its
# complement: releases that DO exist upstream, are in scope by version floor, and are
# still deliberately absent. Without it every audit re-discovers these and re-reports
# them as findings, and a reader who learns to skim findings will eventually skim a
# real one.
#
# Each entry carries a RE-CHECK TRIGGER: the specific condition that turns it from
# deferred into actionable. Delete the entry once its trigger fires and it is added.
#
# --- triton 3.8.0 (Linux) and triton-windows 3.8.0.post28 ---  deferred 2026-08-29
#   triton 3.8.0 published to PyPI 2026-08-28 (cp310-cp314, manylinux x86_64/aarch64);
#   triton-windows 3.8.0.post28 published 2026-08-29 (cp310-cp314).
#   WHY DEFERRED: triton is not a standalone data structure in this program. It exists
#   only as the "triton" pin and "triton_compat" list INSIDE torch_python_triton rows.
#   No RELEASED torch pins 3.8 — torch 2.13.0 pins 3.7.1, and only release/2.14 pins
#   3.8.0. Adding a 3.8 entry today would create a row unreachable from any torch
#   version, which is the P18 failure mode in reverse.
#   RE-CHECK TRIGGER: torch 2.14.0 appears on PyPI / the stable wheel index.
#   ALREADY-CAPTURED GROUND TRUTH for when that happens (re-verify, do not trust this
#   blind — per P16 the release branch keeps moving after the tag):
#     - triton-windows README maps torch 2.14 -> triton 3.8.
#     - Bundled toolchain, release/3.8.x-windows/cmake/nvidia-toolchain-version.json:
#         ptxas 12.9.86, ptxas-blackwell 13.3.33, cuobjdump 13.1.80, nvdisasm 13.1.80,
#         cudacrt 13.1.80, cudart 13.1.80, cupti 12.8.90,
#         cupti-windows 13.3.35, cupti-blackwell 13.3.35   <-- TWO NEW KEYS in 3.8.x
#     - P5 AGAIN: the README's summary table says "3.8 | 12.9", which is correct for the
#       BASE ptxas only. The real bundle is mixed 12.8 / 12.9 / 13.3. Use the JSON.
#     - P20 AGAIN: that same JSON carries a "windows-arm64" block pulling CUDA 13.4 from
#       packages.nvidia.com/prerelease/cuda/13.4.0/. Prerelease, not a released toolkit,
#       absent from the redistribution index. It does NOT go in cuda_metapackages.
#
# --- torch 2.14.0 ---  deferred 2026-08-29
#   RC-only. Wheels exist on the TEST channel (test/cu126, test/cu130, test/cu132);
#   absent from PyPI and from the stable wheel index.
#   WHY DEFERRED: the program tracks released wheels. A test-channel wheel can be
#   rebuilt or withdrawn, and release/2.14 has already reversed a whole CUDA arch once
#   (13.4 was added, then removed on 2026-08-21).
#   RE-CHECK TRIGGER: torch 2.14.0 on PyPI / the stable index.
#   KNOWN STATE of release/2.14 as of 2026-08-29 (PROVISIONAL — re-read on trigger):
#     CUDA_ARCHES = ["12.6", "13.0", "13.2"], CUDA_STABLE = "13.0"
#     full versions 12.6.3 / 13.0.3 / 13.2.1; cu129 dropped entirely
#     cuDNN pins: 12.6 -> 9.10.2.21;  13.0 and 13.2 -> 9.24.0.43
#     triton pin 3.8.0;  FULL_PYTHON_VERSIONS 3.10-3.15 incl. 3.14t/3.15t
#     CUDA_ARCHES_NO_WINDOWS = []  (no Windows exclusions — see the new P2 entry)
#   Stale 2.14.0+cu134 Linux wheels remain on test/cu134 from before the 13.4 revert.
#   They are NOT evidence that cu134 is coming back. Ignore them.
#   TWO PARSE RECIPES IN THIS FILE BREAK ON 2.14 — fix them when 2.14 is added:
#     1. PYTORCH_EXTRA_INSTALL_REQUIREMENTS no longer lists individual nvidia-*-cuXX
#        packages. Each arch is now one
#        cuda-toolkit[nvrtc,cudart,cupti,cufft,curand,cusolver,cusparse,cublas,cufile,
#        nvjitlink,nvtx]==X.Y.Z metapackage, plus cuda-bindings, nvidia-cusparselt,
#        nvidia-nccl, nvidia-nvshmem. The nvidia-cudnn-cuNN== pin (P3/P19) SURVIVES
#        unchanged, so the cuDNN parse still works — but anything reading component
#        versions out of that dict does not.
#     2. The sympy spec MOVED from setup.py to pyproject.toml. The GROUND TRUTH SOURCES
#        entry pointing at setup.py will come up empty.
#
# --- flash-attention FA4 betas (fa4-v4.0.0.betaNN) ---  deferred, ongoing
#   Weekly cadence; beta28 published 2026-08-26 while FA2 2.x sits idle.
#   WHY DEFERRED: prereleases, filtered by P4. Also only 2 assets each, versus 50+ for
#   an FA2 release — not a compatibility matrix.
#   RE-CHECK TRIGGER: a NON-prerelease FA4 tag, or FA2 2.x resuming releases.


************
cuDNN & CUDA
************
# Nvidia promises the CUDA 12.x build of any given cuDNN 9.x release is
# forward-compatible with all CUDA 12.x toolkits. No equivalent promise
# is published for CUDA 13.x.
+-------------------+---------------------------+----------------------+
| cuDNN Package     | CUDA Toolkit              | Windows Support      |
+-------------------+---------------------------+----------------------+
| 9.x for CUDA 13.x | 13.0, 13.1, 13.2, 13.3    | NOT SUPPORTED [1]    |
| 9.x for CUDA 12.x | 12.0-12.6, 12.8, 12.9 [2] | Driver >= 527.41     |
|   + Blackwell GPU | 12.8, 12.9                | Driver >= 570.65 [3] |
+-------------------+---------------------------+----------------------+
[1] Linux-only since cuDNN 9.11.0 (Jul 2025). On Windows, use WSL2 with
    the Windows host driver -- do NOT install a Linux NVIDIA driver
    inside the WSL distro. Host driver minimums above still apply.
[2] CUDA 12.7 was never released by Nvidia, hence the gap.
[3] Blackwell (cc 10.0, 12.0) requires CUDA >= 12.8, Linux driver
    >= 570.26, Windows driver >= 570.65.

* GPU floor: cuDNN 9.11+ requires Turing (cc 7.5). Volta, Pascal, and
  Maxwell were removed in 9.11.0 (Jul 2025); 9.10.2 is the last 9.x
  that supports them. For older hardware, use cuDNN 8.9.x.
* Linux driver minimums: >= 525.60.13 (CUDA 12.x build),
  >= 580.65.06 (CUDA 13.x build).
* Recommended for tuning heuristics: cuDNN 9.25.1 + CUDA 13.3 (Linux).
  (Verbatim from the support matrix: "For best performance, the recommended
  configuration is cuDNN 9.25.1 with CUDA 13.3. This is the configuration used
  for tuning heuristics.")
* Windows-only quirks: side-by-side install dropped in 9.10.0 (must
  manually delete prior C:\Program Files\NVIDIA\CUDNN\v9.x tree before
  upgrading); lib path moved from lib\ to lib\x64\ in 9.x; no static
  archives, no JIT meta-package, no ARM64 on Windows.
* The "NOT SUPPORTED on Windows" claim for the CUDA 13.x build is NVIDIA's own
  (driver column reads N/A), even though nvidia-cudnn-cu13 does publish a
  win_amd64 wheel on PyPI. See the cuDNN entry in GROUND TRUTH SOURCES.

* taken from https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
* current cuDNN release as of last check: 9.25.1.1 (Aug 2026, cu12 and cu13, not yanked)
  NOTE: 9.25.0.15 exists on PyPI but every file is YANKED — not a valid "latest".
  9.25.1.1 is the fixed republish of that yanked build; 9.24.0.43 was the last good
  version before it and is what torch 2.14's release branch currently pins.


*****************************
WINDOWS-SPECIFIC LIMITATIONS
*****************************

# There are TWO distinct reasons a wheel is unusable on Windows. Do not conflate them:
#   "No cuDNN"     - the Windows wheel IS built and downloadable, but no Windows cuDNN
#                    exists for CUDA 13.x, so cuDNN-backed ops are unavailable.
#   "No Win wheel" - PyTorch does not build a Windows wheel for this arch at all. The
#                    download simply does not exist (HTTP 403 on the index).
+--------+--------+-----------------------------------------------+
| Torch  | Wheel  | Windows Status                                |
+--------+--------+-----------------------------------------------+
| 2.13.0 | cu126  | Full support                                  |
| 2.13.0 | cu129  | No Win wheel (12.9 excluded from Win build)   |
| 2.13.0 | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.13.0 | cu132  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.12.1 | cu126  | Full support                                  |
| 2.12.1 | cu129  | No Win wheel (12.9 excluded from Win build)   |
| 2.12.1 | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.12.1 | cu132  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.12.0 | cu126  | Full support                                  |
| 2.12.0 | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.12.0 | cu132  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.11.0 | cu126  | Full support                                  |
| 2.11.0 | cu128  | Full support                                  |
| 2.11.0 | cu129  | No Win wheel (12.9 excluded from Win build)   |
| 2.11.0 | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.10.0 | cu126  | Full support                                  |
| 2.10.0 | cu128  | Full support                                  |
| 2.10.0 | cu129  | No Win wheel (12.9 excluded from Win build)   |
| 2.10.0 | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.9.1  | cu126  | Full support                                  |
| 2.9.1  | cu128  | Full support                                  |
| 2.9.1  | cu129  | No Win wheel (12.9 excluded from Win build)   |
| 2.9.1  | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
| 2.9.0  | cu126  | Full support                                  |
| 2.9.0  | cu128  | Full support                                  |
| 2.9.0  | cu129  | Full support † (wheel exists; not in the tag) |
| 2.9.0  | cu130  | No cuDNN (cuDNN 9.x for CUDA 13 = Linux only) |
+--------+--------+-----------------------------------------------+
* The three cu129 rows for 2.9.1 / 2.10.0 / 2.11.0 read "Full support" until
  2026-08-03. That was WRONG — no Windows cu129 wheel has existed since torch 2.9.1.
  See the cu129-on-Windows history table in the Torch and CUDA section, and P1/P2.
* torch 2.8.0 cu129 DOES have a Windows wheel; the exclusion begins at 2.9.1.
* torch 2.9.0 cu129 also has a Windows wheel, but it is NOT in the v2.9.0 tagged
  matrix (P16), so it carries the "†" out_of_matrix marker. It was previously omitted
  from the program for exactly that reason — a false negative, now corrected.


****************************
Triton, Torch, and Python
****************************

# REPOSITORIES:
#   Original:  https://github.com/woct0rdho/triton-windows  (all releases up to v3.6.0-windows.post25)
#   Official:  https://github.com/triton-lang/triton-windows (v3.6.0-windows.post26 onwards)
#
# BRANCH STRUCTURE (same in both repos):
#   "release/X.Yx"          = upstream triton code (no Windows patches)
#   "release/X.Yx-windows"  = Windows-specific patches; releases are cut from here
#   "main-windows"          = development branch for Windows patches
#   "readme"                = standalone branch containing only the README
#
# RELEASE NAMING: vX.Y.Z-windows.postNN
#   The "postNN" number increments globally across ALL triton minor versions, not per-version.
#
# PYPI vs GITHUB:
#   Not all PyPI releases have a corresponding GitHub tag or release entry. For example,
#   3.2.0.post21 and 3.4.0.post21 exist on PyPI but have no GitHub tag. The maintainer
#   builds wheels from a release branch and uploads to PyPI without always creating a
#   GitHub release. PyPI is authoritative for what versions exist and can be pip installed.
#
# TORCH COMPATIBILITY:
#   Torch ↔ triton compatibility is determined at the MINOR version level (3.2, 3.3, etc.),
#   not at the post level. All post versions within a minor series (e.g., 3.2.0.post11
#   through 3.2.0.post21) share the same torch compatibility. Post versions differ only
#   in bug fixes and improvements (path handling, compiler detection, etc.).
#
# GROUND TRUTH FILES (in ground-truth/ folder):
#   README.md (from "readme" branch)  → torch ↔ triton version compatibility
#   nvidia-toolchain-version.json     → CUDA tool versions bundled inside each triton wheel
#   These two sources provide different (non-conflicting) information:
#   - The README defines which torch version works with which triton version.
#   - The JSON defines which CUDA tools are shipped inside the wheel.
#   The JSON lives on the release branch (e.g. release/3.2.x-windows) and applies to ALL
#   post versions built from that branch, including PyPI-only versions without GitHub tags.
#
# BUNDLED CUDA (from nvidia-toolchain-version.json on release/{X.Y}.x-windows):
#   3.2.x → CUDA 12.4 tools (ptxas 12.4.99, cudart 12.4.99)
#   3.3.x → CUDA 12.4 ptxas + 12.8 cudart (+ separate 12.8 ptxas for Blackwell)
#   3.4.x → CUDA 12.8 tools (ptxas 12.8.93, cudart 12.8.57)
#   3.5.x → CUDA 12.8 tools (identical to 3.4.x)
#   3.6.x → CUDA 12.8 tools (+ 12.9 ptxas for Blackwell)
#   3.7.x → MIXED 12.8 / 13.1 — NOT the same bundle as 3.3-3.6:
#             ptxas 12.8.93, cupti 12.8.90  (unchanged from 3.6.x)
#             ptxas-blackwell 13.1.80, cuobjdump 13.1.80, nvdisasm 13.1.80,
#             cudacrt 13.1.80, cudart 13.1.80
#
#   CORRECTION (2026-08-03): this file previously said "3.7.x → CUDA 12.8 tools
#   (same bundle as 3.3-3.6)". That was WRONG. It came from the README's coarse
#   summary table, which still reads "3.3 .. 3.7 | 12.8" and has not been updated.
#   The JSON is ground truth; the README table lags. See P5.
#   Side-by-side for reference:
#     tool             3.6.x      3.7.x
#     ptxas            12.8.93    12.8.93
#     ptxas-blackwell  12.9.86    13.1.80
#     cuobjdump        12.8.55    13.1.80
#     nvdisasm         12.8.55    13.1.80
#     cudacrt          12.8.61    13.1.80
#     cudart           12.8.57    13.1.80
#     cupti            12.8.90    12.8.90

# Triton-windows releases and their torch compatibility.
# Obtained from: README.md on the "readme" branch of each repository (see above).
# The "Notes" column references specific post versions where a feature was introduced;
# these come from the README author (the maintainer) who knows which build added what.
+--------------------------+----------------+-------------------------------+
| Release                  | Compatible     | Notes                         |
+--------------------------+----------------+-------------------------------+
| v3.7.1-windows.post27    | torch>=2.12    | README maps torch 2.12 AND    |
| v3.7.0-windows.post26    | torch>=2.12    | 2.13 to triton 3.7            |
| v3.6.0-windows.postXX    | torch>=2.10    |                               |
| v3.5.x-windows.postXX    | torch>=2.9     | 3.5.0 adds fp8 on RTX 30xx   |
| v3.4.0-windows.post21    | torch>=2.8     |                               |
| v3.3.x-windows.postXX    | torch>=2.7     | 3.3.0 adds RTX 50xx support  |
| v3.2.0-windows.post21    | torch>=2.6     | fp8 on RTX 20xx              |
+--------------------------+----------------+-------------------------------+
* Turing (GTX 16xx/RTX 20xx) support was dropped in upstream Triton 3.3.
  If you must use Turing, stick with triton-windows 3.2.x.

# Torch hard-pins a specific triton version. This table comes from PyTorch's repo,
# not from triton-windows. The triton-windows README states that patch versions
# within a minor series are compatible (e.g. 3.5.0 and 3.5.1 both work with torch 2.9).
+-------+----------------------------+-------------+--------+
| Torch | CUDA                       | Triton Pin  | Sympy  |
+-------+----------------------------+-------------+--------+
| 2.13.0| cu126, cu129, cu130, cu132 | 3.7.1       | 1.13.3 |
| 2.12.1| cu126, cu129, cu130, cu132 | 3.7.1       | 1.13.3 |
| 2.12.0| cu126, cu130, cu132        | 3.7.0       | 1.13.3 |
| 2.11.0| cu126, cu128, cu129, cu130 | 3.6.0       | 1.13.3 |
| 2.10.0| cu126, cu128, cu129, cu130 | 3.6.0       | 1.13.3 |
| 2.9.1 | cu126, cu128, cu129, cu130 | 3.5.1       | 1.13.3 |
| 2.9.0 | cu126, cu128, cu129, cu130 | 3.5.0       | 1.13.3 |
| 2.8.0 | cu126, cu128, cu129        | 3.4.0       | 1.13.3 |
| 2.7.1 | cu118, cu126, cu128        | 3.3.1       | 1.13.3 |
| 2.7.0 | cu118, cu126, cu128        | 3.3.0       | 1.13.3 |
| 2.6.0 | cu118, cu124, cu126        | 3.2.0       | 1.13.1 |
+-------+----------------------------+-------------+--------+
* Triton pin from https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt
  (check the tagged release, e.g. v2.11.0, for each torch version)
* Sympy version from https://github.com/pytorch/pytorch/blob/main/.ci/docker/requirements-ci.txt
  (the program stores setup.py's spec, e.g. sympy>=1.13.3, since that is what a user
  installing torch actually resolves; requirements-ci.txt pins == for CI only)
* The CUDA column lists the wheel monikers built for that torch release on ANY platform.
  cu129 for 2.9.1+ is Linux-only — see the Windows-specific limitations section.
* Linux users install the upstream package: pip install triton==3.7.1
  Windows users install the fork with prefix matching, because every triton-windows
  release is a .postN: pip install "triton-windows==3.7.1.*"   (see P9)


*************************
Linux Flash Attention 2
*************************

# GROUND TRUTH: publish.yml from tagged releases in Dao-AILab/flash-attention
#   v2.8.3: https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/.github/workflows/publish.yml
#   v2.8.2: https://github.com/Dao-AILab/flash-attention/blob/v2.8.2/.github/workflows/publish.yml
# Ground truth files saved in: ground-truth/fa2_linux_publish_v2.8.3.yml, fa2_linux_publish_v2.8.2.yml
#
# FA2 has no hard torch version pin (setup.py says install_requires=["torch"]).
# The CI build matrix defines which torch versions are officially tested/built.
# cu12 builds use CUDA 12.9.1 (compiled once, compatible with all CUDA 12.x at runtime).
# v2.8.3 also ships cu13 wheels (torch 2.9/2.10, cp312) for use with CUDA 13.x torch.

# FA2 wheels are built for specific torch versions
+--------------+------------------------------------------------------+
| FA2 Version  | Compatibility (Linux)                                |
+--------------+------------------------------------------------------+
| v2.8.3.post1 | torch 2.4.0 + cuda 12.x + cp39-cp312                 |
| v2.8.3.post1 | torch 2.5.1 + cuda 12.x + cp39-cp313                 |
| v2.8.3.post1 | torch 2.6.0 + cuda 12.x + cp39-cp313                 |
| v2.8.3.post1 | torch 2.7.1 + cuda 12.x + cp39-cp313                 |
| v2.8.3.post1 | torch 2.8.0 + cuda 12.x + cp39-cp313                 |
| v2.8.3.post1 | torch 2.9.0 + cuda 13.x + cp312 (x86_64, aarch64) *! |
+--------------+------------------------------------------------------+
| v2.8.3       | torch 2.4.0 + cuda 12.x + cp39-cp312                 |
| v2.8.3       | torch 2.5.1 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.6.0 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.7.1 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.8.0 + cuda 12.x + cp39-cp313                 |
| v2.8.3       | torch 2.9.0 + cuda 12.x + cp312 (x86_64, aarch64) ** |
| v2.8.3       | torch 2.9.0 + cuda 13.x + cp312 (x86_64) **          |
| v2.8.3       | torch 2.10.0 + cuda 13.x + cp312 (x86_64,aarch64) ** |
+--------------+------------------------------------------------------+
| v2.8.2       | torch 2.4.0 + cuda 12.x + cp39-cp312                 |
| v2.8.2       | torch 2.5.1 + cuda 12.x + cp39-cp313                 |
| v2.8.2       | torch 2.6.0 + cuda 12.x + cp39-cp313                 |
| v2.8.2       | torch 2.7.1 + cuda 12.x + cp39-cp313                 |
+--------------+------------------------------------------------------+
** torch 2.9.0/2.10.0 are NOT in the v2.8.3 publish.yml CI matrix (which only has up to
   2.8.0). These wheels were added to the GitHub release later (manually or via re-run).
   The cu13 wheels (cp312) are for use with CUDA 13.x torch builds.
   This is the same out-of-matrix phenomenon as torch 2.9.0/cu129 (P7/P16), but it is
   NOT rendered with the "†" marker. The marker lives on torch_cuda rows, where the
   question is "does a wheel exist for this platform"; FA2 rows are already an explicit
   per-wheel enumeration built from the actual release assets, so every FA2 row is
   asset-verified by construction and a marker would be redundant. Do not "fix" this
   asymmetry by adding out_of_matrix to flash_attention_linux.

# 2.8.3.post1 — TWO TRAPS. Read before touching this data.
#
# TRAP 1 (coverage regression, see P8): post1 is the LATEST version on PyPI, so
# `pip install flash-attn` resolves to it — but it has FEWER wheels than v2.8.3.
# It DROPPED cu12torch2.9 and cu13torch2.10 entirely. Consequences:
#   - torch 2.10.0 users must use v2.8.3, NOT the "latest" post1.
#   - torch 2.9.0 + CUDA 12.x users must use v2.8.3, NOT post1.
#   - torch 2.9.0 + CUDA 13.x works on both.
# Asset counts: v2.8.3 = 53, v2.8.3.post1 = 50.
#
# TRAP 2 (filename version, see P6): within the post1 release the asset names are
# INCONSISTENT. The 48 cu12 assets are named "flash_attn-2.8.3.post1+cu12torch...",
# but the 2 cu13 assets are named "flash_attn-2.8.3+cu13torch2.9..." — i.e. the
# release version does NOT appear in those filenames. A URL builder that interpolates
# the FA2 version straight into the filename produces a 404 for exactly those wheels.
# This is why flash_attention_linux entries carry an optional "wheel_ver" field:
#   {"fa2": "2.8.3.post1", ..., "cuda": "13", "wheel_ver": "2.8.3"}
#
# ABI note: post1 cu12 assets exist in BOTH cxx11abiTRUE and cxx11abiFALSE; the cu13
# assets are cxx11abiTRUE only. Selection still follows P11 (torch >= 2.7 -> TRUE).
#
# DECISION (2026-08-03): post1 IS tracked despite being a regression, because the
# program's first purpose is a comprehensive, honest compatibility map. Omitting the
# version that `pip install flash-attn` actually resolves to would (a) leave a user who
# already has post1 unable to find their combination, and (b) hide the very trap that
# makes this release dangerous — a torch 2.10 user reaching for "latest" gets nothing.


*************************
WINDOWS Flash Attention 2
*************************

# GROUND TRUTH: build-wheels.yml from kingbri1/flash-attention (main branch)
#   https://github.com/kingbri1/flash-attention/blob/main/.github/workflows/build-wheels.yml
# Ground truth file saved in: ground-truth/fa2_windows_build-wheels.yml
#
# NOTE: build-wheels.yml is a workflow_dispatch (manually triggered). The build matrix
# defines what WOULD be built, but actual release assets depend on which combinations
# were triggered. The table below reflects observed release assets, not the full matrix.
#
# The build-wheels.yml also builds Linux wheels (ubuntu-22.04), but those are separate
# from the official Dao-AILab Linux FA2 wheels built via publish.yml.
#
# LAST VERIFIED: August 29, 2026 — still v2.8.3 (17 assets, published 2025-08-16);
# no newer kingbri1 release exists. Data below re-confirmed unchanged.
# Windows FA2 compatibility data may be outdated. The table below was last verified
# against release assets on the date above. Check kingbri1/flash-attention releases
# for the latest available wheels.

# FA2 wheels are built for specific torch versions
+--------------+------------------------------------------------------+
| FA2 Version  | Compatibility (Windows)                              |
+--------------+------------------------------------------------------+
| v2.8.3       | torch 2.9.1*+ cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.9.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.8.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.7.0 + cuda 12.8 + cp310-cp313                |
| v2.8.3       | torch 2.6.0 + cuda 12.4 + cp311 only                 |
+--------------+------------------------------------------------------+
| v2.8.2       | torch 2.8.0 + cuda 12.8 + cp310-cp313                |
| v2.8.2       | torch 2.7.0 + cuda 12.8 + cp310-cp313                |
| v2.8.2       | torch 2.6.0 + cuda 12.4 + cp310-cp313                |
+--------------+------------------------------------------------------+
* = Assumed compatible (patch version, not officially tested)
* https://github.com/kingbri1/flash-attention


**************
Xformers
**************

# GROUND TRUTH (per tagged release, e.g. v0.0.35):
#   Torch version:  CU_VERSIONS list in .github/workflows/wheels.yml
#   FA2 range:      FLASH_VER_MIN / FLASH_VER_LAST in xformers/ops/fmha/flash.py
#   CUDA monikers:  CU_VERSIONS list in .github/workflows/wheels.yml
# Ground truth files saved in: ground-truth/xformers_wheels_*.yml, xformers_flash_*.py, xformers_cuda_*.yml
#
# CUDA COLUMN: Shows which CU monikers have xformers wheels, expressed as the
# corresponding torch CUDA version for that moniker (for matching against torch_cuda).
# The actual build toolkit (from setup-build-cuda/action.yml) may differ; notably,
# cu126 builds use CUDA 12.8.1 toolkit from v0.0.31+ to avoid Flash3 segfault.
# Only CUDA 12+ entries are shown (cu118 wheels also exist for older versions).
#
# Starting with v0.0.35, xformers declares torch>=2.10 (upward compatible).
# v0.0.34 pyproject.toml says torch>=2.10, but the published PyPI wheel metadata
# pins torch==2.10.0 (exact). Only v0.0.35+ truly allows torch>=2.10. See P10.
#
# As of 2026-08-29, v0.0.35 is still the latest STABLE release (only 0.0.35.devNNNN
# builds since — see P4). Because it declares torch>=2.10, it upward-covers torch
# 2.11 / 2.12.x / 2.13.0 with no data change needed. Caveat for users: no xformers
# wheel has actually been BUILT against 2.12.x or 2.13.0, so those combinations rely
# on the declared floor rather than a tested build.

+------------------+--------+---------------+--------------------------------+
| Xformers Version | Torch  |      FA2      |           CUDA 12+             |
+------------------+--------+---------------+--------------------------------+
| v0.0.35          | 2.10.0 | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.34          | 2.10.0 | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.33.post2    | 2.9.1  | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.33.post1    | 2.9.0  | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.33          | 2.9.0  | 2.7.1 - 2.8.4 | 12.6.3, 12.8.1, 13.0.0         |
| v0.0.32.post2    | 2.8.0  | 2.7.1 - 2.8.2 | 12.6.3, 12.8.1, 12.9.1         |
| v0.0.32.post1    | 2.8.0  | 2.7.1 - 2.8.2 | 12.6.3, 12.8.1, 12.9.1         |
| v0.0.32          | 2.8.0  | 2.7.1 - 2.8.2 | 12.6.3, 12.8.1, 12.9.1  * BUG |
| v0.0.31.post1    | 2.7.1  | 2.7.1 - 2.8.0 | 12.6.3, 12.8.0                 |
| v0.0.31          | 2.7.1  | 2.7.1 - 2.8.0 | 12.6.3, 12.8.0                 |
| v0.0.30          | 2.7.0  | 2.7.1 - 2.7.4 | 12.6.3, 12.8.0                 |
| v0.0.29.post3    | 2.6.0  | 2.7.1 - 2.7.2 | 12.4.1, 12.6.3                 |
| v0.0.29.post2    | 2.6.0  | 2.7.1 - 2.7.2 | 12.4.1, 12.6.3                 |
+------------------+--------+---------------+--------------------------------+
* Torch support: torch_version in wheels.yml build matrix (tagged release)
* FA2 support: FLASH_VER_MIN / FLASH_VER_LAST in xformers/ops/fmha/flash.py (tagged release)
* CUDA monikers: CU_VERSIONS in wheels.yml; versions shown are torch's CUDA for each moniker
* v0.0.32 is YANKED on PyPI (every file), which independently corroborates the "BUG"
  marker on that row. It is deliberately KEPT in the program: a user may already have it
  installed and needs to find out it is the bad one. A completeness audit per P17 will
  report 0.0.32 as "tracked but not upstream" — that is expected, not a defect.
* The xformers list is complete from the 0.0.29.post2 floor onward (audited 2026-08-29).


**************
Bitsandbytes
**************

# GROUND TRUTH: python-package.yml from tagged releases in bitsandbytes-foundation/bitsandbytes
#   e.g. https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/.github/workflows/python-package.yml
# Ground truth files saved in: ground-truth/bnb_python-package_*.yml
#
# CUDA versions come from the cuda_version matrix in the build-cuda job.
# The same matrix builds for Linux (ubuntu), Windows (windows-2025), and ARM.
#
# Python: Wheels are tagged py3 (Python-version agnostic). The supported Python
# range comes from requires-python in pyproject.toml, not the CI matrix.
# v0.50.1: requires-python >= 3.10 (classifiers: 3.10-3.14)
# v0.50.0: requires-python >= 3.10 (classifiers: 3.10-3.14)
# v0.49.x: requires-python >= 3.10 (classifiers: 3.10-3.14)
# v0.48.x: requires-python >= 3.9  (classifiers: 3.9-3.13)
# v0.47.0: requires-python >= 3.9  (classifiers: 3.9-3.13)
#
# v0.50.0 SHRANK the CUDA matrix — another case of a newer release covering LESS
# (see P8). Versus v0.49.2:
#   DROPPED: 12.0.1, 12.2.2, 12.3.2, 12.5.1, 12.9.1
#   ADDED:   13.2.0
#   KEPT:    11.8.0, 12.1.1, 12.4.1, 12.6.3, 12.8.1, 13.0.2
# The 12.9.1 drop matters: a user on torch cu129 (2.9.1 / 2.10.0 / 2.11.0 / 2.12.1 /
# 2.13.0, all Linux) has NO matching bitsandbytes 0.50.0/0.50.1 build and must stay
# on 0.49.x. The gap introduced in 0.50.0 persists in 0.50.1.
# v0.50.x wheels: manylinux_2_24 x86_64/aarch64, win_amd64, win_arm64, macos arm64.
#
# v0.50.1 and v0.50.2 are pure republishes as far as this program is concerned: the
# cuda_version list is byte-identical across 0.50.0 / 0.50.1 / 0.50.2, and pyproject.toml
# is byte-identical across all three (sha256 d88a6f7fd14f5226438a2cab4b48ace737b998fc5444
# bcbb972e6586ec8ce676 for 0.50.1 and 0.50.2). Each row is a straight copy of the 0.50.0
# row. Three consecutive releases with an unchanged build matrix is the normal pattern
# here, not a parsing error — re-verify with a diff rather than assuming a copy is wrong.
#
# TRAP — do NOT add CUDA 13.4.0 to cuda_metapackages. From 0.50.1 the build-cuda job
# carries an extra matrix entry that is NOT part of the tracked matrix:
#     include:
#       - os: windows-11-vs2026-arm
#         cuda_version: "13.4.0"
# That is a Windows-on-ARM (win_arm64) build for NVIDIA RTX/DGX Spark. Per
# .github/scripts/install-cuda-woa.ps1 it installs a PREVIEW toolkit from
# packages.nvidia.com/prerelease/cuda/13.4.0/... , not a released one. 13.4.0 is
# absent from the CUDA redistribution index (404; the index still ends at 13.3.1),
# so there is no redistrib JSON to source component versions from. See P20.
# The program's bitsandbytes CUDA column tracks the Windows x64 matrix, which is
# unchanged.

+------------------+-----------------------------------------------+----------------------+
| bitsandbytes     | CUDA versions (Windows wheels)                | Python (all rows)    |
+------------------+-----------------------------------------------+----------------------+
| v0.50.2          | 11.8.0, 12.1.1, 12.4.1, 12.6.3, 12.8.1,       | 3.10, 3.11, 3.12,    |
|                  | 13.0.2, 13.2.0                                | 3.13, 3.14           |
|                  | (identical to 0.50.1; still NO 12.9.1)        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.50.1          | 11.8.0, 12.1.1, 12.4.1, 12.6.3, 12.8.1,       | 3.10, 3.11, 3.12,    |
|                  | 13.0.2, 13.2.0                                | 3.13, 3.14           |
|                  | (identical to 0.50.0; still NO 12.9.1)        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.50.0          | 11.8.0, 12.1.1, 12.4.1, 12.6.3, 12.8.1,       | 3.10, 3.11, 3.12,    |
|                  | 13.0.2, 13.2.0                                | 3.13, 3.14           |
|                  | (NO 12.9.1 — dropped this release)            | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.49.2          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.10, 3.11, 3.12,    |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.13, 3.14           |
|                  | 13.0.2                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.49.1          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.10, 3.11, 3.12,    |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.13, 3.14           |
|                  | 13.0.2                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.49.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.10, 3.11, 3.12,    |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.13, 3.14           |
|                  | 13.0.2                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.48.2          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12, 3.13           |
|                  | 13.0.1                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.48.1          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12, 3.13           |
|                  | 13.0.1                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.48.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1,       | 3.12, 3.13           |
|                  | 13.0.1                                        | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
| v0.47.0          | 11.8.0, 12.0.1, 12.1.1, 12.2.2, 12.3.2,       | 3.9, 3.10, 3.11,     |
|                  | 12.4.1, 12.5.1, 12.6.3, 12.8.1, 12.9.1        | 3.12, 3.13           |
|                  |                                               | (py3 wheel)          |
+------------------+-----------------------------------------------+----------------------+
"""