<#
.SYNOPSIS
    Build the portable Windows app: one folder holding StereoCraft.exe, the Python
    runtime it needs and the depth model's weights.  No installer, no Python on
    the machine it runs on.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1

.EXAMPLE
    # A build for machines with an NVIDIA card: three times the size, and a
    # tenth of a second a photo rather than twenty.
    powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1 -Cuda

.NOTES
    Wants about 5 GB of free space to work in, or 13 GB with -Cuda.  Everything
    it leaves behind can go afterwards except <Work>\models and <Work>\ffmpeg,
    which save the next build a 1.3 GB download and a small one.
#>
[CmdletBinding()]
param(
    # Python to build with; anything from 3.10 to 3.14 that Torch publishes
    # wheels for.  The build's Python is the app's Python, so this is the only
    # one that ever has to exist.
    [string]$Python = "",
    [string]$Work = "$env:USERPROFILE\StereoCraft-build",
    # da3 measures depth in metres and is what the app uses by default; the da2
    # models only rank it, and ride along as a fallback.
    # realbasicvsr is the super-resolution pass VR180 uses on a clip too small
    # to fill the frame; 210 MB, and the app simply does not offer the option
    # without it rather than reaching for the network.
    # lama paints the edge of the surround, for the clip whose lens never reached
    # the rest of the sphere; 200 MB, and without it that pass still runs and
    # simply stops at what the clip itself saw.
    # lama, sdxl and flux belong to `outpaint`, which is off by default and is
    # not shipped with weights: it was measured against the footage it was built
    # for and did not earn its 13 GB -- see the note at the top of `outpaint`.
    # They stay selectable so the experiment can be picked up again without
    # editing this file, and out of the default so the build stays around 6 GB
    # rather than 20.
    [ValidateSet("da3", "da2-small", "da2-base", "da2-large", "realbasicvsr", "rife",
                 "lama", "sdxl", "flux")]
    [string[]]$Models = @("da3", "realbasicvsr", "rife"),
    # Video needs ffmpeg.  It is fetched and laid beside the exe unless told not
    # to, in which case a copy has to be on the machine the app runs on.
    [switch]$SkipFfmpeg,
    [switch]$Cuda,
    # Which CUDA build of Torch to take when -Cuda is given.  cu126 by default:
    # it runs on any driver from 525 up, where cu130 wants 580 or newer.
    [string]$TorchIndex = "",
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$root = (Get-Item $PSScriptRoot).Parent.Parent.FullName

function Invoke-Tool {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "`n> $Exe $($Arguments -join ' ')" -ForegroundColor DarkCyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

function Resolve-Python {
    if ($Python) { return $Python }
    foreach ($candidate in @("py", "python")) {
        $found = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($found) {
            # `py` is a launcher rather than an interpreter; ask it which one it means.
            $exe = & $found.Source -c "import sys; print(sys.executable)"
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        }
    }
    throw "No Python found.  Install one from python.org and pass -Python <path to python.exe>."
}

$version = (Select-String -Path (Join-Path $root "pyproject.toml") -Pattern '^version\s*=\s*"(.+)"').Matches[0].Groups[1].Value
$flavour = if ($Cuda) { "cuda" } else { "cpu" }
$stage = Join-Path $Work "src"
# One tree per flavour: `pip install torch` leaves an already-satisfied Torch
# alone, so a CPU environment reused for a CUDA build would quietly ship the
# wrong one.
$venv = Join-Path $Work "venv-$flavour"
$vpy = Join-Path $venv "Scripts\python.exe"
$dist = Join-Path $Work "dist-$flavour"
$app = Join-Path $dist "StereoCraft"

Write-Host "StereoCraft $version -- portable Windows build ($flavour)" -ForegroundColor Green
Write-Host "  source : $root"
Write-Host "  build  : $Work"

# --- a copy of the source to build from ------------------------------------
# PyInstaller reads and writes a great deal; keeping all of it on a local disk
# matters when the checkout itself lives somewhere slower, WSL included.
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
foreach ($item in @("stereocraft", "packaging", "pyproject.toml", "requirements.txt", "README.md", "LICENSE")) {
    Copy-Item (Join-Path $root $item) -Destination $stage -Recurse -Force
}
Get-ChildItem $stage -Recurse -Force -Filter "__pycache__" | Remove-Item -Recurse -Force

# --- the environment the app is frozen out of ------------------------------
if (-not (Test-Path $vpy)) {
    Invoke-Tool (Resolve-Python) @("-m", "venv", $venv)
}
Invoke-Tool $vpy @("-m", "pip", "install", "--upgrade", "--quiet", "pip", "wheel")

# Torch comes from its own index: PyPI's Windows wheel carries the CUDA
# runtime, which is gigabytes of no use to a machine without an NVIDIA card.
if (-not $TorchIndex) { $TorchIndex = if ($Cuda) { "cu126" } else { "cpu" } }
Invoke-Tool $vpy @("-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/$TorchIndex", "torch")
# Depth Anything 3 is installed without its own dependency list, which pins a
# different Torch and would quietly replace the one just installed above, along
# with pulling in a reconstruction and visualisation stack the app never calls.
# requirements.txt carries what it actually needs; --no-deps keeps it to that.
$requirements = Get-Content (Join-Path $stage "requirements.txt") |
    Where-Object { $_ -and $_ -notmatch "^\s*#" -and $_ -notmatch "^torch\b" }
Invoke-Tool $vpy (@("-m", "pip", "install") + $requirements + @("pyinstaller"))
# --ignore-requires-python because the wheel caps itself at Python 3.13, which
# was simply the newest version when it was published.  It is pure Python and
# runs on 3.14 unchanged -- checked by importing it and converting a photo there.
Invoke-Tool $vpy @("-m", "pip", "install", "--no-deps", "--ignore-requires-python",
                   "depth-anything-3")

# --- ffmpeg, for video -----------------------------------------------------
# A build-time download rather than something the user has to install: the whole
# promise of this folder is that it runs on a machine with nothing on it.
$ffmpegDir = Join-Path $stage "packaging\windows\ffmpeg"
if (-not $SkipFfmpeg) {
    $ffmpegCache = Join-Path $Work "ffmpeg"
    if (-not (Test-Path (Join-Path $ffmpegCache "ffmpeg.exe"))) {
        $archive = Join-Path $Work "ffmpeg.zip"
        Write-Host "`nFetching ffmpeg ..." -ForegroundColor DarkCyan
        # The GPL build, for x264 and x265 -- the best encoders of the lot at a
        # given file size, and this app is built for its author rather than for
        # handing out.  An LGPL build works too and `video.pick_encoder` copes
        # with it, falling to the graphics card; that is the one to fetch if this
        # folder is ever going to be given to anyone.  See the README.
        Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" `
            -OutFile $archive -UseBasicParsing
        $unpacked = Join-Path $Work "ffmpeg-unpacked"
        if (Test-Path $unpacked) { Remove-Item $unpacked -Recurse -Force }
        Expand-Archive -Path $archive -DestinationPath $unpacked -Force
        New-Item -ItemType Directory -Path $ffmpegCache -Force | Out-Null
        Get-ChildItem $unpacked -Recurse -Include "ffmpeg.exe", "ffprobe.exe" |
            ForEach-Object { Copy-Item $_.FullName -Destination $ffmpegCache -Force }
        Remove-Item $archive, $unpacked -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ffmpegDir -Force | Out-Null
    Copy-Item (Join-Path $ffmpegCache "*.exe") -Destination $ffmpegDir -Force
}

# --- weights ---------------------------------------------------------------
$weights = Join-Path $Work "models"
foreach ($model in $Models) {
    # A repository lands as a folder with a config in it; a single checkpoint
    # lands as one file.  Either way an empty folder means it was never fetched.
    if (-not (Get-ChildItem (Join-Path $weights $model) -File -ErrorAction SilentlyContinue)) {
        Invoke-Tool $vpy (@((Join-Path $stage "packaging\windows\fetch_models.py")) + $Models + @("-o", $weights))
        break
    }
}

# --- freeze ----------------------------------------------------------------
Invoke-Tool $vpy @(
    "-m", "PyInstaller", "--noconfirm", "--clean",
    "--distpath", $dist, "--workpath", (Join-Path $Work "work-$flavour"),
    (Join-Path $stage "packaging\windows\stereocraft.spec")
)

# --- everything that goes beside the exe rather than inside it -------------
foreach ($model in $Models) {
    $to = Join-Path $app "models\$model"
    New-Item -ItemType Directory -Path $to -Force | Out-Null
    Copy-Item (Join-Path $weights "$model\*") -Destination $to -Recurse -Force
}
# Beside the exe rather than inside the archive: that is where `video._tool`
# looks, and where someone adding their own copy would put it.
if (-not $SkipFfmpeg) {
    Copy-Item (Join-Path $ffmpegDir "*.exe") -Destination $app -Force
}
Copy-Item (Join-Path $root "README.md") -Destination $app -Force
Copy-Item (Join-Path $root "LICENSE") -Destination $app -Force
Set-Content -Path (Join-Path $app "Read me first.txt") -Encoding UTF8 -Value @"
StereoCraft $version -- side-by-side 3D photos and video

Double-click StereoCraft.exe.  Nothing to install: the whole app is this folder, so
it can live on a USB stick or anywhere else you like, as long as it stays
together.  StereoCraft-cli.exe is the same program for the command line -- run it
from a terminal with --help to see what it takes.

This is the $flavour build. $(if ($Cuda) { "It uses an NVIDIA card when there is one, and falls back to the processor." } else { "It runs on the processor, so a large photo takes a little while and video is slow going." })
$(if ($SkipFfmpeg) { "Built without ffmpeg, so video needs ffmpeg installed on this machine, or ffmpeg.exe and ffprobe.exe dropped into this folder." } else { "ffmpeg is in this folder, so video works with nothing else installed." })
"@

$size = "{0:N0} MB" -f ((Get-ChildItem $app -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "`nBuilt $app ($size)" -ForegroundColor Green

if (-not $SkipZip) {
    $zip = Join-Path $Work "StereoCraft-$version-win64-$flavour.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Write-Host "Zipping to $zip ..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($app, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
    $zipsize = "{0:N0} MB" -f ((Get-Item $zip).Length / 1MB)
    Write-Host "`nDistribute $zip ($zipsize)" -ForegroundColor Green
}
