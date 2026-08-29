"""Download the checkpoints the portable build ships with.

Straight into a plain folder rather than a Hugging Face cache: `from_pretrained`
on a directory never looks at the network, which is what makes the packaged app
start the same on a machine that has no internet.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from stereocraft import interpolate, outpaint, plate, upscale  # noqa: E402
from stereocraft.depth import DA3_MODEL, MODELS  # noqa: E402

# Every checkpoint the build can ship, under the folder name the app looks for:
# `depth.checkpoint` resolves `models/<name>` beside the exe, and a plain folder
# is something `from_pretrained` reads without touching the network at all.
REPOS = {"da3": DA3_MODEL, **MODELS}
# These are one file rather than a repository, so they are fetched by name and
# checked by hash -- each comes from a mirror rather than from the authors, who
# publish to Google Drive.  `<module>.checkpoint` looks for exactly these paths
# beside the exe.
SINGLE = {"realbasicvsr": (upscale.REPO, upscale.FILENAME, upscale.REVISION),
          "rife": (interpolate.REPO, interpolate.FILENAME, interpolate.REVISION),
          "lama": (plate.REPO, plate.FILENAME, plate.REVISION)}
# Which of them can say whether what arrived is what was expected.  `rife` has no
# hash published to check against; the other two do, and are checked before they
# are copied rather than after.
VERIFY = {"realbasicvsr": upscale.verify, "lama": plate.verify}
# The outpainting backends are whole diffusers pipelines rather than one file,
# so they come down as a snapshot pinned by revision.  `flux` is 13 GB and is
# what makes a bundled build large; leave it out of -Models for a smaller one
# and the app falls back to `sdxl`, then to `lama`, then to the plain wash.
PIPELINES = {name: outpaint.REPOS[name] for name in outpaint.REPOS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+",
                        choices=sorted(set(REPOS) | set(SINGLE) | set(PIPELINES)))
    parser.add_argument("-o", "--output", required=True, help="folder to fill with <model>/")
    args = parser.parse_args(argv)

    import shutil

    from huggingface_hub import hf_hub_download, snapshot_download

    for name in args.models:
        target = os.path.join(args.output, name)
        if name in PIPELINES:
            repo, revision = PIPELINES[name]
            print(f"fetching {repo} -> {target}")
            snapshot_download(repo, revision=revision, local_dir=target,
                              # The repos carry fp32 alongside fp16 and the app
                              # only ever loads one of them.
                              ignore_patterns=["*.bin", "*.onnx", "*.msgpack",
                                               "*non_ema*", "*.png", "*.jpg"])
            shutil.rmtree(os.path.join(target, ".cache"), ignore_errors=True)
            continue
        if name in SINGLE:
            repo, filename, revision = SINGLE[name]
            print(f"fetching {repo}/{filename} -> {target}")
            os.makedirs(target, exist_ok=True)
            got = hf_hub_download(repo, filename, revision=revision)
            if name in VERIFY:
                VERIFY[name](got)  # before it is copied, not after
            shutil.copyfile(got, os.path.join(target, filename))
            continue
        print(f"fetching {REPOS[name]} -> {target}")
        snapshot_download(
            REPOS[name],
            local_dir=target,
            # The repos carry the same weights twice, as .bin and .safetensors;
            # only one of them is wanted, and it is the one that loads faster.
            allow_patterns=["*.json", "*.safetensors"],
        )
        # Bookkeeping for a re-download that will never happen here, and the app
        # would ship it to every machine.
        shutil.rmtree(os.path.join(target, ".cache"), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
