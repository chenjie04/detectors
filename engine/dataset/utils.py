from typing import List, Dict, Union, Tuple
import random
import time
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

from engine.utils import LOGGER
from engine.utils.ops import segments2boxes

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
IMG_FORMATS = {
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
    "heic",
}  # image suffixes
VID_FORMATS = {
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
    "webm",
}  # video suffixes
FORMATS_HELP_MSG = (
    f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
)


def check_file_speeds(
    files: List[str],
    threshold_ms: float = 10,
    threshold_mb: float = 50,
    max_files: int = 5,
    prefix: str = "",
):
    """
    Check dataset file access speed and provide performance feedback.

    This function tests the access speed of dataset files by measuring ping (stat call) time and read speed.
    It samples up to 5 files from the provided list and warns if access times exceed the threshold.

    Args:
        files (List[str]): List of file paths to check for access speed.
        threshold_ms (float, optional): Threshold in milliseconds for ping time warnings.
        threshold_mb (float, optional): Threshold in megabytes per second for read speed warnings.
        max_files (int, optional): The maximum number of files to check.
        prefix (str, optional): Prefix string to add to log messages.

    Examples:
        >>> from pathlib import Path
        >>> image_files = list(Path("dataset/images").glob("*.jpg"))
        >>> check_file_speeds(image_files, threshold_ms=15)
    """
    if not files or len(files) == 0:
        LOGGER.warning(f"{prefix}Image speed checks: No files to check")
        return

    # Sample files (max 5)
    files = random.sample(files, min(max_files, len(files)))

    # Test ping (stat time)
    ping_times = []
    file_sizes = []
    read_speeds = []

    for f in files:
        try:
            # Measure ping (stat call)
            start = time.perf_counter()
            file_size = os.stat(f).st_size
            ping_times.append((time.perf_counter() - start) * 1000)  # ms
            file_sizes.append(file_size)

            # Measure read speed
            start = time.perf_counter()
            with open(f, "rb") as file_obj:
                _ = file_obj.read()
            read_time = time.perf_counter() - start
            if read_time > 0:  # Avoid division by zero
                read_speeds.append(file_size / (1 << 20) / read_time)  # MB/s
        except Exception:
            pass

    if not ping_times:
        LOGGER.warning(f"{prefix}Image speed checks: failed to access files")
        return

    # Calculate stats with uncertainties
    avg_ping = np.mean(ping_times)
    std_ping = np.std(ping_times, ddof=1) if len(ping_times) > 1 else 0
    size_msg = f", size: {np.mean(file_sizes) / (1 << 10):.1f} KB"
    ping_msg = f"ping: {avg_ping:.1f}±{std_ping:.1f} ms"

    if read_speeds:
        avg_speed = np.mean(read_speeds)
        std_speed = np.std(read_speeds, ddof=1) if len(read_speeds) > 1 else 0
        speed_msg = f", read: {avg_speed:.1f}±{std_speed:.1f} MB/s"
    else:
        speed_msg = ""

    if avg_ping < threshold_ms or avg_speed < threshold_mb:
        LOGGER.info(f"{prefix}Fast image access ✅ ({ping_msg}{speed_msg}{size_msg})")
    else:
        LOGGER.warning(
            f"{prefix}Slow image access detected ({ping_msg}{speed_msg}{size_msg}). "
            f"Use local storage instead of remote/mounted storage for better performance. "
            f"See https://docs.ultralytics.com/guides/model-training-tips/"
        )


def img2label_paths(img_paths: List[str]) -> List[str]:
    """Convert image paths to label paths by replacing 'images' with 'labels' and extension with '.txt'."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def load_dataset_cache_file(path: Path) -> Dict:
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix: str, path: Path, x: Dict, version: str):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        with open(
            str(path), "wb"
        ) as file:  # context manager here fixes windows async np.save bug
            np.save(file, x)
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(
            f"{prefix}Cache directory {path.parent} is not writeable, cache not saved."
        )


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)

def get_hash(paths: List[str]) -> str:
    """Return a single hash value of a list of paths (files or dirs)."""
    size = 0
    for p in paths:
        try:
            size += os.stat(p).st_size
        except OSError:
            continue
    h = __import__("hashlib").sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def verify_image_label(args: Tuple) -> List:
    """Verify one image-label pair."""
    im_file, lb_file, num_cls = args
    # Number (missing, found, empty, corrupt), message
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):

                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                points = lb[:, 1:]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels {lb[lb < -0.01]}"

                # All labels
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, nm, nf, ne, nc, msg]

def exif_size(img: Image.Image) -> Tuple[int, int]:
    """Return exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s
