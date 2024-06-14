import torch
from torch.utils.data import dataloader , distributed
import numpy as np
import random
import os
from utils import *
import math
import glob
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
VERBOSE =  True # global verbose mode




class LoadImagesAndVideos:
    """
    YOLOv8 image/video dataloader.

    This class manages the loading and pre-processing of image and video data for YOLOv8. It supports loading from
    various formats, including single image files, video files, and lists of image and video paths.

    Attributes:
        files (list): List of image and video file paths.
        nf (int): Total number of files (images and videos).
        video_flag (list): Flags indicating whether a file is a video (True) or an image (False).
        mode (str): Current mode, 'image' or 'video'.
        vid_stride (int): Stride for video frame-rate, defaults to 1.
        bs (int): Batch size, set to 1 for this class.
        cap (cv2.VideoCapture): Video capture object for OpenCV.
        frame (int): Frame counter for video.
        frames (int): Total number of frames in the video.
        count (int): Counter for iteration, initialized at 0 during `__iter__()`.

    Methods:
        _new_video(path): Create a new cv2.VideoCapture object for a given video path.
    """

    def __init__(self, path, batch=1, vid_stride=1):
        """Initialize the Dataloader and raise FileNotFoundError if file not found."""
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # Define files as images or videos
        images, videos = [], []
        for f in files:
            suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
            if suffix in IMG_FORMATS:
                images.append(f)
           
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.ni = ni  # number of images
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        self.vid_stride = vid_stride  # video frame-rate stride
        self.bs = batch
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found ")

    def __iter__(self):
        """Returns an iterator object for VideoStream or ImageFolder."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next batch of images or video frames along with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # end of file list
                if len(imgs) > 0:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.count]
            if self.video_flag[self.count]:
                self.mode = "video"
                if not self.cap or not self.cap.isOpened():
                    self._new_video(path)

                for _ in range(self.vid_stride):
                    success = self.cap.grab()
                    if not success:
                        break  # end of video or failure

                if success:
                    success, im0 = self.cap.retrieve()
                    if success:
                        self.frame += 1
                        paths.append(path)
                        imgs.append(im0)
                        info.append(f"video {self.count + 1}/{self.nf} (frame {self.frame}/{self.frames}) {path}: ")
                        if self.frame == self.frames:  # end of video
                            self.count += 1
                            self.cap.release()
                else:
                    # Move to the next file if the current video ended or failed to open
                    self.count += 1
                    if self.cap:
                        self.cap.release()
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
            else:
                self.mode = "image"
                im0 = cv2.imread(path)  # BGR
                if im0 is None:
                    raise FileNotFoundError(f"Image Not Found {path}")
                paths.append(path)
                imgs.append(im0)
                info.append(f"image {self.count + 1}/{self.nf} {path}: ")
                self.count += 1  # move to the next file
                if self.count >= self.ni:  # end of image list
                    break

        return paths, imgs, info

    def _new_video(self, path):
        """Creates a new video capture object for the given path."""
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        """Returns the number of batches in the object."""
        return math.ceil(self.nf / self.bs)  # number of files

class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()



class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )