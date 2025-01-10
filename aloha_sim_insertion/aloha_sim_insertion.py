import time

import numpy as np
import pandas as pd
import tensorflow as tf
# from datasets import load_dataset
# from collections import defaultdict
import tensorflow_datasets as tfds

import glob
import os

import torch
import torchvision
from pathlib import Path

import logging


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float = 0.0001,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    assert len(timestamps) == len(closest_frames)
    return closest_frames


class AlohaSimInsertion(tfds.core.GeneratorBasedBuilder):
    """
    Convert a Hugging Face dataset into a TFDS-style episodic dataset with metadata.
    """
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, **kwargs):
        """
        Args:
            dataset_name (str): Name of the Hugging Face dataset.
            episodes (List[int]): List of episode indices to load. Default is None (load all).
        """
        dataset_name = "lerobot/aloha_sim_insertion"
        self.dataset_name = dataset_name
        self.root = "/home/hongyi/Codes/flower_rss24/aloha_sim_insertion_human/"
        super().__init__()

    def _info(self):
        """Define dataset information and features."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"Converted from the Hugging Face dataset {self.dataset_name}.",
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset(
                    {
                        'is_first': tf.bool,
                        'is_last': tf.bool,
                        'observation': tfds.features.FeaturesDict({
                            'state': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                            'images_top': tfds.features.Image(shape=(640, 480, 3), dtype=np.uint8)
                        }),
                        'action': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                        'reward': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'timestamp': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'frame_index': tfds.features.Tensor(shape=(), dtype=tf.int32),
                        'is_terminal': tfds.features.Tensor(shape=(), dtype=tf.bool),
                        'language_instruction': tfds.features.Text(),
                        'discount': tfds.features.Tensor(shape=(), dtype=tf.float32),
                        'metadata': tfds.features.FeaturesDict({
                            'episode_index': tfds.features.Tensor(shape=(), dtype=tf.int32)
                        }),
                    }
                ),
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Tensor(shape=(), dtype=tf.int32),
                })
            }),
        )

    def _split_generators(self, dl_manager):
        """Specify dataset splits."""
        return {
            'train': self._generate_examples()
        }

    def _generate_examples(self):
        """Yield examples grouped by episodes."""
        path_list = get_all_file_names("/home/hongyi/Codes/flower_rss24/aloha_sim_insertion_human/data/chunk-000")
        for path in path_list:
            data = process_episode_data(path)
            yield path, data

def get_video_data(root, ep_idx, time_stamps):
    video_path = 'videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4'
    root = "/home/hongyi/Codes/flower_rss24/aloha_sim_insertion_human"
    video_path = video_path.format(episode_index=ep_idx, video_key='observation.images.top')
    video_path = root + '/' + video_path
    frames = decode_video_frames_torchvision(video_path, time_stamps)
    # frames = frames * 255
    # frames = frames.numpy().astype(np.uint8)
    frames = frames.swapaxes(1, -1)
    return frames.numpy()


def process_episode_data(file_path):
    """
    Reads a .parquet file and converts it into a NumPy array.

    Parameters:
        file_path (str): The path to the .parquet file.

    Returns:
        numpy.ndarray: A NumPy array containing the data from the .parquet file.
    """
    try:
        # Read the parquet file into a pandas DataFrame
        df = pd.read_parquet(file_path)
        steps = []
        video_frames = get_video_data(Path(file_path).parent, df['episode_index'][0], df['timestamp'].to_list())
        for step_idx in range(len(df)):
            steps.append({
                'is_first': step_idx == 0,
                'is_last': step_idx == len(df) - 1,
                'observation': {
                    'state': df['observation.state'][step_idx],
                    'images_top': video_frames[step_idx]
                },
                'action': df['action'][step_idx],
                'reward': 0.0,
                'language_instruction': "Insert the peg into the socket.",
                'is_terminal': df['next.done'][step_idx],
                'discount': 1.0,
                'timestamp': df['timestamp'][step_idx],
                'frame_index': df['frame_index'][step_idx],
                'metadata': {'episode_index': df['episode_index'][step_idx]}
            })
        return {'steps': steps, 'episode_metadata': {'episode_id': df['episode_index'][0]}}

    except Exception as e:
        print(f"An error occurred while reading the parquet file: {e}")
        return None


def get_all_file_names(path):
    """
    Get all file names under a specified path.

    Args:
        path (str): The directory path to search.

    Returns:
        List[str]: A list of file names under the specified path.
    """
    file_names = []
    for root, dirs, files in os.walk(path):  # Traverse the directory
        for file in files:  # Iterate over files in the current directory
            file_names.append(os.path.join(root, file))  # Add full file path
    return file_names


if __name__ == "__main__":
    # Load the dataset
    ds = tfds.load("aloha_sim_insertion")

    for example in ds['train']:
        for step in example['steps']:
            print(step['observation']['images_top'].shape)
        break
    #
    # print(ds)
