import time

import numpy as np
import pandas as pd
import tensorflow as tf
# from datasets import load_dataset
# from collections import defaultdict
import tensorflow_datasets as tfds

import glob
import os

import h5py
import torch
import torchvision
from pathlib import Path


def convert_hdf5_to_rlds(file_path):
    """
    Converts a single HDF5 file to RLDS format

    Args:
        file_path: Path to the HDF5 file

    Returns:
        Dictionary with RLDS formatted episode data
    """
    with h5py.File(file_path, 'r') as f:
        # NOTE: Adjust these paths based on your actual HDF5 structure
        # Uncomment the following lines to help explore your HDF5 structure
        # def print_struct(name, obj):
        #     print(name, type(obj))
        # f.visititems(print_struct)

        # Adjust these paths based on your actual HDF5 structure
        observations = f['observations'][:]  # Modify path as needed
        actions = f['actions'][:]  # Modify path as needed
        rewards = f['rewards'][:]  # Modify path as needed

        # Default discount factor
        discounts = np.ones_like(rewards) * 0.99

        # Create steps list
        steps = []
        for i in range(len(observations) - 1):  # Typical RL dataset has n observations and n-1 actions
            step = {
                rlds_types.OBSERVATION: observations[i],
                rlds_types.ACTION: actions[i],
                rlds_types.REWARD: rewards[i],
                rlds_types.DISCOUNT: discounts[i],
                rlds_types.STEP_TYPE: 1,  # Middle step
            }
            steps.append(step)

        # Mark first and last steps
        if steps:
            steps[0][rlds_types.STEP_TYPE] = 0  # First step
            steps[-1][rlds_types.STEP_TYPE] = 2  # Last step

        return {rlds_types.STEPS: steps}


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
