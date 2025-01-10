import time

import numpy as np
import pandas as pd
import tensorflow as tf
# from datasets import load_dataset
# from collections import defaultdict
import tensorflow_datasets as tfds

import glob
import os

from tqdm import tqdm

import torch
import torchvision
from pathlib import Path

import logging


class CalvinDebug(tfds.core.GeneratorBasedBuilder):
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
        self.root = "/home/hongyi/Codes/flower_rss24/calvin/dataset/calvin_debug_dataset"
        self.training_dataset_path = self.root + "/training"
        self.validation_dataset_path = self.root + "/validation"
        super().__init__()

    def _info(self):
        """Define dataset information and features."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset(
                    {
                        'is_first': tf.bool,
                        'is_last': tf.bool,
                        'observation': tfds.features.FeaturesDict({
                            'state': tfds.features.Tensor(shape=(15,), dtype=tf.float64),
                            'rgb_static': tfds.features.Image(shape=(200, 200, 3), dtype=np.uint8),
                            'rgb_gripper': tfds.features.Image(shape=(84, 84, 3), dtype=np.uint8),
                        }),
                        'action': tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                        'reward': tfds.features.Tensor(shape=(), dtype=tf.float32),
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
            'train': self._generate_examples(self.training_dataset_path),
            'validation': self._generate_examples(self.validation_dataset_path),
        }

    def _generate_examples(self, dataset_path):
        """Yield examples grouped by episodes."""
        f_lang_ann = dataset_path + "/lang_annotations/auto_lang_ann.npy"
        f = np.load(f_lang_ann, allow_pickle=True)
        lang = f.item()['language']['ann']
        lang = np.array([x.encode('utf-8') for x in lang])
        lang_start_end_idx = f.item()['info']['indx']
        num_ep = len(lang_start_end_idx)
        with tqdm(total=num_ep) as pbar:
            for episode_idx, (start_idx, end_idx) in enumerate(lang_start_end_idx):
                pbar.update(1)
                episode = process_episode_data(
                    root_dir=dataset_path,
                    episode_idx=episode_idx,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    lang_instruction=lang[episode_idx]
                )
                yield episode_idx, episode


def process_episode_data(root_dir, episode_idx, start_idx, end_idx, lang_instruction):
    """
    Reads a .parquet file and converts it into a NumPy array.

    Parameters:
        file_path (str): The path to the .parquet file.

    Returns:
        numpy.ndarray: A NumPy array containing the data from the .parquet file.
    """
    steps = []
    step_files = [
        f"episode_{str(i).zfill(7)}.npz"
        for i in range(start_idx, end_idx + 1)
    ]

    for file in step_files:
        filepath = os.path.join(root_dir, file)
        f = np.load(filepath)
        steps.append({
            'is_first': False,
            'is_last': False,
            'observation': {
                'state': f['robot_obs'],
                'rgb_static': f['rgb_static'],
                'rgb_gripper': f['rgb_gripper'],
            },
            'action': f['rel_actions'],
            'reward': 0.0,
            'language_instruction': lang_instruction,
            'is_terminal': False,
            'discount': 1.0,
            'metadata': {'episode_index': episode_idx}
        })

    return {'steps': steps, 'episode_metadata': {'episode_id': episode_idx}}


def process_data_test():
    f = np.load("/home/hongyi/Codes/flower_rss24/calvin/dataset/calvin_debug_dataset/training/lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    lang = f.item()['language']['ann']
    lang = np.array([x.encode('utf-8') for x in lang])
    lang_start_end_idx = f.item()['info']['indx']
    num_ep = len(lang_start_end_idx)
    with tqdm(total=num_ep) as pbar:
        for episode_idx, (start_idx, end_idx) in enumerate(lang_start_end_idx):
            pbar.update(1)
            episode = process_episode_data(
                root_dir="/home/hongyi/Codes/flower_rss24/calvin/dataset/calvin_debug_dataset/training",
                episode_idx=episode_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                lang_instruction=lang[episode_idx]
            )
            print(episode)




if __name__ == "__main__":
    # process_data_test()
    # Load the dataset
    ds = tfds.load("calvin_debug", split="train")
    #
    for example in ds['train']:
        for step in example['steps']:
            print(step['observation']['images_top'].shape)
        break
    #
    # print(ds)
