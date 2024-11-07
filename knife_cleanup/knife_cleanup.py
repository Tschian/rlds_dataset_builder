import os
import cv2
from typing import Iterator, Tuple, Any
from scipy.spatial.transform import Rotation

import glob
import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tqdm import tqdm
import re

tf.config.set_visible_devices([], "GPU")
data_path = "/home/kkuryshev/audio-pipeline/final_data/knife_cleanup"

class KnifeCleanup(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_side_cam': tfds.features.Image( #OAK-D LITE
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Side camera (OAK-D Lite) RGB observation.',
                        ),
                        'image_top_cam': tfds.features.Image( #OAK-D LITE
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Top camera (OAK-D Lite) RGB observation.',
                        ),
                        'image_wrist_cam': tfds.features.Image( #OAK-D SR
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera (OAK-D SR) RGB observation.',
                        ),
                        'image_front_cam': tfds.features.Image( #Realsense
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Front camera (RealSense) RGB observation.',
                        ),
                        'joint_state_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint state. Consists of [7x joint states]',
                        ),
                        'joint_state_vel': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint velocities. Consists of [7x joint velocities]',
                        ),
                        'end_effector_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Current End Effector position in Cartesian space',
                        ),
                        'end_effector_vel': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Current End Effector orientation in Cartesian space as Euler (xyz)',
                        ),
                        'gripper_state': tfds.features.Scalar(
                            dtype=np.float32,
                            doc='Desired gripper width, consists of [1x gripper width] in range [0, 1]',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Delta robot action, consists of [3x delta_end_effector_pos, '
                            '3x delta_end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'action_joint': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Absolute robot action, consists of [3x delta_end_effector_pos, '
                            '3x delta_end_effector_ori (euler: roll, pitch, yaw), 1x des_gripper_width].',
                    ),
                    'action_joint_state': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action in joint space, consists of [7x joint states]',
                    ),
                    'action_joint_vel': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action in joint space, consists of [7x joint velocities]',
                    ),
                    'action_ee_pos': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Delta robot action in joint space, consists of [7x joint states]',
                    ),
                    'action_ee_vel': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Delta robot action in joint space, consists of [7x joint states]',
                    ),
                    'action_gripper_width': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Desired gripper width, consists of [1x gripper width] in range [0, 1]',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(1, 512),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.',
                    ),
                    'traj_length': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Number of samples in trajectorie'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=data_path),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # create list of all examples
        raw_dirs = []
        get_trajectorie_paths_recursive(data_path, raw_dirs)
        print("# of trajectories:", len(raw_dirs))
        
        # for smallish datasets, use single-thread parsing
        for sample in raw_dirs:
            yield _parse_example(sample, self._embed)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

def _parse_example(episode_path, embed=None):
    data = {}
    
    for data_field in os.listdir(episode_path):
        data_field_full_path = os.path.join(episode_path, data_field)
        if os.path.isdir(data_field_full_path) and data_field == "images":
            # load images
            for image_dir in os.listdir(data_field_full_path):
                image_dir_full_path = os.path.join(data_field_full_path, image_dir)
                #print(f"loading: {image_dir_full_path}")
                cam1_image_vector = create_img_vector(image_dir_full_path)
                data.update({image_dir: cam1_image_vector})
        elif os.path.isdir(data_field_full_path) and data_field == "201 leader":
            # load leader data
            for leader_vec in os.listdir(data_field_full_path):
                leader_vec_full_path = os.path.join(data_field_full_path, leader_vec)
                #print(f"loading: {leader_vec_full_path}")
                data.update({f"leader_{leader_vec[:leader_vec.find('.')]}": torch.load(leader_vec_full_path).numpy()})
        elif os.path.isdir(data_field_full_path) and data_field == "202 follower":
            # load follower data
            for follower_vec in os.listdir(data_field_full_path):
                follower_vec_full_path = os.path.join(data_field_full_path, follower_vec)
                #print(f"loading: {follower_vec_full_path}")
                data.update({f"follower_{follower_vec[:follower_vec.find('.')]}": torch.load(follower_vec_full_path).numpy()})

    # print(data.keys())
    trajectory_length = len(data["follower_joint_pos"]) #if len(data["follower_joint_pos"]) < len(data["GoPro"]) else len(data["GoPro"])
    # print("traj_len:", len(data["follower_joint_pos"]))
    # print("GoPro len:", len(data["GoPro"]))

    episode = []
    for i in range(trajectory_length):
        # compute Kona language embedding
        language_embedding = [np.zeros(512)]
        action = np.append(data['leader_ee_pos'][i], data['leader_gripper_state'][i])
        action_joint = np.append(data['leader_joint_pos'][i], data['leader_gripper_state'][i])

        episode.append({
            'observation': {
                'image_top_cam': data['top_cam'][i],
                'image_side_cam': data['side_cam'][i],
                'image_wrist_cam': data['wrist_cam'][i],
                'image_front_cam': data['front_cam'][i],
                'joint_state_pos': data['follower_joint_pos'][i],
                'joint_state_vel': data['follower_joint_vel'][i],
                'end_effector_pos': data['follower_ee_pos'][i],
                'end_effector_vel': data['follower_ee_vel'][i],
                'gripper_state': data['follower_gripper_state'][i]
            },
            'action': action,
            'action_joint': action_joint,
            'action_joint_state': data['leader_joint_pos'][i],
            'action_joint_vel': data['leader_joint_vel'][i],
            'action_ee_pos': data['leader_ee_pos'][i],
            'action_ee_vel': data['leader_ee_vel'][i],
            'action_gripper_width': data['leader_gripper_state'][i],
            'discount': 1.0,
            'reward': float(i == (trajectory_length - 1)),
            'is_first': i == 0,
            'is_last': i == (trajectory_length - 1),
            'is_terminal': i == (trajectory_length - 1),
            'language_instruction': "Pick up the knife and put it in the box.",
            'language_embedding': language_embedding,
        })

    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path,
            'traj_length': trajectory_length,
        }
    }

    # if you want to skip an example for whatever reason, simply return None
    return episode_path, sample

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def create_img_vector(img_folder_path):
    cam_list = []
    cam_path_list = []
    dir_list_sorted = sorted_alphanumeric(os.listdir(img_folder_path))
    for img_name in dir_list_sorted:
        ext = img_name[img_name.find("."):]
        if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
            cam_path_list.append(img_name)
            img_path = os.path.join(img_folder_path, img_name)
            img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
            cam_list.append(img_array)
    return cam_list

def get_trajectorie_paths_recursive(directory, sub_dir_list):
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            sub_dir_list.append(directory) if entry == "images" else get_trajectorie_paths_recursive(full_path, sub_dir_list)
    # return subdirectories

if __name__ == "__main__":
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    # create list of all examples
    raw_dirs = []
    get_trajectorie_paths_recursive(data_path, raw_dirs)
    for trajectorie_path in tqdm(raw_dirs):
        #print(f"entering: {trajectorie_path}")
        _, sample = _parse_example(trajectorie_path, embed)
        #print(sample)