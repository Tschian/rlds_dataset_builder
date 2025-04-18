import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path
import h5py


class CoffeePressButton(tfds.core.GeneratorBasedBuilder):
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
        dataset_name = "coffee_press_button"
        self.dataset_name = dataset_name
        self.raw_data_path = "/home/temp_store/wang/CoffeePressButton/2024-04-25/processed_demo_128_128.hdf5"
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
                            'obj_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float64),
                            'obj_quat': tfds.features.Tensor(shape=(4,), dtype=tf.float64),
                            'obj_to_robot0_eef_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float64),
                            'obj_to_robot0_eef_quat': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                            'object-state': tfds.features.Tensor(shape=(14,), dtype=tf.float64),
                            'point_cloud': tfds.features.Tensor(shape=(49152, 6), dtype=tf.float64),
                            'robot0_agentview_left_depth': tfds.features.Tensor(shape=(128, 128, 1), dtype=tf.float32),
                            'robot0_agentview_right_depth': tfds.features.Tensor(shape=(128, 128, 1), dtype=tf.float32),
                            'robot0_base_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float64),
                            'robot0_base_quat': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                            'robot0_base_to_eef_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float64),
                            'robot0_base_to_eef_quat': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                            'robot0_base_to_eef_quat_site': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                            'robot0_eef_pos': tfds.features.Tensor(shape=(3,), dtype=tf.float64),
                            'robot0_eef_quat': tfds.features.Tensor(shape=(4,), dtype=tf.float64),
                            'robot0_eef_quat_site': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
                            'robot0_eye_in_hand_depth': tfds.features.Tensor(shape=(128, 128, 1), dtype=tf.float32),
                            'robot0_gripper_qpos': tfds.features.Tensor(shape=(2,), dtype=tf.float64),
                            'robot0_gripper_qvel': tfds.features.Tensor(shape=(2,), dtype=tf.float64),
                            'robot0_joint_pos': tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                            'robot0_joint_pos_cos': tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                            'robot0_joint_pos_sin': tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                            'robot0_joint_vel': tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                            'robot0_proprio-state': tfds.features.Tensor(shape=(61,), dtype=tf.float64),
                            'sampled_point_cloud': tfds.features.Tensor(shape=(1024, 6), dtype=tf.float64),
                            'segmented_pc_size': tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                            'segmented_point_cloud': tfds.features.Tensor(shape=(18009, 6), dtype=tf.float64),
                            'segmented_sampled_point_cloud': tfds.features.Tensor(shape=(1024, 6), dtype=tf.float64),
                            'segmented_uniform_sampled_point_cloud': tfds.features.Tensor(shape=(1024, 6), dtype=tf.float64),
                            'uniform_sampled_point_cloud': tfds.features.Tensor(shape=(1024, 6), dtype=tf.float64),
                        }),
                        'state': tfds.features.Tensor(shape=(178,), dtype=tf.float64),
                        'action': tfds.features.Tensor(shape=(12,), dtype=tf.float64),
                        'reward': tfds.features.Tensor(shape=(), dtype=tf.float64),
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
        episodes = process_episode_data(self.raw_data_path)
        for i in range(len(episodes)):
            yield self.raw_data_path, episodes[i]


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
        f = h5py.File(file_path, 'r')
        demo_keys_list = list(f["data"].keys())

        num_episodes = len(demo_keys_list)

        observations = data['obs']
        actions = data['actions']
        rewards = data['rewards']
        states = data['states']

        episodes = []





        steps = []
        for idx in range(num_episodes):
            state = states[idx].cpu().numpy()
            gripper_state = episode_data[idx]['follower_robot']['gripper_state']
            gripper_state = np.array([gripper_state], dtype=np.float32)
            proprioceptive_state = np.concatenate((state, gripper_state), axis=0)

            action = episode_data[idx + 1]['leader_robot']['joint_pos'].cpu().numpy()
            gripper_action = episode_data[idx + 1]['leader_robot']['gripper_state']
            gripper_action = np.array([gripper_action], dtype=np.float32)
            action = np.concatenate((action, gripper_action), axis=0)

            orb_0 = episode_data[idx]['ORB_0']['image']
            orb_1 = episode_data[idx]['ORB_1']['image']

            steps.append({
                'is_first': idx == 0,
                'is_last': idx == len(episode_data) - 2,
                'observation': {
                    'state': proprioceptive_state,
                    'orb_0': orb_0,
                    'orb_1': orb_1
                },
                'action': action,
                'reward': 0.0,
                'language_instruction': "Use the mixer.",
                'is_terminal': idx == len(episode_data) - 2,
                'discount': 1.0,
                'timestamp': idx,
                'frame_index': idx,
                'metadata': {'episode_index': 0}
            })
        return {'steps': steps, 'episode_metadata': {'episode_id': 0}}

    except Exception as e:
        print(f"An error occurred while reading the parquet file: {e}")
        return None


if __name__ == "__main__":
    hdf5_path = "/home/temp_store/wang/CoffeePressButton/2024-04-25/processed_demo_128_128.hdf5"

    episode_data = process_episode_data(hdf5_path)
