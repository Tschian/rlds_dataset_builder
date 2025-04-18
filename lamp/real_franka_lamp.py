import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path
import pickle


class RealFrankaLamp(tfds.core.GeneratorBasedBuilder):
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
        dataset_name = "real_franka_lamp"
        self.dataset_name = dataset_name
        self.raw_data_path = "/home/hongyi/DATA/real_franka_datasets/mix_machine_new"
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
                            'state': tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                            'orb_0': tfds.features.Image(shape=(180, 320, 3), dtype=np.uint8),
                            'orb_1': tfds.features.Image(shape=(180, 320, 3), dtype=np.uint8)
                        }),
                        'action': tfds.features.Tensor(shape=(8,), dtype=tf.float32),
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
        path_list = get_all_file_names(self.raw_data_path)
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
        episode = pickle.load(open(file_path, 'rb'))
        episode_data = episode['data']

        steps = []

        for idx in range(len(episode_data) - 1):
            state = episode_data[idx]['follower_robot']['joint_pos'].cpu().numpy()
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


def get_all_file_names(path):
    """
    Get all file names under a specified path.

    Args:
        path (str): The directory path to search.

    Returns:
        List[str]: A list of file names under the specified path.
    """

    # Path to the folder you want to search in
    folder_path = Path(path).resolve()  # resolve() makes the path absolute

    # Get all .pkl files in the specified folder
    pkl_files = [str(file.absolute()) for file in folder_path.glob('*.pkl')]

    return pkl_files


if __name__ == "__main__":
    folder_path = "/home/hongyi/DATA/real_franka_datasets/mix_machine_new"

    pkl_files = get_all_file_names(folder_path)

    # print(pkl_files)

    episode_data = process_episode_data(pkl_files[0])
    print(episode_data)
    # Load the dataset
    # ds = tfds.load("real_franka_fold")

    # for episode in ds['train']:
    #     for step in episode['steps']:
    #         print(step)
    #     break