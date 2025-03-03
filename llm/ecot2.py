import os
import pickle
from typing import Callable, Optional, Dict

import tensorflow_datasets as tfds
import dlimp as dl
import inspect
import json

from PIL import Image
from google.protobuf.internal.well_known_types import Any
from huggingface_hub import hf_hub_download
import shutil

from typing import Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import enum


def tree_map(fn: Callable, tree: Dict) -> Dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


class NormalizationType(str, enum.Enum):
    # fmt: off
    NORMAL = "normal"  # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"  # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"  # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_cot_database_keys():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }


def map_key(
        dictionary: dict,
        key_map: Callable[[str], str]) -> dict:
    """
    Maps the keys of a dictionary to new keys.
    """
    return {key_map(key): value for key, value in dictionary.items()}


name = "bridge:1.0.0"
data_dir = "/root"
builder = tfds.builder(name, data_dir=data_dir)

with open(os.path.join(data_dir, "bridge", "1.0.0",
                       "dataset_statistics_2797227d870df7bc581a43ae655e8a273853f21c9ec3bb90c4527453359c5bf9.json"),
          "r") as f:
    dataset_statistics = json.load(f)

dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=False, num_parallel_reads=32)

download_path = hf_hub_download(
    repo_id="Embodied-CoT/embodied_features_bridge",
    filename="embodied_features_bridge.json",
    repo_type="dataset",
    local_dir=os.path.join(data_dir, "bridge", "1.0.0"),
    local_dir_use_symlinks=False  # This forces a copy instead of a symlink
)

# Now try reading it directly
with open(os.path.join(data_dir, "bridge", "1.0.0", "embodied_features_bridge.json"), "r") as f:
    reasoning_dataset = json.load(f)


def make_dataset_from_rlds(
        name: str,
        data_dir: str,
        *,
        train: bool,
        standardize_fn: Optional[Callable[[dict], dict]] = None,
        shuffle: bool = True,
        image_obs_keys: Dict[str, Optional[str]] = {},
        depth_obs_keys: Dict[str, Optional[str]] = {},
        state_obs_keys: List[Optional[str]] = (),
        language_key: Optional[str] = None,
        dataset_statistics: Optional[Union[dict, str]] = None,
        absolute_action_mask: Optional[List[bool]] = None,
        num_parallel_reads: int = tf.data.AUTOTUNE,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        reasoning_dataset_path: str = "~/.cache/reasonings_dataset.json",
) -> Tuple[dl.DLataset, dict]:
    """
    This function is responsible for loading a specific RLDS dataset from storage and getting it into a standardized
    format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the trajectory
    into a standard format, which includes the keys "observation" and "action". Entry "observation" should be a
    dictionary containing some number of additional keys, which will be extracted into an even more standardized format
    according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in place of an
    old name to insert padding. For example, if after `standardize_fn`, your "observation" dict has RGB images called
    "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary": None, "wrist": "wrist"}`, then
    the resulting dataset will have an "observation" dict containing the keys "image_primary", "image_secondary", and
    "image_wrist", where "image_primary" corresponds to "workspace", "image_secondary" is a padding image, and
    "image_wrist" corresponds to "wrist".

    Entry `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which will
    be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be inserted for each
    None entry.

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will contain the
    key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset, since one
            file usually contains many trajectories)!
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to extract from the
            "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in image_obs_keys.items()}`.
            If a value of `old` is None, inserts a padding image instead (empty string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from the
            "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding for each None entry.
        language_key (str, optional): If provided, the "task" dict will contain the key "language_instruction",
            extracted from `traj[language_key]`.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        absolute_action_mask (Sequence[bool], optional): By default, all action dimensions are assumed to be
            relative. This is important for when `future_action_window_size > 0`: actions that are taken
            from beyond the end of the trajectory (or beyond the goal timestep when goal relabeling is used)
            need to be made "neutral" to indicate that the task has been completed. For relative actions,
            "neutral" means zero, but for absolute actions, "neutral" means repeating the last valid action.
            This mask, if provided, indicates which action dimensions are absolute.
        action_normalization_mask (Sequence[bool], optional): If provided, indicates which action dimensions
            should be normalized. For example, you might not want to normalize the gripper action dimension if
            it's always exactly 0 or 1. By default, all action dimensions are normalized.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    if os.path.isfile(reasoning_dataset_path):
        print(f"Loading from local checkpoint path `{reasoning_dataset_path}`.")
    else:
        print(f"Dataset file `{reasoning_dataset_path}` not found, loading from HF.")

        download_path = hf_hub_download(
            repo_id="Embodied-CoT/embodied_features_bridge",
            filename="embodied_features_bridge.json",
            repo_type="dataset",
        )

        shutil.copyfile(download_path, reasoning_dataset_path)

    with open(reasoning_dataset_path, "r") as f:
        reasoning_dataset = json.load(f)

    def make_tf_dict(raw_dict):
        print("Building the reasoning dict...")
        keys = []
        values = []
        move = []

        def reasoning_dict_to_str(d):
            tags = get_cot_tags_list()[:-1]  # exclude ACTION
            database_keys = get_cot_database_keys()
            reasoning_parts = [(tag, d[database_keys[tag]]) for tag in tags]

            return "@".join(f"{tag}@{part}" for tag, part in reasoning_parts)

        has_reasoning = [0, 0]

        for file_name in raw_dict.keys():
            for episode_id in raw_dict[file_name].keys():
                if "reasoning" not in raw_dict[file_name][episode_id].keys():
                    has_reasoning[0] += 1
                    continue
                else:
                    has_reasoning[1] += 1

                for i in raw_dict[file_name][episode_id]["reasoning"].keys():
                    keys.append(file_name + "_" + str(episode_id) + "_" + i)
                    reasoning_dict = raw_dict[file_name][episode_id]["reasoning"][i]

                    gripper_lookahead_n = 5  # list this many future positions of the gripper
                    trajectory_features = raw_dict[file_name][episode_id]["features"]

                    reasoning_dict["gripper"] = ""
                    if "gripper_position" in trajectory_features.keys():
                        if trajectory_features["gripper_position"] is not None:
                            # 0 1 2 3 4
                            if 0 <= int(i) < len(trajectory_features["gripper_position"]):
                                future_positions = []
                                for j in range(gripper_lookahead_n):
                                    if int(i) + j < len(trajectory_features["gripper_position"]):
                                        future_positions += trajectory_features["gripper_position"][int(i) + j]
                                    else:
                                        future_positions += future_positions[-2:]

                                reasoning_dict["gripper"] = str(future_positions)

                    reasoning_dict["bboxes"] = ""
                    if "bboxes" in trajectory_features.keys():
                        if trajectory_features["bboxes"] is not None:
                            if 0 <= int(i) < len(trajectory_features["bboxes"]):
                                if len(trajectory_features["bboxes"][int(i)]) > 0:
                                    boxes_list = trajectory_features["bboxes"][int(i)]
                                    reasoning_dict["bboxes"] = ", ".join(
                                        [f"{name} {box!s}" for prob, name, box in boxes_list]
                                    )

                    values.append(reasoning_dict_to_str(reasoning_dict))
                    move.append(reasoning_dict["move"])

        print("Example reasoning:", keys[0], values[0])
        print("Reasoning presence statistics [# has not, # has]:", has_reasoning)

        return (
            tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=""),
            tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, move), default_value=""),
        )

    reasoning_dataset, gripper_dataset = make_tf_dict(reasoning_dataset)

    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    (
                        tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                        if key is None
                        else tf.cast(old_obs[key], tf.float32)
                    )
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, " "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)

        file_name = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_id = traj["traj_metadata"]["episode_metadata"]["episode_id"][0]

        file_names = tf.repeat(file_name, traj_len)
        episode_ids = tf.as_string(tf.repeat(episode_id, traj_len))
        indices = tf.as_string(tf.range(traj_len))
        reasonings = reasoning_dataset.lookup(file_names + "_" + episode_ids + "_" + indices)
        gripper = gripper_dataset.lookup(file_names + "_" + episode_ids + "_" + indices)

        traj = {
            "file_name": file_name,
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
            "reasoning": reasonings,
            "move": gripper,
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    builder = tfds.builder(name, data_dir=data_dir)

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)

    # construct the dataset
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)
    dataset = dataset.traj_map(restructure, num_parallel_calls)

    return dataset, dataset_statistics


dataset, dataset_statistics = make_dataset_from_rlds(
    name="bridge:1.0.0",
    data_dir="/root",
    train=True,
    standardize_fn=None,
    shuffle=False,
    image_obs_keys={'image_0': 'image_0'},
    depth_obs_keys={},
    state_obs_keys=[],
    language_key=None,
    dataset_statistics=dataset_statistics,
    absolute_action_mask=None,
    num_parallel_reads=32,
    num_parallel_calls=32,
    reasoning_dataset_path="/root/bridge/1.0.0/embodied_features_bridge.json",
)

import os
import pickle
import lmdb
from tqdm import tqdm

def create_lmdb_dataset_for_repeated(
    dataset,
    output_dir="/root/RSP/demo/lmdb_dataset",
    map_size=300 * 1024**3  # Aim for ~300GB to account for some overhead
):
    """
    Create an LMDB dataset in which each sample is one entire trajectory.
    This allows for repeated sampling at read time.

    Parameters:
      dataset: an iterable (e.g., TF Dataset or similar) producing one trajectory at a time.
               Each "traj" item is assumed to have:
                 - "observation"]["image_image_0"] => an array/list of raw image bytes
                 - "move" => an array/list of move strings (or bytes)
      output_dir: where to place the LMDB file
      map_size: maximum size database may grow to; used to prevent map full errors.
    """

    os.makedirs(output_dir, exist_ok=True)
    lmdb_path = os.path.join(output_dir, "trajectories.lmdb")

    # Open LMDB environment with a large map size
    env = lmdb.open(lmdb_path, map_size=map_size, writemap=True)

    with env.begin(write=True) as txn:
        sample_idx = 0
        traj_idx = 0

        for traj in tqdm(dataset, desc="Creating LMDB dataset"):
            # 'observation["image_image_0"]' is a tensor of shape [T], each entry raw image bytes
            observation = traj["observation"]["image_image_0"].numpy()  # shape (T,)
            # 'move' is similarly a tensor of shape [T], each entry presumably a bytes-like object
            move = traj["move"].numpy()

            # Convert file_name from TF byte-string to str
            traj_name = traj["file_name"].numpy().decode("utf-8")
            # Clean the name for file usage
            traj_name = traj_name.replace("/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/", "")
            traj_name = traj_name.replace("/train/out.npy", "")
            traj_name = traj_name.replace("/", "_")

            # Convert observation and move data to Python lists for pickling
            image_list = list(observation)
            move_list = [m.decode("utf-8") for m in move]  

            # Serialize data
            images_pkl = pickle.dumps(image_list)
            moves_pkl = pickle.dumps(move_list)

            # Construct a single entry for the entire trajectory
            sample_key = f"traj{traj_idx:06d}_{traj_name}".encode("ascii")
            sample_value = pickle.dumps({'images': images_pkl, 'moves': moves_pkl})

            # Store in LMDB using a transaction for performance
            txn.put(sample_key, sample_value)

            sample_idx += 1
            traj_idx += 1

    print(f"Created {sample_idx} trajectory-level samples in LMDB dataset at {lmdb_path}")

create_lmdb_dataset_for_repeated(dataset, output_dir="/root/RSP/demo/lmdb_dataset")