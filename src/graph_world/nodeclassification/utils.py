# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import copy
import dataclasses
import enum
import math
from typing import Dict, List, Optional, Sequence, Tuple
import random

import graph_tool
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

import networkx as nx


@dataclasses.dataclass
class NodeClassificationDataset:
  """Stores data for node classification tasks.
  Attributes:
    graph: graph-tool Graph object.
    graph_memberships: list of integer node classes.
    node_features: numpy array of node features.
    feature_memberships: list of integer node feature classes.
    edge_features: map from edge tuple to numpy array. Only stores undirected
      edges, i.e. (0, 1) will be in the map, but (1, 0) will not be.
  """
  graph: graph_tool.Graph = Ellipsis
  graph_memberships: np.ndarray = Ellipsis
  node_features: np.ndarray = Ellipsis
  feature_memberships: np.ndarray = Ellipsis
  edge_features: Dict[Tuple[int, int], np.ndarray] = Ellipsis


def nodeclassification_data_to_torchgeo_data(
    nodeclassification_data: NodeClassificationDataset) -> Data:
  edge_tuples = []
  edge_feature_data = []
  for edge in nodeclassification_data.graph.iter_edges():
    edge_tuples.append([edge[0], edge[1]])
    edge_tuples.append([edge[1], edge[0]])
    ordered_tuple = (edge[0], edge[1])
    if edge[0] > edge[1]:
      ordered_tuple = (edge[1], edge[0])
    edge_feature_data.append(
        nodeclassification_data.edge_features[ordered_tuple])
    edge_feature_data.append(
        nodeclassification_data.edge_features[ordered_tuple])

  node_features = torch.tensor(nodeclassification_data.node_features,
                               dtype=torch.float)
  edge_index = torch.tensor(edge_tuples, dtype=torch.long)
  edge_feature_data_np = np.array(edge_feature_data)
  edge_attr = torch.from_numpy(edge_feature_data_np).float()
  labels = torch.tensor(nodeclassification_data.graph_memberships,
                        dtype=torch.long)
  return Data(x=node_features, edge_index=edge_index.t().contiguous(),
              edge_attr=edge_attr, y=labels)


def sample_kclass_train_sets(example_indices: List[int],
                             k_train: int, k_val: int) -> \
    Tuple[List[int], List[int], List[int]]:
  # Helper function
  def get_num_train_val(n_elements, p_train):
    num_train = round(n_elements * p_train)
    num_val = round(n_elements * (1 - p_train))
    if num_train == 0:
      num_train = 1
      num_val = n_elements - 1
    if num_train == n_elements:
      num_train = num_elements - 1
      n_val = 1
    return num_train, num_val

  # If the class has less than 2 elements, throw an error.
  if len(example_indices) < 2:
    raise ValueError("attempted to sample k-from-class on class with size < 2.")
  # If the class has exactly 2 elements, assign one to train / test randomly.
  elif len(example_indices) == 2:
    train_index = random.choice([0, 1])
    test_index = 1 - train_index
    return ([example_indices[train_index]],
            [], [example_indices[test_index]])
  # If the class has less than k_train + k_val + 1 elements, assign 1
  # element to test, and split the rest proportionally.
  elif len(example_indices) < k_train + k_val + 1:
    train_proportion = k_train / (k_val + k_train)
    num_test = 1
    num_train, num_val = get_num_train_val(len(example_indices) - 1,
                                           train_proportion)
  else:
    num_train = k_train
    num_val = k_val
    num_test = len(example_indices) - (num_train + num_val)

  assert num_train + num_val + num_test == len(example_indices)

  # Do sampling
  random_indices = copy.deepcopy(example_indices)
  random.shuffle(random_indices)
  return (random_indices[:num_train],
          random_indices[num_train:(num_train + num_val)],
          random_indices[(num_train + num_val):])


def get_kclass_masks(nodeclassification_data: NodeClassificationDataset,
                     k_train: int = 30, k_val: int = 20) -> \
    Tuple[List[int], List[int], List[int]]:
  # Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # Get graph ground-truth clusters.
  clusters = collections.defaultdict(list)
  for node_index, cluster_index in enumerate(
      nodeclassification_data.graph_memberships):
    clusters[cluster_index].append(node_index)
  # Sample train and val sets.
  training_mask = [False] * len(nodeclassification_data.graph_memberships)
  validate_mask = [False] * len(nodeclassification_data.graph_memberships)
  test_mask = [False] * len(nodeclassification_data.graph_memberships)
  for cluster_index, cluster in clusters.items():
    cluster_train_set, cluster_val_set, cluster_test_set = \
      sample_kclass_train_sets(cluster, k_train, k_val)
    for index in cluster_train_set:
      training_mask[index] = True
    for index in cluster_val_set:
      validate_mask[index] = True
    for index in cluster_test_set:
      test_mask[index] = True
  # return (training_mask,
  #         validate_mask,
  #         test_mask)
  return (torch.as_tensor(training_mask).reshape(-1),
          torch.as_tensor(validate_mask).reshape(-1),
          torch.as_tensor(test_mask).reshape(-1))

def get_label_masks(
    y: torch.Tensor,
    num_train_per_class: int = 5, # Adjusted default for smaller graphs
    num_val: int = 50,           # Adjusted default for smaller graphs
    num_min_test: int = 10,      # Minimum nodes required for the test set
    random_seed: int = 12345
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates train, validation, and test masks for semi-supervised learning.

    - Training set: Fixed number of nodes per class.
    - Validation set: Fixed total number of nodes, sampled proportionally
                      from the class distribution remaining after training nodes
                      are removed.
    - Test set: All remaining nodes.

    Args:
        y: Tensor containing the labels for all nodes.
        num_train_per_class: The number of training nodes to select for each class.
        num_val: The total number of validation nodes desired.
        num_min_test: The minimum number of nodes required to be left for the test set.
        random_seed: Seed for the random number generator.

    Returns:
        A tuple containing boolean tensors for train_mask, val_mask, test_mask.

    Raises:
        RuntimeError: If constraints (e.g., not enough nodes) cannot be met.
    """
    random_gen = torch.Generator()
    random_gen.manual_seed(random_seed)
    num_samples = y.size(0)
    classes = torch.unique(y) # More reliable than set(y.numpy())
    num_classes = classes.size(0)

    train_mask = torch.zeros(num_samples, dtype=bool)
    val_mask = torch.zeros(num_samples, dtype=bool)
    test_mask = torch.zeros(num_samples, dtype=bool)

    # --- 1. Select Training Nodes (Fixed per Class) ---
    nodes_used_in_train = 0
    for c in classes:
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        num_nodes_in_class = idx.size(0)
        num_train_for_this_class = min(num_nodes_in_class, num_train_per_class)
        if num_train_for_this_class < num_train_per_class:
             print(f"Warning: Class {c.item()} has only {num_nodes_in_class} nodes, using {num_train_for_this_class} for training.")
        
        if num_train_for_this_class > 0:
            perm = torch.randperm(num_nodes_in_class, generator=random_gen)[:num_train_for_this_class]
            train_idx = idx[perm]
            train_mask[train_idx] = True
            nodes_used_in_train += num_train_for_this_class

    # --- 2. Identify Remaining Nodes ---
    remaining_indices = (~train_mask).nonzero(as_tuple=False).view(-1)
    num_remaining = remaining_indices.size(0)

    # --- 3. Check if Enough Nodes Remain ---
    if num_remaining < num_val + num_min_test:
        # Not enough nodes left for the desired validation size and a minimal test set.
        # Try reducing validation size.
        adjusted_num_val = num_remaining - num_min_test
        if adjusted_num_val < 1: # Still not enough even for 1 validation node + min_test
             raise RuntimeError(
                 f"Not enough nodes remaining ({num_remaining}) after training ({nodes_used_in_train}) "
                 f"to create validation and test sets (required min test: {num_min_test}). "
                 f"Graph size: {num_samples}, Classes: {num_classes}, Train/Class: {num_train_per_class}."
             )
        print(f"Warning: Reducing num_val from {num_val} to {adjusted_num_val} to ensure at least {num_min_test} test nodes.")
        num_val = adjusted_num_val # Use the adjusted number

    if num_val <= 0:
        print("Warning: No nodes allocated for validation set after adjustments.")
        # All remaining go to test set
        test_mask[remaining_indices] = True
        # Verify train set still has multiple classes if needed elsewhere
        # (e.g. some loss functions might require it)
        train_classes = torch.unique(y[train_mask])
        if len(train_classes) < 2:
             print(f"Warning: Training set only contains {len(train_classes)} class(es).")
        return train_mask, val_mask, test_mask # val_mask is all False

    # --- 4. Select Validation Nodes (Stratified Proportional) ---
    remaining_labels = y[remaining_indices]
    val_indices_list = []

    # Calculate class counts and proportions in the remaining set
    unique_remaining_classes, counts_in_remainder_tensor = torch.unique(remaining_labels, return_counts=True)
    counts_in_remainder = {c.item(): count.item() for c, count in zip(unique_remaining_classes, counts_in_remainder_tensor)}

    # Determine target number of validation samples per class
    target_val_float = {c: (counts_in_remainder[c] / num_remaining) * num_val for c in counts_in_remainder}
    final_val_counts = {c: round(target_val_float[c]) for c in counts_in_remainder}

    # Adjust rounding to match exact num_val
    current_val_total = sum(final_val_counts.values())
    diff = num_val - current_val_total

    # Sort classes by rounding remainder (descending) to add/remove nodes fairly
    remainder_tuples = sorted(
        [(c, target_val_float[c] - final_val_counts[c]) for c in counts_in_remainder],
        key=lambda item: item[1],
        reverse=True # Add to classes with largest positive remainder first
    )

    if diff > 0: # Need to add nodes
        for i in range(diff):
            class_to_increment = remainder_tuples[i % len(remainder_tuples)][0]
            # Ensure we don't take more than available for this class in remainder
            if final_val_counts[class_to_increment] < counts_in_remainder[class_to_increment]:
                final_val_counts[class_to_increment] += 1
            else:
                # If we can't increment the top choice, try the next one in the sorted list
                # This simple loop might not perfectly distribute if many classes are maxed out,
                # but is generally good enough. A more complex redistribution might be needed
                # for extreme edge cases. Search for next available class to increment.
                for j in range(1, len(remainder_tuples)):
                    next_class_idx = (i + j) % len(remainder_tuples)
                    class_to_increment = remainder_tuples[next_class_idx][0]
                    if final_val_counts[class_to_increment] < counts_in_remainder[class_to_increment]:
                         final_val_counts[class_to_increment] += 1
                         break # Found one
    elif diff < 0: # Need to remove nodes
         # Sort ascending by remainder (remove from classes rounded up most aggressively)
         remainder_tuples.sort(key=lambda item: item[1])
         for i in range(abs(diff)):
            class_to_decrement = remainder_tuples[i % len(remainder_tuples)][0]
            # Ensure we don't remove if count is already 0
            if final_val_counts[class_to_decrement] > 0:
                final_val_counts[class_to_decrement] -= 1
            else:
                 # Find next available class to decrement
                 for j in range(1, len(remainder_tuples)):
                     next_class_idx = (i + j) % len(remainder_tuples)
                     class_to_decrement = remainder_tuples[next_class_idx][0]
                     if final_val_counts[class_to_decrement] > 0:
                         final_val_counts[class_to_decrement] -= 1
                         break # Found one

    # Sample validation nodes for each class from the remaining nodes
    for c_item, count in final_val_counts.items():
        if count == 0:
            continue
        c = torch.tensor(c_item, device=y.device) # Ensure tensor on correct device
        # Find indices within 'remaining_indices' that belong to class 'c'
        class_mask_in_remaining = (remaining_labels == c)
        candidate_indices = remaining_indices[class_mask_in_remaining]

        num_available = candidate_indices.size(0)
        num_to_select = min(count, num_available) # Should be equal if logic above is sound

        if num_to_select > 0:
             perm = torch.randperm(num_available, generator=random_gen)[:num_to_select]
             val_idx_for_class = candidate_indices[perm]
             val_indices_list.append(val_idx_for_class)

    if val_indices_list:
        val_indices = torch.cat(val_indices_list)
        val_mask[val_indices] = True

    # --- 5. Select Test Nodes ---
    # Test nodes are everything not in train or validation
    test_mask = ~train_mask & ~val_mask

    # --- 6. Final Checks (Optional but Recommended) ---
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == num_samples
    assert not torch.logical_and(train_mask, val_mask).any()
    assert not torch.logical_and(train_mask, test_mask).any()
    assert not torch.logical_and(val_mask, test_mask).any()
    if val_mask.sum() > 0 and torch.unique(y[val_mask]).numel() < 2 :
         print(f"Warning: Stratified validation resulted in only one class (Class: {torch.unique(y[val_mask]).item()}). This might happen if remaining nodes are heavily skewed or num_val is very small.")
    if test_mask.sum() < num_min_test:
         print(f"Warning: Final test set size ({test_mask.sum()}) is less than desired minimum ({num_min_test}).")


    return train_mask, val_mask, test_mask