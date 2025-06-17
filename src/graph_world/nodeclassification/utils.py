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
  super_memberships: np.ndarray = None


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
    num_train_per_class: int = 5,
    num_val_per_class: int = 5,
    min_test: int = 10,
    random_seed: int = 12345
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns boolean masks (train_mask, val_mask, test_mask) for splitting indices.
    
    - Exactly num_train_per_class training examples per class.
    - Exactly num_val_per_class validation examples per class (if possible).
    - All other samples go to test (guaranteeing at least `min_test`).
    - Fully reproducible given random_seed.
    """
    device = y.device
    gen = torch.Generator(device=device).manual_seed(random_seed)
    classes = torch.unique(y)

    N = y.size(0)
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)

    for c in classes:
        idx_c = (y == c).nonzero(as_tuple=False).view(-1)
        if idx_c.numel() < num_train_per_class + num_val_per_class:
            raise ValueError(f"Class {c.item()} has only {idx_c.numel()} samples, "
                             f"but needs at least {num_train_per_class + num_val_per_class}.")

        perm = torch.randperm(idx_c.size(0), generator=gen)
        train_idx = idx_c[perm[:num_train_per_class]]
        val_idx = idx_c[perm[num_train_per_class:num_train_per_class + num_val_per_class]]

        train_mask[train_idx] = True
        val_mask[val_idx] = True

    test_mask = ~(train_mask | val_mask)

    # Ensure enough test samples
    if test_mask.sum().item() < min_test:
        raise RuntimeError(f"Only {test_mask.sum().item()} test samples available; "
                           f"minimum required is {min_test}.")

    return train_mask, val_mask, test_mask
