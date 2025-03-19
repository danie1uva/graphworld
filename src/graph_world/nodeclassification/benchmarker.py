import copy
import gin
import logging
import numpy as np
import graph_tool.all as gt
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import torch
import wandb  # Ensure wandb is imported

from ..models.models import PyGBasicGraphModel
from ..beam.benchmarker import Benchmarker, BenchmarkerWrapper


class NNNodeBenchmarker(Benchmarker):
  def __init__(self, generator_config, model_class, benchmark_params, h_params, torch_data):
    super().__init__(generator_config, model_class, benchmark_params, h_params, torch_data)
    # Remove meta entries from h_params.
    self._epochs = benchmark_params['epochs']
    self._model = model_class(**h_params)
    # TODO: make optimizer configurable.
    self._optimizer = torch.optim.Adam(self._model.parameters(),
                                       lr=benchmark_params['lr'],
                                       weight_decay=benchmark_params['weight_decay'])
    self._criterion = torch.nn.CrossEntropyLoss()
    self._train_mask = None
    self._val_mask = None
    self._test_mask = None

  def AdjustParams(self, generator_config, torch_data):
    if self._h_params is not None:
      # Adjust num_clusters to correct out_channels and update config in LFR graphs.
      if generator_config['generator_name'] == 'LFR':
        generator_config['num_clusters'] = len(set(torch_data.y.numpy()))
      self._h_params['out_channels'] = generator_config['num_clusters']

  def SetMasks(self, train_mask, val_mask, test_mask):
    self._train_mask = train_mask
    self._val_mask = val_mask
    self._test_mask = test_mask

  def train_step(self, data):
    self._model.train()
    self._optimizer.zero_grad()  # Clear gradients.
    out = self._model(data.x, data.edge_index)  # Forward pass.
    loss = self._criterion(out[self._train_mask],
                           data.y[self._train_mask])  # Loss on training nodes.
    loss.backward()  # Backpropagation.
    self._optimizer.step()  # Update parameters.
    return loss

  def test(self, data, test_on_val=False):
    self._model.eval()
    out = self._model(data.x, data.edge_index)
    # Apply softmax to obtain probabilities.
    out = torch.nn.functional.softmax(out, dim=-1)
      
    if test_on_val:
        pred = out[self._val_mask].detach().numpy()
    else:
        pred = out[self._test_mask].detach().numpy()

    pred_best = pred.argmax(-1)
    if test_on_val:
        correct = data.y[self._val_mask].numpy()
    else:
        correct = data.y[self._test_mask].numpy()
      
    n_classes = out.shape[-1]
    correct_onehot = np.eye(n_classes)[correct]

    try:
      rocauc_ovr = sklearn.metrics.roc_auc_score(correct_onehot, pred, multi_class='ovr')
      rocauc_ovo = sklearn.metrics.roc_auc_score(correct_onehot, pred, multi_class='ovo')
      logloss = sklearn.metrics.log_loss(correct_onehot, pred)
    except Exception as e:
      rocauc_ovr = float('nan')
      rocauc_ovo = float('nan')
      logloss = float('nan')

    results = {
      'accuracy': sklearn.metrics.accuracy_score(correct, pred_best),
      'f1_micro': sklearn.metrics.f1_score(correct, pred_best, average='micro'),
      'f1_macro': sklearn.metrics.f1_score(correct, pred_best, average='macro'),
      'rocauc_ovr': rocauc_ovr,
      'rocauc_ovo': rocauc_ovo,
      'logloss': logloss
    }
    return results 

  def train(self, data, tuning_metric: str, tuning_metric_is_loss: bool, early_stopping: int = 250):
    losses = []
    best_val_metric = np.inf if tuning_metric_is_loss else -np.inf
    test_metrics = None
    best_val_metrics = None
    early_stopping_counter = 0

    print(self._epochs)
    # Log epoch-level metrics.
    for i in range(self._epochs):
      epoch_loss = float(self.train_step(data))
      losses.append(epoch_loss)
      val_metrics = self.test(data, test_on_val=True)

      # Log per-epoch values to wandb.
      wandb.log({
        "epoch": i + 1,
        "train_loss": epoch_loss,
        "val_accuracy": val_metrics.get("accuracy"),
        "val_f1_micro": val_metrics.get("f1_micro"),
        "val_f1_macro": val_metrics.get("f1_macro")
      })

      # Check if improvement has occurred.
      if ((tuning_metric_is_loss and val_metrics[tuning_metric] < best_val_metric) or
          (not tuning_metric_is_loss and val_metrics[tuning_metric] > best_val_metric)):
        best_val_metric = val_metrics[tuning_metric]
        best_val_metrics = copy.deepcopy(val_metrics)
        test_metrics = self.test(data, test_on_val=False)
        early_stopping_counter = 0  # Reset if improvement.
      else:
        early_stopping_counter += 1

      if early_stopping_counter >= early_stopping:
        print(f"Early stopping at epoch {i+1}")
        break

    if test_metrics is None:
      test_metrics = self.test(data, test_on_val=False)
    
    if best_val_metrics is None:
      best_val_metrics = self.test(data, test_on_val=True)

    # Optionally log the complete loss history.
    wandb.log({"loss_over_epochs": losses})

    return losses, test_metrics, best_val_metrics

  def Benchmark(self, element, tuning_metric: str = None, tuning_metric_is_loss: bool = False, early_stopping: int = 250):
    torch_data = element['torch_data']
    masks = element['masks']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {'skipped': skipped, 'results': None}
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {}
    out['test_metrics'] = {}

    if skipped:
      print(f'Skipping benchmark for sample id {sample_id}', flush=True)
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    train_mask, val_mask, test_mask = masks
    self.SetMasks(train_mask, val_mask, test_mask)
    
    val_metrics = {}
    test_metrics = {}
    losses = None
    try:
      losses, test_metrics, val_metrics = self.train(torch_data, tuning_metric=tuning_metric,
                                                     tuning_metric_is_loss=tuning_metric_is_loss,
                                                     early_stopping=early_stopping)
    except Exception as e:
      print(f'Failed to run for sample id {sample_id} using model {self._model.__class__.__name__}: {str(e)}')
      out['skipped'] = True

    out['losses'] = losses
    out['test_metrics'].update(test_metrics)
    out['val_metrics'].update(val_metrics)

    # Log final metrics and hyperparameters with wandb.
    wandb.log({
      "sample_id": sample_id,
      "model": self._model.__class__.__name__,
      "final_test_accuracy": test_metrics.get("accuracy"),
      "final_val_accuracy": val_metrics.get("accuracy"),
      "final_test_f1_micro": test_metrics.get("f1_micro"),
      "final_val_f1_macro": val_metrics.get("f1_macro"),
      "final_loss": losses[-1] if losses else None,
      "lr": self._optimizer.param_groups[0]["lr"] if self._optimizer.param_groups else None,
      "weight_decay": self._optimizer.param_groups[0].get("weight_decay") if self._optimizer.param_groups else None,
      # Optionally add more hyperparameters here.
    })

    return out


class NNNodeBaselineBenchmarker(Benchmarker):

  def __init__(self, generator_config, model_class, benchmark_params, h_params, torch_data=None):
    super().__init__(generator_config, model_class, benchmark_params, h_params, torch_data=None)
    # Remove meta entries from h_params.
    self._alpha = h_params['alpha']

  def GetModelName(self):
    return 'PPRBaseline'

  def test(self, data, graph, masks, test_on_val=False):
    train_mask, val_mask, test_mask = masks
    node_ids = np.arange(train_mask.shape[0])
    labels = data.y.numpy()
    nodes_train, nodes_val, nodes_test = node_ids[train_mask], node_ids[val_mask], node_ids[test_mask]
    n_classes = max(data.y.numpy()) + 1
    pers = graph.new_vertex_property("double")
    if test_on_val:
      pred = np.zeros((len(nodes_val), n_classes))
      for idx, node in enumerate(nodes_val):
        pers.a = 0
        pers[node] = 1
        pprs = np.array(gt.pagerank(graph, damping=1-self._alpha, pers=pers, max_iter=100).a)
        pred[idx, labels[nodes_train]] += pprs[nodes_train]
    else:
      pred = np.zeros((len(nodes_test), n_classes))
      for idx, node in enumerate(nodes_test):
        pers.a = 0
        pers[node] = 1
        pprs = np.array(gt.pagerank(graph, damping=1-self._alpha, pers=pers, max_iter=100).a)
        pred[idx, labels[nodes_train]] += pprs[nodes_train]

    pred_best = pred.argmax(-1)
    if test_on_val:
      correct = labels[nodes_val]
    else:
      correct = labels[nodes_test]

    pred_onehot = np.zeros((len(pred_best), n_classes))
    pred_onehot[np.arange(pred_best.shape[0]), pred_best] = 1

    correct_onehot = np.zeros((len(correct), n_classes))
    correct_onehot[np.arange(correct.shape[0]), correct] = 1

    results = {
        'accuracy': sklearn.metrics.accuracy_score(correct, pred_best),
        'f1_micro': sklearn.metrics.f1_score(correct, pred_best, average='micro'),
        'f1_macro': sklearn.metrics.f1_score(correct, pred_best, average='macro'),
        'rocauc_ovr': sklearn.metrics.roc_auc_score(correct_onehot, pred_onehot, multi_class='ovr'),
        'rocauc_ovo': sklearn.metrics.roc_auc_score(correct_onehot, pred_onehot, multi_class='ovo'),
        'logloss': sklearn.metrics.log_loss(correct, pred)}
    return results

  def Benchmark(self, element, tuning_metric: str = None, tuning_metric_is_loss: bool = False):
    gt_data = element['gt_data']
    torch_data = element['torch_data']
    masks = element['masks']
    skipped = element['skipped']
    sample_id = element['sample_id']

    out = {'skipped': skipped, 'results': None}
    out.update(element)
    out['losses'] = None
    out['val_metrics'] = {}
    out['test_metrics'] = {}

    if skipped:
      logging.info(f'Skipping benchmark for sample id {sample_id}')
      return out

    try:
      out['val_metrics'].update(self.test(torch_data, gt_data, masks, test_on_val=True))
      out['test_metrics'].update(self.test(torch_data, gt_data, masks, test_on_val=False))
    except Exception:
      logging.info(f'Failed to compute test accuracy for sample id {sample_id}')
      out['skipped'] = True

    return out


@gin.configurable
class NNNodeBenchmark(BenchmarkerWrapper):
  def GetBenchmarker(self):
    return NNNodeBenchmarker(self._model_class, self._benchmark_params, self._h_params)
  def GetBenchmarkerClass(self):
    return NNNodeBenchmarker

@gin.configurable
class NNNodeBaselineBenchmark(BenchmarkerWrapper):
  def GetBenchmarker(self):
    return NNNodeBaselineBenchmarker(self._model_class, self._benchmark_params, self._h_params)
  def GetBenchmarkerClass(self):
    return NNNodeBaselineBenchmarker
