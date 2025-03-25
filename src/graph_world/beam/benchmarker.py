# Copyright 2020 Google LLC
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

import json
import math

from abc import ABC, abstractmethod
import apache_beam as beam
import gin
import numpy as np

from ..models.utils import ComputeNumPossibleConfigs, SampleModelConfig, GetCartesianProduct

# Hyperopt for TPE-based Bayesian optimisation
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
except ImportError:
    raise ImportError("Please install hyperopt (e.g. `pip install hyperopt`).")

class Benchmarker(ABC):
    """
    Base class for training and testing a model. Override Benchmark() as needed.
    """

    def __init__(self, generator_config,
                 model_class=None, benchmark_params=None, h_params=None, torch_data=None):
        self._model_name = model_class.__name__ if model_class is not None else ''
        self._model_class = model_class
        self._benchmark_params = benchmark_params
        self._h_params = h_params
        self.AdjustParams(generator_config, torch_data)

    def AdjustParams(self, generator_config, torch_data):
        """
        Override this function if the input data affects the model architecture.
        """
        pass

    def GetModelName(self):
        return self._model_name

    @abstractmethod
    def Benchmark(self, element,
                  test_on_val: bool = False,
                  tuning_metric: str = None,
                  tuning_metric_is_loss: bool = False):
        """
        Train and test the model on the given element (e.g. data).
        Returns a dict with at least 'losses' and 'test_metrics'.
        """
        del element  # unused
        del test_on_val  # unused
        del tuning_metric  # unused
        del tuning_metric_is_loss  # unused
        return {'losses': [], 'test_metrics': {}}


class BenchmarkerWrapper(ABC):
    """
    Wraps a model and benchmark configuration so it can be passed around in Beam.
    """

    def __init__(self, model_class=None, benchmark_params=None, h_params=None):
        self._model_class = model_class
        self._benchmark_params = benchmark_params
        self._h_params = h_params

    @abstractmethod
    def GetBenchmarker(self):
        return Benchmarker()

    @abstractmethod
    def GetBenchmarkerClass(self):
        return Benchmarker

    def GetModelClass(self):
        return self._model_class

    def GetModelHparams(self):
        return self._h_params

    def GetBenchmarkParams(self):
        return self._benchmark_params

class BenchmarkGNNParDo(beam.DoFn):
    """
    A Beam DoFn that runs TPE-based search over numeric or discrete hyperparameters.
    """

    def __init__(self, benchmarker_wrappers, num_tuning_rounds,
                 tuning_metric, tuning_metric_is_loss=False, save_tuning_results=False):
        """
        benchmarker_wrappers: list of callables that instantiate the benchmarkers
                              (or their classes).
        num_tuning_rounds: number of TPE evaluations
        tuning_metric: metric name (e.g. 'accuracy')
        tuning_metric_is_loss: if True, lower is better; if False, we invert it for TPE
        save_tuning_results: if True, store trial details in output_data
        """
        # Store references rather than objects for backward-compatibility
        self._benchmarker_classes = [
            wrapper().GetBenchmarkerClass() for wrapper in benchmarker_wrappers
        ]
        self._model_classes = [
            wrapper().GetModelClass() for wrapper in benchmarker_wrappers
        ]
        self._benchmark_params = [
            wrapper().GetBenchmarkParams() for wrapper in benchmarker_wrappers
        ]
        self._h_params = [
            wrapper().GetModelHparams() for wrapper in benchmarker_wrappers
        ]

        self._output_path = None
        self._num_tuning_rounds = num_tuning_rounds
        self._tuning_metric = tuning_metric
        self._tuning_metric_is_loss = tuning_metric_is_loss
        self._save_tuning_results = save_tuning_results

    def SetOutputPath(self, output_path):
        self._output_path = output_path

    def process(self, element):
        """
        element: a dict from earlier pipeline stages, e.g.:
            {
                'generator_config': {...},
                'torch_data': ...,
                'metrics': {...},
                'skipped': bool,
                'sample_id': ...,
                ...
            }

        Yields: JSON string of final results.
        """
        # 1) Prepare output data
        output_data = {}
        output_data.update(element.get('generator_config', {}))
        output_data.update(element.get('metrics', {}))
        output_data['marginal_param'] = element.get('marginal_param')
        output_data['fixed_params'] = element.get('fixed_params')
        output_data['skipped'] = element.get('skipped', False)
        output_data['sample_id'] = element.get('sample_id')

        if output_data['skipped']:
            yield json.dumps(output_data)

        # 2) For each benchmarker in the list, run TPE-based search
        for benchmarker_class, benchmark_params, model_class, h_params in zip(
            self._benchmarker_classes,
            self._benchmark_params,
            self._model_classes,
            self._h_params
        ):
            print(f"Running {benchmarker_class.__name__} with model {model_class}.")
            print("Hyperparameters:", h_params)
            print("Benchmark parameters:", benchmark_params)

            # Decide how many tuning rounds
            num_possible = ComputeNumPossibleConfigs(benchmark_params, h_params)
            actual_rounds = min(num_possible, self._num_tuning_rounds)
            print("Possible configs:", num_possible, "Using rounds:", actual_rounds)

            # Single or zero round => skip TPE
            if actual_rounds <= 1 or not self._tuning_metric:
                bench_params_sample, hyperparams_sample = SampleModelConfig(benchmark_params, h_params)

                # Instantiate and run training
                benchmarker = benchmarker_class(
                    element.get('generator_config'),
                    model_class,
                    bench_params_sample,
                    hyperparams_sample,
                    element.get('torch_data')
                )
                out = benchmarker.Benchmark(
                    element,
                    tuning_metric=self._tuning_metric,
                    tuning_metric_is_loss=self._tuning_metric_is_loss
                )
                val_metrics = out.get('val_metrics', {})
                test_metrics = out.get('test_metrics', {})

            else:
                # ---------- TPE-BASED BAYESIAN OPTIMISATION ----------
                trials = Trials()

                # Build search space from h_params
                search_space = {}
                for param_name, val in h_params.items():
                    if isinstance(val, list):
                        search_space[param_name] = hp.choice(param_name, val)
                    elif isinstance(val, tuple) and len(val) == 2:
                        low, high = val
                        search_space[param_name] = hp.uniform(param_name, low, high)
                    # adapt as needed for int ranges or log scale

                def objective(candidate):
                    # Convert indices to actual values if using hp.choice
                    final_hparams = {}
                    for k, v in candidate.items():
                        final_hparams[k] = v

                    # Sample benchmark_params if desired
                    if benchmark_params is not None:
                        bench_params_sample, _ = SampleModelConfig(benchmark_params, None)
                    else:
                        bench_params_sample = None

                    # Train with these hyperparams
                    benchmarker = benchmarker_class(
                        element.get('generator_config'),
                        model_class,
                        bench_params_sample,
                        final_hparams,
                        element.get('torch_data')
                    )
                    result = benchmarker.Benchmark(
                        element,
                        tuning_metric=self._tuning_metric,
                        tuning_metric_is_loss=self._tuning_metric_is_loss
                    )
                    val_score = result.get('val_metrics', {}).get(self._tuning_metric, None)
                    if val_score is None:
                        val_score = 999999.0

                    # Hyperopt minimises the loss => invert a "max" metric
                    if not self._tuning_metric_is_loss:
                        val_score = -val_score

                    return {'loss': val_score, 'status': STATUS_OK}

                best_params = fmin(
                    fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=actual_rounds,
                    trials=trials
                )

                # Convert best param indices to real values
                final_hparams = {}
                for k, v in best_params.items():
                    if isinstance(h_params[k], list):
                        final_hparams[k] = h_params[k][v]
                    else:
                        final_hparams[k] = v

                # Evaluate final config
                if benchmark_params is not None:
                    bench_params_sample, _ = SampleModelConfig(benchmark_params, None)
                else:
                    bench_params_sample = None

                # Run the benchmark one last time
                benchmarker = benchmarker_class(
                    element.get('generator_config'),
                    model_class,
                    bench_params_sample,
                    final_hparams,
                    element.get('torch_data')
                )
                best_out = benchmarker.Benchmark(
                    element,
                    tuning_metric=self._tuning_metric,
                    tuning_metric_is_loss=self._tuning_metric_is_loss
                )

                val_metrics = best_out.get('val_metrics', {})
                test_metrics = best_out.get('test_metrics', {})

                print("Best hyperparams:", final_hparams)
                print("Best val metrics:", val_metrics)
                print("Best test metrics:", test_metrics)

                # If desired, store the trial details
                if self._save_tuning_results:
                    output_data[f'{benchmarker.GetModelName()}__hyperopt_trials'] = trials.results


            # Collect final metrics
            for key, value in val_metrics.items():
                output_data[f'{benchmarker.GetModelName()}__val_{key}'] = value
            for key, value in test_metrics.items():
                output_data[f'{benchmarker.GetModelName()}__test_{key}'] = value

            # Store final hyperparams used
            # (either TPE best or single-run random sample)
            chosen_hparams = locals().get("final_hparams", None) or locals().get("hyperparams_sample", {})
            if chosen_hparams:
                for k, v in chosen_hparams.items():
                    output_data[f'{benchmarker.GetModelName()}__model_{k}'] = v
            if locals().get("bench_params_sample", None):
                for k, v in bench_params_sample.items():
                    output_data[f'{benchmarker.GetModelName()}__train_{k}'] = v

        yield json.dumps(output_data)