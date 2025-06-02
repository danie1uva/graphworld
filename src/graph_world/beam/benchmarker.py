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
import time

from ..models.utils import ComputeNumPossibleConfigs, SampleModelConfig, GetCartesianProduct

# Hyperopt for TPE-based Bayesian optimisation
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
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
            print(f"\nRunning {benchmarker_class.__name__} with model {model_class}.")
            print("Hyperparameters:", h_params)
            print("Benchmark parameters:", benchmark_params)

            # Decide how many tuning rounds
            num_possible = ComputeNumPossibleConfigs(benchmark_params, h_params)
            actual_rounds = min(num_possible, self._num_tuning_rounds)
            print("Possible configs:", num_possible, "Using rounds:", actual_rounds)

            # Single or zero round => skip TPE
            if actual_rounds <= 1 or not self._tuning_metric:
                bench_params_sample, hyperparams_sample = SampleModelConfig(benchmark_params, h_params)
                chosen_hparams = hyperparams_sample

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
                search_space = {}
                for param_name, val in h_params.items():
                    if isinstance(val, list):
                        search_space[param_name] = hp.choice(param_name, val)
                    elif isinstance(val, tuple) and len(val) == 2:
                        low, high = val
                        search_space[param_name] = hp.uniform(param_name, low, high)

                bench_params_sample, _ = SampleModelConfig(benchmark_params, None)

                def objective(candidate_hparams):
                    # ... (objective function definition as before, minimal comments inside ok) ...
                    # ... (it should return {'loss': ..., 'status': ..., 'attachments': {'full_result': ...}}) ...
                    final_hparams = candidate_hparams.copy()
                    try:
                        benchmarker = benchmarker_class(
                            element.get('generator_config'), model_class,
                            bench_params_sample, final_hparams, element.get('torch_data')
                        )
                        result = benchmarker.Benchmark(
                            element, tuning_metric=self._tuning_metric,
                            tuning_metric_is_loss=self._tuning_metric_is_loss
                        )
                        val_score = result.get('val_metrics', {}).get(self._tuning_metric, float('inf'))
                        loss = -val_score if not self._tuning_metric_is_loss else val_score
                        if loss == float('inf'):
                            print(f"Warning: Tuning metric '{self._tuning_metric}' not found. Params: {final_hparams}.")
                            return {'loss': loss, 'status': STATUS_FAIL, 'attachments': {'full_result': result}}
                        return {'loss': loss, 'status': STATUS_OK, 'attachments': {'full_result': result}}
                    except Exception as e:
                        import traceback
                        print(f"ERROR during trial. Params: {final_hparams}\n{traceback.format_exc()}")
                        return {'loss': float('inf'), 'status': STATUS_FAIL, 'attachments': {'error': str(e)}}

                trials = Trials()
                print(f"\nStarting TPE for {actual_rounds} rounds...")

                best_raw_params = fmin(
                    fn=objective, space=search_space, algo=tpe.suggest,
                    max_evals=actual_rounds, # Use actual_rounds based on space size vs budget
                    trials=trials, verbose=0
                )
                print("Optimization finished.\n")

                try:
                     best_final_hparams = space_eval(search_space, best_raw_params)
                     print(f"Best hyperparams evaluated: {best_final_hparams}")
                except Exception as e:
                     print(f"Error during space_eval: {e}. Cannot determine best hyperparameters.")
                     best_final_hparams = None
                     val_metrics = {} # Ensure these are defined for aggregation
                     test_metrics = {}
                     continue # Skip final run and aggregation for this model

                # Since 'attachments' with full results aren't reliably retrieved from trials.best_trial,
                # re-run the benchmark once using the best found hyperparameters.
                print("Re-running benchmark with best hyperparameters...\n")
                val_metrics = {}
                test_metrics = {}
                if best_final_hparams is not None:
                    try:
                        final_benchmarker = benchmarker_class(
                            element.get('generator_config'), model_class,
                            bench_params_sample, best_final_hparams, element.get('torch_data')
                        )
                        best_out = final_benchmarker.Benchmark(
                             element, tuning_metric=self._tuning_metric,
                             tuning_metric_is_loss=self._tuning_metric_is_loss
                        )
                        test_metrics = best_out.get('test_metrics', {})
                        print("Final run test metrics:", test_metrics) # Optional
                    except Exception as e:
                        print(f"ERROR during final benchmark run. Params: {best_final_hparams}")
                        # Metrics remain empty
                if self._save_tuning_results:
                    model_name_for_saving = model_class.__name__
                    output_data[f'{model_name_for_saving}__hyperopt_trials'] = trials

            # --- Aggregation ---
            model_name = model_class.__name__

            if actual_rounds <= 1 or not self._tuning_metric:
                chosen_hparams = locals().get("hyperparams_sample", {})
                chosen_bench_params = locals().get("bench_params_sample", {})
                # val/test metrics already assigned in the 'if' branch's locals() scope
                val_metrics = locals().get("val_metrics", {})
                test_metrics = locals().get("test_metrics", {})
            else: # TPE branch
                chosen_hparams = best_final_hparams if best_final_hparams is not None else {}
                chosen_bench_params = bench_params_sample
                # val_metrics/test_metrics assigned from the re-run block

            for key, value in val_metrics.items():
                output_data[f'{model_name}__val_{key}'] = value
            for key, value in test_metrics.items():
                output_data[f'{model_name}__test_{key}'] = value

            if chosen_hparams:
                for k, v in chosen_hparams.items():
                    output_data[f'{model_name}__model_{k}'] = v
            if chosen_bench_params:
                for k, v in chosen_bench_params.items():
                    output_data[f'{model_name}__train_{k}'] = v

        # --- End loop over benchmarkers ---
        yield json.dumps(output_data)