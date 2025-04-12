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

                # Build search space from h_params
                search_space = {}
                for param_name, val in h_params.items():
                    if isinstance(val, list):
                        search_space[param_name] = hp.choice(param_name, val)
                    elif isinstance(val, tuple) and len(val) == 2:
                        low, high = val
                        search_space[param_name] = hp.uniform(param_name, low, high)

                bench_params_sample, _ = SampleModelConfig(benchmark_params, None)

                # Define the objective function for hyperopt
                def objective(candidate_hparams):
                    """
                    Objective function for hyperopt to minimize.
                    Takes candidate hyperparameters, runs the benchmark, returns loss and results.
                    """
                    # candidate_hparams already contains the actual values selected by hyperopt
                    # (no need for index lookup here like in the post-processing of hp.choice before)
                    final_hparams = candidate_hparams.copy()

                    try:
                        # Instantiate the benchmarker with the candidate hyperparameters
                        benchmarker = benchmarker_class(
                            element.get('generator_config'),
                            model_class,
                            bench_params_sample, # Using the fixed sample
                            final_hparams,
                            element.get('torch_data')
                        )

                        # Run the benchmark
                        result = benchmarker.Benchmark(
                            element,
                            tuning_metric=self._tuning_metric,
                            tuning_metric_is_loss=self._tuning_metric_is_loss
                        )

                        # Extract the validation score to be optimized
                        # Use a high value (infinity) if metric not found, suitable for minimization
                        val_score = result.get('val_metrics', {}).get(self._tuning_metric, float('inf'))

                        # Calculate the loss for hyperopt (always minimization)
                        # If the metric is like accuracy (higher is better), negate it
                        loss = -val_score if not self._tuning_metric_is_loss else val_score

                        if loss == float('inf'):
                            print(f"Warning: Tuning metric '{self._tuning_metric}' not found in val_metrics for params: {final_hparams}. Treating as failure.")
                            return {'loss': loss, 'status': STATUS_FAIL, 'attachments': {'full_result': result}}

                        # Return results - use 'attachments' for extra data hyperopt doesn't strictly need
                        return {
                            'loss': loss,
                            'status': STATUS_OK,
                            'attachments': { # Store full results here to retrieve later
                                'full_result': result
                            }
                        }

                    except Exception as e:
                        import traceback
                        print(f"ERROR during benchmarking trial with params: {final_hparams}")
                        print(traceback.format_exc())
                        # Report failure to hyperopt
                        return {
                            'loss': float('inf'), # Ensure failed trials aren't considered 'best'
                            'status': STATUS_FAIL,
                            'attachments': {'error': str(e)}
                        }

                # Initialize the Trials object to store history
                trials = Trials()

                print(f"Starting Bayesian Optimization with TPE for {self._num_tuning_rounds} rounds...")

                # Run the optimization using fmin
                # fmin will manage the loop internally for num_tuning_rounds evaluations
                best_raw_params = fmin(
                    fn=objective,            # The function to minimize
                    space=search_space,      # The hyperparameter space
                    algo=tpe.suggest,        # The TPE algorithm
                    max_evals=self._num_tuning_rounds, # The total number of evaluations
                    trials=trials,           # The object to store trial results
                    verbose=1                # Set >0 to see hyperopt progress (optional)
                )

                print("\nOptimization finished.")

                # --- Post-Optimization ---

                # `best_raw_params` contains the raw values (e.g., indices for hp.choice) found by fmin.
                # Use `space_eval` to convert them back to the actual parameter values.
                best_final_hparams = space_eval(search_space, best_raw_params)
                chosen_hparams = best_final_hparams

                # Find the best trial results from the Trials object
                # `trials.best_trial` holds information about the trial with the minimum loss
                best_trial_info = trials.best_trial

                # --- Add Debug Print ---
                print("DEBUG: best_trial_info:", best_trial_info)
                # --- End Debug Print ---

                if best_trial_info and best_trial_info['result']['status'] == STATUS_OK:
                    # Extract the full results stored in attachments during the best trial run
                    best_result_dict = best_trial_info['attachments']['full_result']
                    best_loss = best_trial_info['result']['loss']

                    val_metrics = best_result_dict.get('val_metrics', {})
                    test_metrics = best_result_dict.get('test_metrics', {}) # Evaluate test set cautiously

                    print(f"Best hyperparams found (loss={best_loss:.4f}):")
                    print(best_final_hparams)
                    print("Best trial's validation metrics:")
                    print(val_metrics)
                    # Be mindful about reporting test metrics obtained during hyperparameter search
                    # It's best practice to retrain the final model on train+val and evaluate on test once
                    print("Best trial's test metrics (use with caution):")
                    print(test_metrics)

                    # Assign output for external use if needed (replace 'best_out' logic)
                    best_out = best_result_dict

                else:
                    print("No successful trials completed or an issue occurred finding the best trial.")
                    best_out = {} # Assign empty dict or handle error
                    val_metrics = {}
                    test_metrics = {}

                # If desired, store the trial details (unchanged)
                if self._save_tuning_results:
                    # Need a way to get the model name - either pass it or instantiate benchmarker briefly
                    temp_benchmarker_for_name = benchmarker_class(None, model_class, None, None, None) # Adjust as needed
                    model_name = temp_benchmarker_for_name.GetModelName()
                    # Store results keyed by model name, using the full trials object
                    output_data[f'{model_name}__hyperopt_trials'] = trials # Store the whole object or trials.results



            # Collect final metrics
            model_name = model_class.__name__ # Get name directly from class
            for key, value in val_metrics.items():
                output_data[f'{model_name}__val_{key}'] = value
            for key, value in test_metrics.items():
                output_data[f'{model_name}__test_{key}'] = value

            # Store final hyperparams used
            if chosen_hparams: # Check if dict is not empty/None
                for k, v in chosen_hparams.items():
                    output_data[f'{model_name}__model_{k}'] = v # Use model_name from Fix 1
            if bench_params_sample: # Check if dict is not empty/None
                for k, v in bench_params_sample.items():
                    output_data[f'{model_name}__train_{k}'] = v # Use model_name from Fix 1

        yield json.dumps(output_data)