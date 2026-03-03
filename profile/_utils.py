import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from datetime import datetime
import time
import csv
from _sys_usage import SystemProfiler


@dataclass
class ProfilingStat:
    name: str
    value: Any
    unit: str | None = None

    def __repr__(self):
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.name}: {self.value}{unit}"


class ProfilerBase(ABC):

    def __init__(
        self,
        models_names: list[str],
        logger: logging.Logger,
        *,
        run_forever: bool,
        n_threads: int | None,
    ):
        self._model_names = models_names
        self._run_forever = run_forever
        self._logger = logger
        # Reduced sampling interval from 25ms to 10ms for better CPU measurement granularity
        self._sys_prof = SystemProfiler(interval_ms=10, hist_length=2000)

        self._all_stats: dict[
            str, dict[str, ProfilingStat | dict[str, ProfilingStat]]
        ] = {
            "environment": {"max_threads": ProfilingStat("Max CPU threads", n_threads)},
            "infer_stats": {
                model_name: {
                    "n_iters": ProfilingStat("Num inferences", 0),
                    "tot_infer_time": ProfilingStat("Total inference time", 0, "ms"),
                    "avg_infer_time": ProfilingStat("Average inference time", 0, "ms"),
                    "avg_sys_usage": {},
                }
                for model_name in self._model_names
            },
        }
        # store per-model load times (seconds)
        self._load_times: dict[str, float] = {}

    @abstractmethod
    def _get_inference_time(
        self, model_name: str, input_item: str | None = None
    ) -> float:
        """Run a single inference for `model_name` using `input_item` and
        return inference time in seconds."""
        ...

    @abstractmethod
    def _cleanup(self, model_name: str): ...

    def _update_env_param(self, param: str, value: ProfilingStat):
        self._all_stats["environment"].update({param: value})

    def _update_sys_usage(self, model_name: str):
        sys_usage = self._all_stats["infer_stats"][model_name]["avg_sys_usage"]
        for cpu, usage in self._sys_prof.avg_cpu_usage().items():
            sys_usage[cpu] = ProfilingStat(cpu.upper(), round(100 * usage, 2), "%")
        sys_usage["npu"] = ProfilingStat(
            "NPU", round(100 * self._sys_prof.avg_npu_usage(), 2), "%"
        )
        sys_usage["mem"] = ProfilingStat(
            "RAM", round(self._sys_prof.avg_mem_usage() / 1_048_576, 2), "GB"
        )
        self._sys_prof.reset()

    def profile_models(
        self,
        n_iters: int,
        inputs: list[str] | None = None,
        store: str | None = None,
        print_stats: bool = True,
    ) -> dict[str, dict]:

        def _cleanup_models():
            for model_name in self._model_names:
                self._cleanup(model_name)

        # prepare inputs list
        if inputs is None:
            inputs = [None]

        self._sys_prof.start()
        # Allow profiler to collect initial samples (reduced sampling interval needs more time)
        time.sleep(0.1)
        results: list[dict] = []
        try:
            while True:
                for model_name in self._model_names:
                    self._logger.info(
                        f"Profiling '{model_name}' ({n_iters} iters per input)..."
                    )
                    infer_stats: dict[str, ProfilingStat] = self._all_stats[
                        "infer_stats"
                    ][model_name]
                    for input_item in inputs:
                        try:
                            for iter_idx in range(n_iters):

                                now = datetime.now()
                                formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
                                # Snapshot before inference
                                bf_cpu_raw = self._sys_prof.curr_cpu_usage
                                bf_cpu = {
                                    k: round(v * 100, 3) for k, v in bf_cpu_raw.items()
                                }
                                # Calculate average CPU across all cores
                                bf_cpu_avg = round(
                                    sum(bf_cpu.values()) / len(bf_cpu) if bf_cpu else 0.0,
                                    3
                                )
                                bf_npu = round(100 * self._sys_prof.curr_npu_usage, 3)
                                bf_mem = round(
                                    self._sys_prof.curr_mem_usage / 1024.0, 3
                                )

                                infer_time = self._get_inference_time(
                                    model_name, input_item
                                )
                                # print(infer_time)

                                # Snapshot after inference (matching 'before' measurement method)
                                # Using curr_* for consistency with bf_cpu snapshot approach
                                af_cpu_raw = self._sys_prof.curr_cpu_usage
                                af_cpu = {
                                    k: round(v * 100, 3) for k, v in af_cpu_raw.items()
                                }
                                # Calculate average CPU across all cores (same method as bf_cpu)
                                af_cpu_avg = round(
                                    sum(af_cpu.values()) / len(af_cpu) if af_cpu else 0.0,
                                    3
                                )
                                af_npu = round(100 * self._sys_prof.curr_npu_usage, 3)
                                af_mem = round(
                                    self._sys_prof.curr_mem_usage / 1024.0, 3
                                )
                                infer_stats["tot_infer_time"].value += infer_time
                                infer_stats["n_iters"].value += 1
                                results.append(
                                    {
                                        "input": input_item,
                                        "infer_time_s": infer_time,
                                        "current_time": formatted_datetime,
                                        "bf_cpu": bf_cpu_avg,
                                        "bf_npu": bf_npu,
                                        "bf_mem_mb": bf_mem,
                                        "af_cpu": af_cpu_avg,
                                        "af_npu": af_npu,
                                        "af_mem_mb": af_mem,
                                    }
                                )
                                
                        except Exception as e:
                            self._logger.warning(
                                f"Stopping inference due to error: {e}"
                            )
                            break
                    # update averaged sys usage after completing inputs
                    self._update_sys_usage(model_name)
                if not self._run_forever:
                    break
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            _cleanup_models()
            self._sys_prof.stop()

        # finalize stats
        for model_name, infer_stats in self._all_stats["infer_stats"].items():
            n_iters_done = infer_stats["n_iters"].value
            infer_stats["avg_infer_time"].value = round(
                infer_stats["tot_infer_time"].value / (n_iters_done or 1) * 1000, 3
            )
            infer_stats["tot_infer_time"].value = round(
                infer_stats["tot_infer_time"].value * 1000, 3
            )

        if store:
            # flatten results to CSV
            keys = results[0].keys()
            with open(store, 'w', newline='', encoding='utf-8') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(results)
            self._logger.info("Saved profiling CSV to %s", store)

        if print_stats:
            self.print_stats()
        return self._all_stats

    def print_stats(self):
        SPACER = " " * 4
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        print("\n\nProfiling report")
        print("------------------------------------")
        print("Environment:")
        for env_var in self._all_stats["environment"].values():
            print(SPACER + str(env_var))
        for model_name, infer_stats in self._all_stats["infer_stats"].items():
            print(f"\nStats for '{model_name}':")
            for stat_name, stat in infer_stats.items():
                if isinstance(stat, dict) and stat_name == "avg_sys_usage":
                    print(SPACER + "System usage:")
                    print(
                        2 * SPACER
                        + YELLOW
                        + "NOTE: Measurements are affected by other running processes"
                        + RESET
                    )
                    for sys_stat in stat.values():
                        print(2 * SPACER + str(sys_stat))
                else:
                    print(SPACER + str(stat))
        print()


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    model_choices: list[str],
    default_model: str | list[str],
    default_input: str,
    input_desc: str,
):
    if isinstance(default_model, str):
        default_model: list[str] = [default_model]
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        metavar="MODEL",
        nargs="+",
        choices=model_choices,
        default=default_model,
        help="MiniLM models to profile (default: %(default)s, available: %(choices)s)",
    )
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=1,
        help="Number of iterations to repeat inference (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        default=[default_input],
        help=input_desc + ' (one or more, default: "%(default)s")',
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)",
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)",
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        default=False,
        help="Run profiling forever, alternating between provided models",
    )
    parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="If provided, path to write CSV with per-iteration profiling records",
    )


def configure_logging(verbosity: str):
    level = getattr(logging, verbosity.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {verbosity}")

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
