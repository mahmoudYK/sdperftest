#!/usr/bin/env python3
# License: LGPL-2.1-or-later

import statistics
import sys
import os
import subprocess
import argparse
import re
import functools
import collections
import math
import json
import dataclasses
import stat
import time
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DEFAULT_START_SERVICES_NUM = 100
DEFAULT_STEP_SERVICES_NUM = 100
DEFAULT_TEST_NUM = 10
ROOT_UID = 0
ROOT_GID = 0
DEFAULT_NONROOT_USER_UID = 1000
DEFAULT_NONROOT_USER_GID = 1000
MICROSEC_TO_SEC = 1000000
MILISEC_TO_SEC = 1000
MIN_TO_SEC = 60
HOUR_TO_SEC = 60 * 60
FIG_WIDTH = 25
FIG_HEIGHT = 15
DEFAULT_PERF_FREQUENCY = "max"
DEFAULT_PERF_SLEEP_PERIOD = 5  # 5 seconds
PERF_OUTPUT_DATA_FILE = "perf.data"
PYTHON_PATH = "/usr/bin/python"
TIME_CMD_PATH = "/usr/bin/time"
PERF_CMD_PATH = "/usr/bin/perf"
SERVICES_GENERATOR_SCRIPT = "generate_services.py"
SYSTEMD_SYSTEM_PATH = "/run/systemd/system/"
DEFAULT_OUTPUT_ARTIFACTS_FILE_NAME = "sd_load_test"
SD_BUILD_SCRIPT = "./build_sd.sh"
SD_BUILD_DIR = "sd_build_load_test"
DEFAULT_OUTPUT_ARTIFACTS_DIR = os.getcwd()


@dataclasses.dataclass
class ErrorStats:
    """error statistics data class"""

    abs_error: list[float]
    mean_abs: float
    max_abs: float
    min_abs: float
    stddev_abs: float
    percent_error: list[float]
    mean_percent: float
    max_percent: float
    min_percent: float
    stddev_percent: float


def fail(fail_message: str) -> None:
    """print error message and exit with status 1"""
    print(f"Error: {fail_message}", file=sys.stderr)
    sys.exit(1)


def print_line(num_of_lines: int = 1, length: int = 100, char: str = "-") -> None:
    """print a dash line to stdout"""
    for _ in range(num_of_lines):
        print(
            char * length,
            file=sys.stdout,
        )


def is_installed(program_name: str) -> bool:
    '''check if a program is installed'''
    installed = shutil.which(program_name) is not None
    if not installed:
        print(f"{program_name} is not installed!")
    return installed


def assemble_service_gen_cmd(
    types: list,
    remove: bool = False,
    services_num: int = None,
    dag_edge_probability: float = None,
    gen_dot: bool = False,
    dot_dir: str = DEFAULT_OUTPUT_ARTIFACTS_DIR,
) -> list:
    """assemble the generate_services.py arguments"""
    cmd = [PYTHON_PATH, SERVICES_GENERATOR_SCRIPT]
    for gen_type in types:
        cmd.append("-t")
        cmd.append(gen_type)
    if remove:
        cmd.append("-r")
    else:
        if services_num:
            cmd.append("-n")
            cmd.append(str(services_num))
        if dag_edge_probability:
            cmd.append("-p")
            cmd.append(str(dag_edge_probability))
        if gen_dot:
            cmd.append("-z")
            cmd.append("-d")
            cmd.append(dot_dir)

    return cmd


def run_cmd(
    cmd: list,
    uid: int,
    gid: int,
    stdout_file=None,
    stderr_file=None,
    non_blocking: bool = False,
) -> subprocess.CompletedProcess | subprocess.Popen:
    """using subprocess to run a command and return subprocess.CompletedProcess"""
    cmd_str = " ".join(cmd)
    print(f"running: {cmd_str}")

    try:
        if non_blocking:
            return subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                user=uid,
                group=gid,
            )
        else:
            return subprocess.run(
                cmd,
                check=True,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                user=uid,
                group=gid,
            )
    except Exception as ex:
        fail(f"running {cmd_str} failed:\n{ex}")


def parse_sd_path_mode(args: argparse.Namespace) -> list:
    """parse sd_path_mode option and return the sd exe paths accordingly"""
    match args.sd_path_mode:
        case "exe":
            return [args.sd_ref_exe_path, args.sd_comp_exe_path]
        case "commit":
            commit_hash_list = [args.sd_commit_ref, args.sd_commit_comp]
            sd_exe_path_list = []
            os.chmod(SD_BUILD_SCRIPT, 0o755)
            for commit_hash in commit_hash_list:
                sd_build_cmd = [SD_BUILD_SCRIPT, "-d", SD_BUILD_DIR, "-c", commit_hash]
                run_cmd(sd_build_cmd, args.user_uid, args.user_gid)
                # exe path example:
                # sd_build_load_test/6fadf01cf3cdd98f78b7829f4c6c892306958394/systemd/build/systemd
                sd_exe_path_list.append(
                    os.path.join(
                        SD_BUILD_DIR, commit_hash, "systemd", "build", "systemd"
                    )
                )
            return sd_exe_path_list
        case _:
            fail("not supportd sd path mode")


def to_seconds(func):
    """decorator to convert time resolution to seconds"""

    def wrapper(*args, **kwargs):
        sd_test_time = func(*args, **kwargs).split(" ")
        load_time = 0
        for t_res in sd_test_time:
            t_unit = re.search(r"[a-z]+", t_res)
            t_value = re.search(r"[^a-z]+", t_res)
            if not t_unit or not t_value:
                fail("expected time value and unit while running systemd --test")
            t_value = float(t_value.group(0))
            match t_unit.group(0):
                case "us":
                    load_time += t_value / MICROSEC_TO_SEC
                case "ms":
                    load_time += t_value / MILISEC_TO_SEC
                case "s":
                    load_time += t_value
                case "min":
                    load_time += t_value * MIN_TO_SEC
                case "h":
                    load_time += t_value * HOUR_TO_SEC
                case _:
                    load_time = float(math.inf)
        return load_time

    return functools.update_wrapper(wrapper, func)


@to_seconds
def parse_sd_test_cmd_result(result: subprocess.CompletedProcess) -> str:
    """parse systemd --test stderr to extract units load time in seconds"""
    units_load_time_line = "Loaded units and determined initial transaction in"
    units_load_time = re.match(
        r"\d.+[.]$",
        str(result.stderr).split(units_load_time_line)[1].split("\n")[0].strip(),
    )
    time_cmd_output = re.findall(
        r"^(?:real|user|sys)\s\d+[.]\d+$", str(result.stderr), re.MULTILINE
    )
    if time_cmd_output:
        [print(f"{time_type} s") for time_type in time_cmd_output]
    if units_load_time:
        return units_load_time.group(0).removesuffix(".")
    fail("can't parse the stderr output of the cmd: systemd --test.")


class ErrorStatsCalculator:
    """calculate error statistics"""

    @classmethod
    def calc_rmse(cls, res_dict: dict) -> float:
        """calculate the root mean square error of 2 equal length lists"""
        if len(res_dict) < 2:
            return 0
        if len(res_dict) > 2:
            fail("calc_rmse fn needs a res_dict with only 2 lists of values")
        else:
            results_list = list(res_dict.values())
            if len(results_list[0]) != len(results_list[1]):
                fail("calc_rmse fn needs a res_dict with only 2 lists of equal length")
            else:
                return math.sqrt(
                    statistics.mean(
                        [
                            math.pow(i - j, 2)
                            for i, j in zip(results_list[0], results_list[1])
                        ]
                    )
                )

    @classmethod
    def calc_error_stats(cls, res_dict: dict) -> ErrorStats:
        """calculate min, max, mean and stddev of absolute error"""
        if len(res_dict) != 2:
            fail("calc_error_stats fn needs a res_dict with 2 lists of values")
        results_list = list(res_dict.values())
        abs_error_list = [
            round(abs(first - second), 4)
            for first, second in zip(results_list[0], results_list[1])
        ]
        # assuming that the first systemd binary (passed by -d) is the reference binary
        percent_error_list = [
            round((abs_error / first) * 100, 4)
            for abs_error, first in zip(abs_error_list, results_list[0])
        ]
        max_error_abs = max(abs_error_list)
        min_error_abs = min(abs_error_list)
        mean_error_abs = round(statistics.mean(abs_error_list), 4)
        stddev_error_abs = round(statistics.stdev(abs_error_list), 4)
        max_error_perc = max(percent_error_list)
        min_error_perc = min(percent_error_list)
        mean_error_perc = round(statistics.mean(percent_error_list), 4)
        stddev_error_perc = round(statistics.stdev(percent_error_list), 4)
        return ErrorStats(
            abs_error_list,
            mean_error_abs,
            max_error_abs,
            min_error_abs,
            stddev_error_abs,
            percent_error_list,
            mean_error_perc,
            max_error_perc,
            min_error_perc,
            stddev_error_perc,
        )


@dataclasses.dataclass
class Reporter:
    """report load test results using different methods"""

    sdpath_loadtime_dict: dict[str, list[float]]
    generated_services_num_list: list[int]
    rmse: float
    output_files_name: str
    output_files_dir: str
    generator_types: list[str]
    dag_edge_probability: float
    error_stats: ErrorStats

    def plot(self) -> None:
        """plot units load time against the number of test services"""
        print("plotting ...")
        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        generator_types_str = (
            "generator types: [" + " ".join(self.generator_types) + "]"
        )
        if "dag" in self.generator_types:
            generator_types_str += (
                f"\ndag edge probability = {self.dag_edge_probability}"
            )
        rmse_str = f"RMSE: {self.rmse:.{4}f}"
        fig.suptitle(
            f"units load time test\n\n{generator_types_str}\n{rmse_str}", fontsize=20
        )
        colors = ["green", "red"]
        horizontal_alignments = ["right", "left"]
        offsets = [-1, 1]
        patches = []
        x_ticks = []
        for sd_path, color, horizontal_alignment, offset in zip(
            self.sdpath_loadtime_dict.keys(),
            colors,
            horizontal_alignments,
            offsets,
        ):
            load_time = self.sdpath_loadtime_dict[sd_path]
            plt.plot(
                self.generated_services_num_list,
                load_time,
                color=color,
                linewidth=1,
            )
            patches.append(mpatches.Patch(color=color, label=sd_path))
            x_ticks, _ = plt.xticks()
            x_offset = (x_ticks[1] - x_ticks[0]) / 25
            for idx, num_services in enumerate(self.generated_services_num_list):
                plt.text(
                    num_services + (offset * x_offset),
                    load_time[idx],
                    load_time[idx],
                    ha=horizontal_alignment,
                    color=color,
                    fontsize=10,
                )
                plt.axvline(
                    x=num_services, color="black", linestyle="dotted", linewidth=0.1
                )
                plt.axhline(
                    y=load_time[idx], color="black", linestyle="dotted", linewidth=0.1
                )
        bottom_y_limit, _ = plt.ylim()
        for idx, num_services in enumerate(self.generated_services_num_list):
            if num_services not in x_ticks:
                plt.text(
                    num_services,
                    bottom_y_limit,
                    num_services,
                    rotation=45,
                    color="black",
                    fontsize=10,
                )
        plt.xlabel("number of test services", fontsize=15)
        plt.ylabel("units load time (sec)", fontsize=15)
        plt.legend(handles=patches, loc="upper left")
        fig_path = os.path.join(self.output_files_dir, self.output_files_name + ".jpg")
        plt.savefig(fig_path, dpi=250)
        print(f"saved figure at {fig_path}")

    def write_json(self) -> None:
        """write test results to json file"""
        print("writing json file ...")
        json_str_dict = {}
        for res_dict_key, res_dict_value in self.sdpath_loadtime_dict.items():
            json_str_dict[res_dict_key] = {
                "num of services:service load time": dict(
                    zip(self.generated_services_num_list, res_dict_value)
                )
            }
        json_str_dict["generator types"] = self.generator_types
        if "dag" in self.generator_types:
            json_str_dict["dag edge probability"] = self.dag_edge_probability
        json_str_dict["rmse"] = round(self.rmse, 4)
        json_str_dict["absolute error"] = dict(
            zip(self.generated_services_num_list, self.error_stats.abs_error)
        )
        json_str_dict["percentage error"] = dict(
            zip(self.generated_services_num_list, self.error_stats.percent_error)
        )
        json_str_dict["absolute error stats"] = {
            "max_abs": self.error_stats.max_abs,
            "min_abs": self.error_stats.min_abs,
            "mean_abs": self.error_stats.mean_abs,
            "stddev_abs": self.error_stats.stddev_abs,
        }
        json_str_dict["percent error stats"] = {
            "max_percent": self.error_stats.max_percent,
            "min_percent": self.error_stats.min_percent,
            "mean_percent": self.error_stats.mean_percent,
            "stddev_percent": self.error_stats.stddev_percent,
        }
        json_file_path = os.path.join(
            self.output_files_dir, self.output_files_name + ".json"
        )
        with open(
            json_file_path,
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(json_str_dict, json_file, indent=2)
        print(f"saved json data at {json_file_path}")

    def print_error_stats(self) -> None:
        """print the error statistics calculated from res_dict value lists"""
        print("error statistics:\n")
        print_line(length=120)
        print(
            f"{'services number' : <20}\
    {'load time(s)' : <20}\
    {'load time(s)' : <20}\
    {'absolute error' : <25}\
    {'percentage error(%)' : <25}"
        )
        print(
            f"{'generated' : <20}\
    {'reference sd' : <20}\
    {'compared sd' : <20}\
    {'abs(ref_lt - comp_lt)' : <25}\
    {'(abs(ref_lt - comp_lt)/ref_lt)*100' : <25}"
        )
        print_line(length=120)
        # 2 lists have the same size
        ref_sd_load_time_list = list(self.sdpath_loadtime_dict.values())[0]
        compared_sd_load_time_list = list(self.sdpath_loadtime_dict.values())[1]
        for test_num, load_time in enumerate(ref_sd_load_time_list):
            print(
                f"{self.generated_services_num_list[test_num] : <20}\
    {load_time : <20}\
    {compared_sd_load_time_list[test_num] : <20}\
    {self.error_stats.abs_error[test_num] : <25}\
    {self.error_stats.percent_error[test_num] : <25}"
            )
        print_line(length=120)
        print(
            f"{'measure' : <20}\
    {' ' : <45}\
    {'absolute error' : <25}\
    {'percentage error(%)' : <25}"
        )
        print_line(length=120)
        print(
            f"{'max' : <20}\
    {' ' : <45}\
    {self.error_stats.max_abs : <25}\
    {self.error_stats.max_percent : <25}"
        )
        print(
            f"{'min' : <20}\
    {' ' : <45}\
    {self.error_stats.min_abs : <25}\
    {self.error_stats.min_percent : <25}"
        )
        print(
            f"{'mean' : <20}\
    {' ' : <45}\
    {self.error_stats.mean_abs : <25}\
    {self.error_stats.mean_percent : <25}"
        )
        print(
            f"{'stddev' : <20}\
    {' ' : <45}\
    {self.error_stats.stddev_abs : <25}\
    {self.error_stats.stddev_percent : <25}"
        )

    def report_test_results(self) -> None:
        """generate load test reports"""
        self.plot()
        print_line(char="#")
        self.write_json()
        print_line(char="#")
        self.print_error_stats()
        print_line(char="#")


class PerfController:
    """linux perf tool prfiling data recorder and flamegraph generator"""

    def __init__(self) -> None:
        self.ctl_dir = "/tmp/"
        self.ctl_fifo = os.path.join(self.ctl_dir, "perf_ctl.fifo")
        self.ctl_ack_fifo = os.path.join(self.ctl_dir, "perf_ctl_ack.fifo")
        self.ctl_fifo_file = None
        self.ctl_ack_fifo_file = None
        self.perf_proc = None

    def check_fifo_exists(self, path: str) -> bool:
        """check if a named pipe exists"""
        try:
            return stat.S_ISFIFO(os.stat(path).st_mode)
        except FileNotFoundError:
            return False

    def __create_fifos(self) -> None:
        """create perf ctl and ack nemd pipes"""
        if self.check_fifo_exists(self.ctl_fifo):
            print(f"{self.ctl_fifo} exists, deleteing...")
            os.unlink(self.ctl_fifo)
        if self.check_fifo_exists(self.ctl_ack_fifo):
            print(f"{self.ctl_ack_fifo} exists, deleteing...")
            os.unlink(self.ctl_ack_fifo)
        try:
            os.mkfifo(self.ctl_fifo, 0o660)
            os.mkfifo(self.ctl_ack_fifo, 0o660)
        except OSError as ex:
            fail(f"failed to open perf fifos: {ex}")

    def __write_to_ctl_fifo(self, text: str) -> None:
        """write data to the perf ctl named pipe"""
        self.ctl_fifo_file.write(text)
        self.ctl_fifo_file.flush()

    def __read_ack(self) -> str:
        """wait till perf writes ack to the ack named pipe"""
        while self.ctl_ack_fifo_file.readline().strip() != "ack":
            time.sleep(0.01)

    def __check_perf_proc_alive(self) -> bool:
        """check if perf process still running"""
        return self.perf_proc.poll() is None

    def run_perf_record(self, sampling_frequency: str, sleep_period: int) -> None:
        """run perf record cmd and generate perf.data"""
        self.__create_fifos()
        perf_record_cmd = [
            PERF_CMD_PATH,
            "record",
            "-a",
            "-g",
            "-D",
            "-1",
            "--control",
            f"fifo:{self.ctl_fifo},{self.ctl_ack_fifo}",
            "-F",
            sampling_frequency,
            "-e",
            "cycles",
            "--",
            "sleep",
            str(sleep_period),
        ]
        self.perf_proc = run_cmd(perf_record_cmd, ROOT_UID, ROOT_GID, non_blocking=True)
        self.ctl_fifo_file = open(self.ctl_fifo, "w")
        self.ctl_ack_fifo_file = open(self.ctl_ack_fifo, "r")
        self.__write_to_ctl_fifo("enable\n")
        self.__read_ack()

    def stop_perf_record(self) -> None:
        """stop perf process if it is still running, close and delete the named pipes"""
        if self.__check_perf_proc_alive():
            self.__write_to_ctl_fifo("stop\n")
            while self.__check_perf_proc_alive():
                time.sleep(0.01)
        self.ctl_fifo_file.close()
        self.ctl_ack_fifo_file.close()
        os.unlink(self.ctl_fifo)
        os.unlink(self.ctl_ack_fifo)

    @classmethod
    def gen_flamegraph(
        cls, output_files_dir: str, output_files_name: str, services_num: int
    ):
        """generate a flamegraph using perf script flamegraph.py"""
        if (
            os.path.isfile(PERF_OUTPUT_DATA_FILE)
            and os.stat(PERF_OUTPUT_DATA_FILE).st_size > 0
        ):
            perf_script_cmd = [
                PERF_CMD_PATH,
                "script",
                "report",
                "flamegraph",
                "-o",
                os.path.join(
                    output_files_dir, output_files_name + str(services_num) + ".html"
                ),
            ]
            run_cmd(perf_script_cmd, ROOT_UID, ROOT_GID)
            os.remove(PERF_OUTPUT_DATA_FILE)
        else:
            print(
                "perf.data file is missing or size = 0, may be increase perf sleep period [-S|--perf_sleep_period]"
            )


def remove_exisiting_test_services() -> None:
    """remove all test services"""
    print(f"removing existing test services at {SYSTEMD_SYSTEM_PATH}")
    service_types = ["parallel", "single_path", "dag"]
    services_remove_cmd = assemble_service_gen_cmd(service_types, remove=True)
    run_cmd(services_remove_cmd, ROOT_UID, ROOT_GID)


def run_tests(args: argparse.Namespace) -> None:
    """generate services and run tests on the 2 systemd binaries"""
    print_line(char="#")
    print(
        f"generating test services and running systemd in test mode for {args.tests_num} times..."
    )
    sd_exe_paths = parse_sd_path_mode(args)
    results_dict = collections.defaultdict(list)
    services_remove_cmd = assemble_service_gen_cmd(args.gen_types, remove=True)
    run_perf = args.gen_flamegraph and is_installed("perf")
    profiler = PerfController()
    services_num_list = list(
        range(
            args.start_services_num,
            args.tests_num * args.step_services_num + args.start_services_num,
            args.step_services_num,
        )
    )
    for services_num in services_num_list:
        services_gen_cmd = assemble_service_gen_cmd(
            args.gen_types,
            remove=False,
            services_num=services_num,
            dag_edge_probability=args.dag_edge_probability,
            gen_dot=args.gen_graphviz_dot,
            dot_dir=args.output_files_dir,
        )
        run_cmd(services_gen_cmd, ROOT_UID, ROOT_GID)
        for sd_exe_path in sd_exe_paths:
            if run_perf:
                profiler.run_perf_record(args.perf_frequency, args.perf_sleep_period)
            sd_test_cmd = [
                TIME_CMD_PATH,
                "-p",
                sd_exe_path,
                "--test",
                "--system",
                "--unit",
                "multi-user.target",
                "--no-pager",
            ]
            result = run_cmd(
                sd_test_cmd,
                args.user_uid,
                args.user_gid,
                stdout_file=subprocess.DEVNULL,
                stderr_file=subprocess.PIPE,
            )

            if run_perf:
                profiler.stop_perf_record()
                PerfController.gen_flamegraph(
                    args.output_files_dir, args.output_files_name, services_num
                )

            units_load_time_in_sec = parse_sd_test_cmd_result(result)
            print(f"units load time in seconds = {units_load_time_in_sec} s")
            results_dict[sd_exe_path].append(units_load_time_in_sec)
    rmse = ErrorStatsCalculator.calc_rmse(results_dict)
    print_line(char="#")
    print(f"rmse error = {rmse:.{4}f}")
    print_line(char="#")
    error_stats = ErrorStatsCalculator.calc_error_stats(results_dict)
    reporter = Reporter(
        results_dict,
        services_num_list,
        rmse,
        args.output_files_name,
        args.output_files_dir,
        args.gen_types,
        args.dag_edge_probability,
        error_stats,
    )
    reporter.report_test_results()
    run_cmd(services_remove_cmd, ROOT_UID, ROOT_GID)
    print_line(char="#")
    print("done!")


def parse_args() -> argparse.Namespace:
    """define and parse script arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--sd_path_mode",
        type=str,
        help="use 2 systemd executable paths or 2 systemd upstream commit hashes [exe,commit]",
    )

    parser.add_argument(
        "-e",
        "--sd_ref_exe_path",
        help="reference systemd executable path",
    )

    parser.add_argument(
        "-f",
        "--sd_comp_exe_path",
        help="compared systemd executable path",
    )

    parser.add_argument(
        "-c",
        "--sd_commit_ref",
        help="commit hash of the reference systemd repo",
    )

    parser.add_argument(
        "-d",
        "--sd_commit_comp",
        help="commit hash of the compared systemd repo",
    )
    parser.add_argument(
        "-t",
        "--gen_types",
        action="append",
        help="type of generated test services [parallel,single_path,dag]",
    )

    parser.add_argument(
        "-p",
        "--dag_edge_probability",
        type=float,
        help="edge probability for DAG services generator",
    )

    parser.add_argument(
        "-s",
        "--start_services_num",
        type=int,
        default=DEFAULT_START_SERVICES_NUM,
        help="number of generated services to start the first test",
    )

    parser.add_argument(
        "-j",
        "--step_services_num",
        type=int,
        default=DEFAULT_STEP_SERVICES_NUM,
        help="number of generated services added to the previouse test to jump to the next test",
    )

    parser.add_argument(
        "-n",
        "--tests_num",
        type=int,
        default=DEFAULT_TEST_NUM,
        help="number of tests to run",
    )

    parser.add_argument(
        "-u",
        "--user_uid",
        type=int,
        default=DEFAULT_NONROOT_USER_UID,
        help="user UID to run systemd in test mode",
    )

    parser.add_argument(
        "-g",
        "--user_gid",
        type=int,
        default=DEFAULT_NONROOT_USER_GID,
        help="user GID to run systemd in test mode",
    )

    parser.add_argument(
        "-o",
        "--output_files_name",
        type=str,
        default=DEFAULT_OUTPUT_ARTIFACTS_FILE_NAME,
        help="name of output json data and jpeg plot files",
    )

    parser.add_argument(
        "-r",
        "--output_files_dir",
        type=str,
        default=DEFAULT_OUTPUT_ARTIFACTS_DIR,
        help="output artifacts dir",
    )

    parser.add_argument(
        "-z",
        "--gen_graphviz_dot",
        action="store_true",
        help="generate graphviz dot file",
    )

    parser.add_argument(
        "-l",
        "--gen_flamegraph",
        action="store_true",
        help="generate per test flamegraph",
    )

    parser.add_argument(
        "-F",
        "--perf_frequency",
        type=str,
        default=DEFAULT_PERF_FREQUENCY,
        help="perf profiling frequency",
    )

    parser.add_argument(
        "-S",
        "--perf_sleep_period",
        type=int,
        default=DEFAULT_PERF_SLEEP_PERIOD,
        help="perf command sleep time before stop recording",
    )

    return parser.parse_args()


def main() -> None:
    """main"""

    if os.geteuid() != ROOT_UID:
        fail("please run the script as root")

    if len(sys.argv) <= 1:
        fail("No options used, please use [-h|--help] to list the availbale options.")

    args = parse_args()

    if args.sd_path_mode == "exe" and not all(
        (args.sd_ref_exe_path, args.sd_comp_exe_path)
    ):
        fail("2 systemd exe paths should be used.")

    elif args.sd_path_mode == "commit" and not all(
        (args.sd_commit_ref, args.sd_commit_comp)
    ):
        fail("2 systemd upstream commit hashes should be used.")

    elif not args.sd_path_mode:
        fail("-m|--sd_path_mode should be assigned either exe or commit.")

    if len(args.gen_types) < 1:
        fail(
            "at least 1 service generator type should be used [parallel,single_path,dag]."
        )

    elif args.tests_num < 2:
        fail(
            f"at least 2 tests should be used (-n), default is: {DEFAULT_TEST_NUM} if left empty"
        )

    remove_exisiting_test_services()

    run_tests(args)


if __name__ == "__main__":
    main()
