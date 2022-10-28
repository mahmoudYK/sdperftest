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
import git
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DEFAULT_START_SERVICES_NUM = 100
DEFAULT_STEP_SERVICES_NUM = 100
DEFAULT_TEST_NUM = 10
ROOT_UID = 0
ROOT_GID = 0
DEFAULT_NONROOT_USER_UID = 1000
DEFAULT_NONROOT_USER_GID = 1000
PYTHON_PATH = "/usr/bin/python"
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


@dataclasses.dataclass
class OutputArtifactsInfo:
    sdpath_loadtime_dict: dict[str,list[float]]
    generated_services_num_list: list[int]
    rmse: float
    output_files_name: str
    output_files_dir: str
    generator_types: list[str]
    dag_edge_probability: float
    error_stats: ErrorStats


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


def assemble_service_gen_cmd(
    types: list,
    remove: bool,
    services_num: int = None,
    dag_edge_probability: float = None,
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
    return cmd


def run_cmd(
    cmd: list, uid: int, gid: int, capture: bool = False
) -> subprocess.CompletedProcess:
    """using subprocess to run a command and return subprocess.CompletedProcess"""
    cmd_str = " ".join(cmd)
    print(f"running: {cmd_str}")
    try:
        if capture:
            return subprocess.run(
                cmd, check=True, capture_output=True, text=True, user=uid, group=gid
            )
        return subprocess.run(cmd, check=True, user=uid, group=gid)
    except subprocess.CalledProcessError as ex:
        fail(f"running {cmd_str} failed, returncode: {ex.returncode}")


def parse_sd_path_mode(args:argparse.Namespace) -> list:
    #args.sd_path_mode == "exe" and (not args.sd_ref_exe_path or not args.sd_comp_exe_path)
    match args.sd_path_mode:
        case "exe":
            return [args.sd_ref_exe_path, args.sd_comp_exe_path]
        case "commit":
            commit_hash_list = [args.sd_commit_ref, args.sd_commit_comp]
            sd_exe_path_list = []
            os.chmod(SD_BUILD_SCRIPT,0o755)
            for commit_hash in commit_hash_list:
                sd_build_cmd = [SD_BUILD_SCRIPT, "-d", SD_BUILD_DIR, "-c", commit_hash]
                result = run_cmd(sd_build_cmd, args.user_uid, args.user_gid, capture=True)
                # exe path example:
                # sd_build_load_test/6fadf01cf3cdd98f78b7829f4c6c892306958394/systemd/build/systemd
                sd_exe_path_list.append(os.path.join(SD_BUILD_DIR, commit_hash,"systemd","build","systemd"))
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
                    load_time += t_value / 1000000
                case "ms":
                    load_time += t_value / 1000
                case "s":
                    load_time += t_value
                case "min":
                    load_time += t_value * 60
                case "h":
                    load_time += t_value * 60 * 60
                case _:
                    load_time = float(math.inf)
        return load_time

    return functools.update_wrapper(wrapper, func)


@to_seconds
def parse_sd_test_cmd_result(result: subprocess.CompletedProcess) -> str:
    """parse systemd --test stderr to extract units load time in seconds"""
    units_load_time_line = "Loaded units and determined initial transaction in"
    units_load_time = re.match(
        r"\d.*\.$", result.stderr.split(units_load_time_line)[1].strip()
    )
    if units_load_time:
        return units_load_time.group(0).removesuffix(".")
    fail("can't parse the stderr output of the cmd: systemd --test.")


def plot(info: OutputArtifactsInfo) -> None:
    """plot units load time against the number of test services"""
    print("plotting ...")
    fig = plt.figure(figsize=(25, 15))
    generator_types_str = "generator types: [" + " ".join(info.generator_types) + "]"
    if "dag" in info.generator_types:
        generator_types_str += f"\ndag edge probability = {info.dag_edge_probability}"
    rmse_str = f"RMSE: {info.rmse:.{4}f}"
    fig.suptitle(
        f"units load time test\n\n{generator_types_str}\n{rmse_str}", fontsize=20
    )
    colors = ["green", "red"]
    horizontal_alignments = ["right", "left"]
    offsets = [-1, 1]
    patches = []
    x_ticks = []
    for sd_path, color, horizontal_alignment, offset in zip(
        info.sdpath_loadtime_dict.keys(), colors, horizontal_alignments, offsets
    ):
        load_time = info.sdpath_loadtime_dict[sd_path]
        plt.plot(info.generated_services_num_list, load_time, color=color, linewidth=1)
        patches.append(mpatches.Patch(color=color, label=sd_path))
        x_ticks, _ = plt.xticks()
        x_offset = (x_ticks[1] - x_ticks[0]) / 25
        for idx, num_services in enumerate(info.generated_services_num_list):
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
    for idx, num_services in enumerate(info.generated_services_num_list):
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
    fig_path = os.path.join(info.output_files_dir, info.output_files_name + ".jpg")
    plt.savefig(fig_path, dpi=250)
    print(f"saved figure at {fig_path}")


def write_json(info: OutputArtifactsInfo) -> None:
    """write test results to json file"""
    print("writing json file ...")
    json_str_dict = {}
    for res_dict_key, res_dict_value in info.sdpath_loadtime_dict.items():
        json_str_dict[res_dict_key] = {
            "num of services:service load time": dict(
                zip(info.generated_services_num_list, res_dict_value)
            )
        }
    json_str_dict["generator types"] = info.generator_types
    if "dag" in info.generator_types:
        json_str_dict["dag edge probability"] = info.dag_edge_probability
    json_str_dict["rmse"] = round(info.rmse, 4)
    json_file_path = os.path.join(
        info.output_files_dir, info.output_files_name + ".json"
    )
    with open(
        json_file_path,
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(json_str_dict, json_file, indent=2)
    print(f"saved json data at {json_file_path}")


def calc_rmse(res_dict: dict) -> float:
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


def calc_error_stats(res_dict: dict) -> ErrorStats:
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


def print_error_stats(info: OutputArtifactsInfo) -> None:
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
    ref_sd_load_time_list = list(info.sdpath_loadtime_dict.values())[0]
    compared_sd_load_time_list = list(info.sdpath_loadtime_dict.values())[1]
    for test_num, load_time in enumerate(ref_sd_load_time_list):
        print(
            f"{info.generated_services_num_list[test_num] : <20}\
{load_time : <20}\
{compared_sd_load_time_list[test_num] : <20}\
{info.error_stats.abs_error[test_num] : <25}\
{info.error_stats.percent_error[test_num] : <25}"
        )
    print_line(length=120)
    print(
        f"{'measure' : <20}{' ' : <40}{'absolute error' : <25}{'percentage error(%)' : <25}"
    )
    print_line(length=120)
    print(
        f"{'max' : <20}{' ' : <40}\
{info.error_stats.max_abs : <25}\
{info.error_stats.max_percent : <25}"
    )
    print(
        f"{'min' : <20}{' ' : <40}\
{info.error_stats.min_abs : <25}\
{info.error_stats.min_percent : <25}"
    )
    print(
        f"{'mean' : <20}{' ' : <40}\
{info.error_stats.mean_abs : <25}\
{info.error_stats.mean_percent : <25}"
    )
    print(
        f"{'stddev' : <20}{' ' : <40}\
{info.error_stats.stddev_abs : <25}\
{info.error_stats.stddev_percent : <25}"
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
            False,
            services_num,
            args.dag_edge_probability,
        )
        run_cmd(services_gen_cmd, ROOT_UID, ROOT_GID)
        for sd_exe_path in sd_exe_paths:
            sd_test_cmd = [
                sd_exe_path,
                "--test",
                "--system",
                "--unit",
                "multi-user.target",
                "--no-pager",
            ]
            result = run_cmd(sd_test_cmd, args.user_uid, args.user_gid, capture=True)
            units_load_time_in_sec = parse_sd_test_cmd_result(result)
            print(f"units load time in seconds = {units_load_time_in_sec} s")
            results_dict[sd_exe_path].append(units_load_time_in_sec)
    rmse = calc_rmse(results_dict)
    print_line(char="#")
    print(f"rmse error = {rmse:.{4}f}")
    print_line(char="#")
    error_stats = calc_error_stats(results_dict)
    info = OutputArtifactsInfo(
        results_dict,
        services_num_list,
        rmse,
        args.output_files_name,
        args.output_files_dir,
        args.gen_types,
        args.dag_edge_probability,
        error_stats,
    )
    plot(info)
    print_line(char="#")
    write_json(info)
    print_line(char="#")
    print_error_stats(info)
    print_line(char="#")
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
        "-c",
        "--sd_commit_ref",
        help="commit hash of the reference systemd repo",
    )

    parser.add_argument(
        "-d",
        "--sd_commit_comp",
        help="commit hash of the compared systemd repo",
    )

    return parser.parse_args()


def main() -> None:
    """main"""

    if os.geteuid() != ROOT_UID:
        fail("please run the script as root")

    if len(sys.argv) <= 1:
        fail("No options used, please use [-h|--help] to list the availbale options.")

    args = parse_args()

    if args.sd_path_mode == "exe" and not all((args.sd_ref_exe_path, args.sd_comp_exe_path)):
        fail("2 systemd exe paths should be used.")

    elif args.sd_path_mode == "commit" and not all((args.sd_commit_ref, args.sd_commit_comp)):
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
