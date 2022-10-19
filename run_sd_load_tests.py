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
SYSTEMD_SYSTEM_PATH = "/etc/systemd/system/"
DEFAULT_OUTPUT_ARTIFACTS_FILE_NAME = "sd_load_test"


def fail(fail_message: str) -> None:
    """print error message and exit with status 1"""
    print(f"Error: {fail_message}", file=sys.stderr)
    sys.exit(1)


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


def plot(
    res_dict: dict, num_services_list: list, rmse: float, output_file_name: str
) -> None:
    """plot units load time against the number of test services"""
    fig = plt.figure(figsize=(25, 15))
    fig.suptitle(f"units load time test\n(RMSE = {rmse:.{4}f})", fontsize=20)
    colors = ["green", "red"]
    horizontal_alignments = ["right", "left"]
    offsets = [-1, 1]
    patches = []
    x_ticks = []
    for sd_path, color, horizontal_alignment, offset in zip(
        res_dict.keys(), colors, horizontal_alignments, offsets
    ):
        load_time = res_dict[sd_path]
        plt.plot(num_services_list, load_time, color=color, linewidth=1)
        patches.append(mpatches.Patch(color=color, label=sd_path))
        x_ticks, _ = plt.xticks()
        x_offset = (x_ticks[1] - x_ticks[0]) / 25
        for idx, num_services in enumerate(num_services_list):
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
    for idx, num_services in enumerate(num_services_list):
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
    plt.savefig(output_file_name + ".jpg", dpi=250)


def write_json(
    res_dict: dict, services_num_list: list, rmse: float, output_file_name: str
) -> None:
    """write test results to json file"""
    json_str_dict = {}
    for res_dict_key, res_dict_value in res_dict.items():
        json_str_dict[res_dict_key] = {
            "num of services:service load time": dict(
                zip(services_num_list, res_dict_value)
            )
        }
    json_str_dict["rmse"] = round(rmse, 4)
    with open(output_file_name + ".json", "w", encoding="utf-8") as json_file:
        json.dump(json_str_dict, json_file, indent=2)


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


def remove_exisiting_test_services() -> None:
    """remove all test services"""
    service_types = ["parallel", "single_path", "dag"]
    services_remove_cmd = assemble_service_gen_cmd(service_types, remove=True)
    run_cmd(services_remove_cmd, ROOT_UID, ROOT_GID)


def run_tests(args: argparse.Namespace) -> None:
    """generate services and run tests on the 2 systemd binaries"""
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
        for sd_bin_path in args.systemd_bin_path:
            sd_test_cmd = [sd_bin_path, "--test", "--system", "--no-pager"]
            result = run_cmd(sd_test_cmd, args.user_uid, args.user_gid, capture=True)
            units_load_time_in_sec = parse_sd_test_cmd_result(result)
            print(f"units load time in seconds = {units_load_time_in_sec} s")
            results_dict[sd_bin_path].append(units_load_time_in_sec)
    rmse = calc_rmse(results_dict)
    print(f"rmse error = {rmse:.{4}f}")
    plot(results_dict, services_num_list, rmse, args.output_files_name)
    write_json(results_dict, services_num_list, rmse, args.output_files_name)
    run_cmd(services_remove_cmd, ROOT_UID, ROOT_GID)


def parse_args() -> argparse.Namespace:
    """define and parse script arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--systemd_bin_path",
        action="append",
        help="list of systemd bin directories",
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

    return parser.parse_args()


def main() -> None:
    """main"""

    if os.geteuid() != ROOT_UID:
        fail("please run the script as root")

    if len(sys.argv) <= 1:
        fail("No options used, please use [-h|--help] to list the availbale options.")

    args = parse_args()

    if len(args.systemd_bin_path) != 2:
        fail("at least 2 systemd bin dirs should be used.")

    elif len(args.gen_types) < 1:
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