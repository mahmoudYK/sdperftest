#!/usr/bin/env python3
# License: LGPL-2.1-or-later

import re
import os
import sys
import subprocess
import argparse
import itertools
import random
import pathlib
import dataclasses
import types
import functools
import abc
import glob

# https://github.com/facebookincubator/pystemd
import pystemd.systemd1
import pystemd.dbuslib


SYSTEMD_SYSTEM_PATH = "/run/systemd/system/"
DEFAULT_NUM_OF_SERVICES = 500
DEFAULT_EDGE_PROBABILITY = 0.1
RANDOM_SEED = 2

random.seed(RANDOM_SEED)


@dataclasses.dataclass
class DefaultTemplate:
    """default systemd service file template"""

    template: dict = dataclasses.field(
        default_factory=lambda: {
            "Unit": {"Description": ""},
            "Service": {
                "Type": "simple",
                "RemainAfterExit": "yes",
                "ExecStart": "/usr/bin/sleep 1",
            },
            "Install": {"WantedBy": "multi-user.target"},
        }
    )


def add_service_section(sections_dict: dict, section_name: str) -> list:
    """return a list of key=value strings for a certain section"""
    output_list = [f"[{section_name}]"]
    for key, value in sections_dict[section_name].items():
        output_list.append(f"{key}={value}")
    return output_list


def build_service_template(sections: dict) -> str:
    """build a service template string from sections dictionary"""
    service_text = []
    service_text.extend(add_service_section(sections, "Unit"))
    service_text.append("\n")
    service_text.extend(add_service_section(sections, "Service"))
    service_text.append("\n")
    service_text.extend(add_service_section(sections, "Install"))
    return "\n".join(service_text)


def fail(fail_message: str) -> None:
    """print error message and exit with status 1"""
    print(f"Error: {fail_message}", file=sys.stderr)
    sys.exit(1)


def enable_service(manager: pystemd.systemd1.Manager, service: str) -> None:
    """enable systemd service"""
    status, _ = manager.Manager.EnableUnitFiles([service], False, False)
    if not status:
        fail(f"enabling {service} failed")
    print(f"Enabled Service {service}", end="\r")


def disable_service(manager: pystemd.systemd1.Manager, service: str) -> None:
    """disable systemd service"""
    manager.Manager.DisableUnitFiles([service], False)


def disable_test_services(service_template_prefix: str) -> int:
    """disable all the test services"""
    systemd_system_path = pathlib.Path(SYSTEMD_SYSTEM_PATH)
    disabled_services_counter = 0
    with pystemd.systemd1.Manager() as manager:
        for entry in systemd_system_path.iterdir():
            if entry.is_file() and re.match(
                service_template_prefix + r"[0-9]+\.service", entry.name
            ):
                disable_service(manager, entry.name)
                print(f"Disbaled Service {entry.name}", end="\r")
                disabled_services_counter += 1
    return disabled_services_counter


def remove_test_services(path:str, service_template_prefix: str) -> None:
    """remove all test services which match a certain template prefix"""
    service_suffix = "*.service"
    files_to_remove = glob.glob(path+service_template_prefix+service_suffix)
    if files_to_remove:
        for file_name in files_to_remove:
            os.remove(file_name)
        

def write_service_file(path: str, service_text: str) -> None:
    """write service text to a .service file at path"""
    with open(path, "w", encoding="utf-8") as service_writer:
        try:
            service_writer.write(service_text)
        except OSError:
            fail(f"Can't write service{path}")

def print_time_from_microseconds(time_var: str, val: int) -> None:
    """write property = val (m|s|ms|us) to console"""
    time_in_seconds = val / 1000000
    if time_in_seconds >= 60:
        time_in_minutes = int(time_in_seconds // 60)
        time_in_seconds -= time_in_minutes * 60
        print(f"{time_var} = {time_in_minutes}m {time_in_seconds:.{3}f}s")
    elif 60 > time_in_seconds >= 1:
        print(f"{time_var} = {time_in_seconds:.{3}f}s")
    elif 1 > time_in_seconds >= 0.001:
        print(f"{time_var} = {time_in_seconds * 1000:.{3}f}ms")
    else:
        print(f"{time_var} = {time_in_seconds * 1000000:.{3}f}us")


def to_namespace(func):
    '''decorator to convert a dict to namespace object'''
    def wrapper(*args, **kwargs):
        return types.SimpleNamespace(**func(*args, **kwargs))

    return functools.update_wrapper(wrapper, func)


@to_namespace
def get_systemd_properties() -> dict:
    """call GetAll systemd dbus method and get all the exposed properties"""
    properties = {}
    cargs = pystemd.dbuslib.apply_signature(
        b"s",  # signature
        [""],
    )
    with pystemd.dbuslib.DBus() as bus:
        properties = bus.call_method(
            b"org.freedesktop.systemd1",
            b"/org/freedesktop/systemd1",
            b"org.freedesktop.DBus.Properties",
            b"GetAll",
            cargs,
        ).body
    return {k.decode(): v for k, v in properties.items()}


def analyze_time() -> None:
    """acquire and display boot time of systemd"""
    properties = get_systemd_properties()
    # print(properties)
    if properties.FirmwareTimestampMonotonic > 0:
        print_time_from_microseconds(
            "firmware_time",
            properties.FirmwareTimestampMonotonic - properties.LoaderTimestampMonotonic,
        )

    if properties.LoaderTimestampMonotonic > 0:
        print_time_from_microseconds("loader_time", properties.LoaderTimestampMonotonic)

    if properties.InitRDTimestampMonotonic > 0:
        print_time_from_microseconds(
            "kernel_done_time", properties.InitRDTimestampMonotonic
        )
        print_time_from_microseconds(
            "initrd_time",
            properties.UserspaceTimestampMonotonic
            - properties.InitRDTimestampMonotonic,
        )
    else:
        print_time_from_microseconds(
            "kernel_done_time", properties.UserspaceTimestampMonotonic
        )

    print_time_from_microseconds(
        "userspace_time",
        properties.FinishTimestampMonotonic - properties.UserspaceTimestampMonotonic,
    )
    print_time_from_microseconds(
        "startup_finish_time",
        properties.FirmwareTimestampMonotonic + properties.FinishTimestampMonotonic,
    )
    print_time_from_microseconds(
        "security_module_setup_time",
        properties.SecurityFinishTimestampMonotonic
        - properties.SecurityStartTimestampMonotonic,
    )
    print_time_from_microseconds(
        "generators_time",
        properties.GeneratorsFinishTimestampMonotonic
        - properties.GeneratorsStartTimestampMonotonic,
    )
    print_time_from_microseconds(
        "units_load_time",
        properties.UnitsLoadFinishTimestampMonotonic
        - properties.UnitsLoadStartTimestampMonotonic,
    )


class ServiceGeneratorInterface(abc.ABC):
    '''abstract service generator class, defines a set of must implement methods'''

    @abc.abstractmethod
    def get_test_service_prefix(self) -> str:
        """return unique test file name per generator class"""

    @abc.abstractmethod
    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""

    @abc.abstractmethod
    def __str__(self) -> str:
        """return the generator type as a string"""

class ParallelServices(ServiceGeneratorInterface):
    """generator of services that doesn't have any dependencies between each other"""

    def __init__(self) -> None:
        self.test_file_prefix = "test_parallel"

    def create_parallel_services_template(self, properties: dict) -> str:
        """generate parallel service template string"""
        properties["Unit"]["Description"] = "parallel test service {0}"
        return build_service_template(properties)

    def gen_service_text(self, template: str, service_num: int) -> str:
        """generate parallel service text"""
        return template.format(service_num)

    def get_test_service_prefix(self) -> str:
        """return the string: test_parallel"""
        return self.test_file_prefix

    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""

        test_file_name = self.test_file_prefix + "{0}.service"
        service_template = self.create_parallel_services_template(
            DefaultTemplate().template
        )

        with pystemd.systemd1.Manager() as manager:
            for i in range(num_of_services):
                write_service_file(
                    path + test_file_name.format(i),
                    self.gen_service_text(service_template, i),
                )
                enable_service(manager, test_file_name.format(i))

        return num_of_services

    def __str__(self) -> str:
        """return the string: parallel"""
        return "parallel"


class SinglePathServices(ServiceGeneratorInterface):
    """generator of services with only 1 After dependency on the previous test service"""

    def __init__(self) -> None:
        self.test_file_prefix = "test_single_path"

    def create_single_path_services_template(
        self, properties: dict, first_service: bool
    ) -> str:
        """generate single path service template string"""
        properties["Unit"]["Description"] = "single path test service {0}"
        if not first_service:
            properties["Unit"]["After"] = "test_single_path{1}.service"
        return build_service_template(properties)

    def gen_service_text(
        self, template: str, service_num: int, prev_service_num: int
    ) -> str:
        """generate single path service text"""
        if service_num == 0:
            # 'After=' line is not existing in test0.service instance as it should be the root
            return template.format(service_num)
        return template.format(service_num, prev_service_num)

    def get_test_service_prefix(self) -> str:
        """return the string: test_single_path"""
        return self.test_file_prefix

    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""

        test_file_name = self.test_file_prefix + "{0}.service"
        service_template = self.create_single_path_services_template(
            DefaultTemplate().template, False
        )
        first_service_template = self.create_single_path_services_template(
            DefaultTemplate().template, True
        )

        with pystemd.systemd1.Manager() as manager:
            for i in range(num_of_services):
                write_service_file(
                    path + test_file_name.format(i),
                    self.gen_service_text(
                        service_template if i > 0 else first_service_template, i, i - 1
                    ),
                )
                enable_service(manager, test_file_name.format(i))

        return num_of_services

    def __str__(self) -> str:
        """return the string: single_path"""
        return "single_path"


class DAGServices(ServiceGeneratorInterface):
    """generator of DAG services that may have many or no dependencies between each other"""

    def __init__(self, edge_probability_arg: float) -> None:
        self.test_file_prefix = "test_DAG"
        self.edge_probability = edge_probability_arg

    def create_dag_services_template(self, properties: dict, parallel: bool) -> str:
        """generate DAG service template string"""
        properties["Unit"]["Description"] = "DAG test service {0}"
        if not parallel:
            properties["Unit"]["Before"] = "{1}"
        return build_service_template(properties)

    def gen_service_text(
        self, template: str, service_name: str, services_list: dict
    ) -> str:
        """generate DAG services text"""
        first_edge = services_list[0]
        service_num = first_edge[0]
        before_service_names = service_name.format(
            first_edge[1]
        )  # start without space. eg, Before=test_dag0.service test_dag1.service ...
        if len(services_list) > 1:
            for _, node in services_list:
                if not str(node) in before_service_names:
                    before_service_names += " " + service_name.format(
                        node
                    )  # space separated list of services
        return template.format(service_num, before_service_names)

    def get_test_service_prefix(self) -> str:
        """return the string: test_DAG"""
        return self.test_file_prefix

    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""
        test_file_name = self.test_file_prefix + "{0}.service"
        service_template = self.create_dag_services_template(
            DefaultTemplate().template, False
        )
        parallel_service_template = self.create_dag_services_template(
            DefaultTemplate().template, True
        )

        # enumerate a directed acyclic graph (DAG)
        edges = itertools.permutations(range(num_of_services), 2)
        previous_service = -1
        current_service_with_edges = 0
        last_service_num = num_of_services - 1
        edge_list = []

        with pystemd.systemd1.Manager() as manager:
            for edge in edges:
                first_node = edge[0]
                second_node = edge[1]

                if first_node - previous_service == 1:
                    write_service_file(
                        path + test_file_name.format(first_node),
                        ParallelServices().gen_service_text(parallel_service_template, first_node),
                    )
                    enable_service(manager, test_file_name.format(first_node))
                    previous_service = first_node

                if (
                    random.random() < self.edge_probability and first_node < second_node
                ) or (
                    first_node == last_service_num
                ):  # (first_node < second_node) to make sure that the graph is acyclic
                    if current_service_with_edges != first_node and edge_list:
                        write_service_file(
                            path + test_file_name.format(current_service_with_edges),
                            self.gen_service_text(
                                service_template, test_file_name, edge_list
                            ),
                        )
                        enable_service(
                            manager, test_file_name.format(current_service_with_edges)
                        )
                        edge_list.clear()

                    if first_node != last_service_num:
                        edge_list.append(edge)
                        current_service_with_edges = first_node
                    else:
                        break

        return num_of_services

    def __str__(self) -> str:
        """return the string: dag"""
        return "dag"


def parse_args() -> argparse.Namespace:
    """define and parse script arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_of_services",
        action="append",
        help="number of services to generate",
    )
    parser.add_argument(
        "-t",
        "--type",
        action="append",
        help="type of generated test services [parallel,single_path,dag]",
    )
    parser.add_argument(
        "-g",
        "--generate",
        help="generate and enable all the test services",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--remove",
        help="disbale and remove all the test services",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--edge_probability",
        type=float,
        default=DEFAULT_EDGE_PROBABILITY,
        help="edge probability for DAG services generator",
    )
    parser.add_argument(
        "-a",
        "--analyze",
        help="analyze systemd boot time",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    '''main'''
    if os.geteuid() != 0:
        fail("please run the script as root")

    if len(sys.argv) <= 1:
        fail("No options used, please use [-h|--help] to list the availbale options.")

    args = parse_args()

    if args.type:
        generator_types = [
            SinglePathServices(),
            ParallelServices(),
            DAGServices(args.edge_probability),
        ]
        generator_types_str = [str(x) for x in generator_types]
        # if numbr of -n less than number of -t, use the last -n by default
        if args.num_of_services:
            num_of_services = args.num_of_services + [args.num_of_services[-1]] * (
                len(args.type) - len(args.num_of_services)
            )
        else:
            num_of_services = [DEFAULT_NUM_OF_SERVICES] * len(args.type)

        for type_arg in args.type:
            if type_arg in generator_types_str:
                obj = generator_types[generator_types_str.index(type_arg)]

                if args.remove and args.generate:
                    fail("can't set generate and remove together")

                elif args.remove:
                    services_count = disable_test_services(
                        obj.get_test_service_prefix()
                    )
                    remove_test_services(SYSTEMD_SYSTEM_PATH, obj.get_test_service_prefix())
                    print(
                        f"disabled and removed {services_count} services of type {str(obj)}"
                    )

                else:
                    # generate by default
                    args_num_of_services = int(num_of_services.pop(0))
                    services_count = obj.gen_test_services(
                        SYSTEMD_SYSTEM_PATH, args_num_of_services
                    )
                    print(
                        f"generated and enabled {services_count} test services of type {str(obj)}"
                    )

    if args.analyze:
        analyze_time()


if __name__ == "__main__":
    main()
