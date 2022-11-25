#!/usr/bin/env python3
# License: LGPL-2.1-or-later

import os
import sys
import argparse
import itertools
import random
import dataclasses
import types
import functools
import abc
import glob
import subprocess
import decimal
import typing
import graphviz

# https://github.com/facebookincubator/pystemd
import pystemd.dbuslib


SYSTEMD_SYSTEM_PATH = "/run/systemd/system/"
DEFAULT_NUM_OF_SERVICES = 500
DEFAULT_EDGE_PROBABILITY = 0.1
RANDOM_SEED = 2
DEFAULT_GRAPHVIZ_DOT_OUTPUT_DIR = os.getcwd()
SYSTEMCTL_EXE_PATH = "/usr/bin/systemctl"
RM_EXE_PATH = "/usr/bin/rm"

random.seed(RANDOM_SEED)

_ServiceTempType = dict[str,dict[str,str]]

@dataclasses.dataclass
class DefaultTemplate:
    """default systemd service file template"""

    template: _ServiceTempType = dataclasses.field(
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


def add_service_section(sections_dict: _ServiceTempType, section_name: str) -> list[str]:
    """return a list of key=value strings for a certain section"""
    output_list = [f"[{section_name}]"]
    for key, value in sections_dict[section_name].items():
        output_list.append(f"{key}={value}")
    return output_list


def build_service_template(sections: _ServiceTempType) -> str:
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


_FILE = typing.Union[None, int, typing.IO[typing.Any]]

# security considerations of shell=True:
# https://docs.python.org/3/library/subprocess.html#security-considerations
def run_shell_cmd(
    cmd: str,
    stdout: _FILE = subprocess.DEVNULL,
    stderr: _FILE = subprocess.DEVNULL,
) -> None:
    """run cmd with shell=True"""
    try:
        subprocess.run(cmd, shell=True, stdout=stdout, stderr=stderr, check=True)
    except subprocess.CalledProcessError as ex:
        fail(f"running {cmd} failed:\n{ex}")


def control_test_services(service_template_prefix: str, verb: str) -> int:
    """apply systemctl verb on all the test services"""
    service_suffix = "*.service"
    services_num = len(
        glob.glob1(SYSTEMD_SYSTEM_PATH, f"{service_template_prefix}{service_suffix}")
    )
    if services_num:
        run_shell_cmd(
            f"{SYSTEMCTL_EXE_PATH} {verb} --root=/ {SYSTEMD_SYSTEM_PATH}{service_template_prefix}{service_suffix}"
        )
    return services_num


def enable_test_services(service_template_prefix: str, generator_type: str) -> int:
    """enable systemd test services"""
    print(f"\nenable {generator_type} generated test_services...")
    return control_test_services(service_template_prefix, "enable")


def disable_test_services(service_template_prefix: str, generator_type: str) -> int:
    """disable systemd test services"""
    print(f"disable {generator_type} generated test_services...")
    return control_test_services(service_template_prefix, "disable")


def remove_test_services(
    path: str, service_template_prefix: str, generator_type: str
) -> None:
    """remove all test services which match a certain template prefix"""
    print(f"remove {generator_type} generated test_services...")
    service_suffix = "*.service"
    files_to_remove = glob.glob1(path, service_template_prefix + service_suffix)
    if files_to_remove:
        run_shell_cmd(f"{RM_EXE_PATH} {path}{service_template_prefix}{service_suffix}")


def write_service_file(path: str, service_text: str) -> None:
    """write service text to a .service file at path"""
    with open(path, "w", encoding="utf-8") as service_writer:
        try:
            service_writer.write(service_text)
        except OSError:
            fail(f"Can't write service{path}")


def print_time_from_microseconds(time_var: str, val: int) -> None:
    """write property = val (m|s|ms|us) to console"""
    decimal.getcontext().rounding = decimal.ROUND_DOWN
    time_in_seconds =decimal.Decimal(val / 1000000)
    if time_in_seconds >= 60:
        time_in_minutes = int(time_in_seconds // 60)
        time_in_seconds -= time_in_minutes * 60
        print(f"{time_var} = {time_in_minutes}min {float(round(time_in_seconds, 3))}s")
    elif 60 > time_in_seconds >= 1:
        print(f"{time_var} = {float(round(time_in_seconds, 3))}s")
    elif 1 > time_in_seconds >= 0.001:
        print(f"{time_var} = {float(round(time_in_seconds * 1000, 3))}ms")
    else:
        print(f"{time_var} = {float(round(time_in_seconds * 1000000, 3))}us")


def to_namespace(func):
    """decorator to convert a dict to namespace object"""

    def wrapper(*args, **kwargs):
        return types.SimpleNamespace(**func(*args, **kwargs))

    return functools.update_wrapper(wrapper, func)


@to_namespace
def get_systemd_properties() -> dict[str,int]:
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

_EdgeListType = list[tuple[int,int]]
_NodeListType = list[int]

def gen_dot_file(
    node_list: _NodeListType,
    edge_list: _EdgeListType,
    digraph_name: str,
    digraph_comment: str,
    digraph_filename: str,
    digraph_dir: str,
    node_label_template: str,
) -> None:
    """generate graphviz dot file"""
    print(f"Generate graphviz dot file for {len(node_list)} {digraph_name} services...")
    dot = graphviz.Digraph(
        name=digraph_name,
        comment=digraph_comment,
        filename=digraph_filename,
    )
    for node in node_list:
        dot.node(str(node), node_label_template + f"{node}.service")
    for edge in edge_list:
        dot.edge(str(edge[0]), str(edge[1]))
    dot.save(filename=digraph_filename, directory=digraph_dir)
    print(f"Generated {digraph_filename} at {digraph_dir}.")


class ServiceGeneratorInterface(abc.ABC):
    """abstract service generator class, defines a set of must implement methods"""

    def __init__(self, test_file_prefix: str, node_list: _NodeListType, edge_list: _EdgeListType) -> None:
        self.__test_file_prefix = test_file_prefix
        self.__node_list = node_list
        self.__edge_list = edge_list
        super().__init__()

    @property
    def test_service_prefix(self) -> str:
        """return unique test file name per generator class"""
        return self.__test_file_prefix

    @property
    def nodes(self) -> _NodeListType:
        """return the list of all the nodes(service numbers)"""
        return self.__node_list

    @property
    def edges(self) -> _EdgeListType:
        """return the list of all the edges(dependencies bwn services)"""
        return self.__edge_list

    @abc.abstractmethod
    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""

    @abc.abstractmethod
    def __str__(self) -> str:
        """return the generator type as a string"""


class ParallelServices(ServiceGeneratorInterface):
    """generator of services that doesn't have any dependencies between each other"""

    def __init__(self, gen_dot: bool = False) -> None:
        self.__gen_dot = gen_dot
        self.__test_file_prefix = "test_parallel"
        self.__node_list = []
        self.__edge_list = []
        super().__init__(self.__test_file_prefix, self.__node_list, self.__edge_list)

    def create_parallel_services_template(self, properties: _ServiceTempType) -> str:
        """generate parallel service template string"""
        properties["Unit"]["Description"] = "parallel test service {0}"
        return build_service_template(properties)

    def gen_service_text(self, template: str, service_num: int) -> str:
        """generate parallel service text"""
        return template.format(service_num)

    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""

        test_file_name = self.__test_file_prefix + "{0}.service"
        service_template = self.create_parallel_services_template(
            DefaultTemplate().template
        )
        for i in range(num_of_services):
            write_service_file(
                path + test_file_name.format(i),
                self.gen_service_text(service_template, i),
            )
            print(f"Generated {i+1} services", end="\r")
            if self.__gen_dot:
                self.__node_list.append(i)

        return num_of_services

    def __str__(self) -> str:
        """return the string: parallel"""
        return "parallel"


class SinglePathServices(ServiceGeneratorInterface):
    """generator of services with only 1 After dependency on the previous test service"""

    def __init__(self, gen_dot: bool = False) -> None:
        self.__gen_dot = gen_dot
        self.__test_file_prefix = "test_single_path"
        self.__node_list = []
        self.__edge_list = []
        super().__init__(self.__test_file_prefix, self.__node_list, self.__edge_list)

    def create_single_path_services_template(
        self, properties: _ServiceTempType, first_service: bool
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

    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""

        test_file_name = self.__test_file_prefix + "{0}.service"
        service_template = self.create_single_path_services_template(
            DefaultTemplate().template, False
        )
        first_service_template = self.create_single_path_services_template(
            DefaultTemplate().template, True
        )
        for i in range(num_of_services):
            write_service_file(
                path + test_file_name.format(i),
                self.gen_service_text(
                    service_template if i > 0 else first_service_template, i, i - 1
                ),
            )
            print(f"Generated {i+1} services", end="\r")
            if self.__gen_dot:
                self.__node_list.append(i)
                if i > 0:
                    self.__edge_list.append((i - 1, i))

        return num_of_services

    def __str__(self) -> str:
        """return the string: single_path"""
        return "single_path"


class DAGServices(ServiceGeneratorInterface):
    """generator of DAG services that may have many or no dependencies between each other"""

    def __init__(self, edge_probability_arg: float, gen_dot: bool = False) -> None:
        self.__edge_probability = edge_probability_arg
        self.__gen_dot = gen_dot
        self.__test_file_prefix = "test_DAG"
        self.__node_list = []
        self.__edge_list = []
        super().__init__(self.__test_file_prefix, self.__node_list, self.__edge_list)

    def create_dag_services_template(self, properties: _ServiceTempType, parallel: bool) -> str:
        """generate DAG service template string"""
        properties["Unit"]["Description"] = "DAG test service {0}"
        if not parallel:
            properties["Unit"]["Before"] = "{1}"
        return build_service_template(properties)

    def gen_service_text(
        self, template: str, service_name: str, services_list: list[tuple[int,int]]
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

    def gen_test_services(self, path: str, num_of_services: int) -> int:
        """write the test service text to num_of_services service files"""
        test_file_name = self.__test_file_prefix + "{0}.service"
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

        for edge in edges:
            first_node = edge[0]
            second_node = edge[1]

            # each node represents a service and it should be written to
            # SYSTEMD_SYSTEM_PATH even if it has an outdegree of 0.
            # if the node has an outdegree > 0, the service file wil be overwritten,
            # if not, it will be generated as a parallel service.
            if first_node - previous_service == 1:
                write_service_file(
                    path + test_file_name.format(first_node),
                    ParallelServices().gen_service_text(
                        parallel_service_template, first_node
                    ),
                )
                print(f"Generated {first_node+1} services", end="\r")
                previous_service = first_node
                if self.__gen_dot:
                    self.__node_list.append(first_node)

            # edge_list is full and a new node number is started, write
            # the edge_list to the previous node number service dependency list,
            # and clear the edge_list to be used for the current new node.
            if current_service_with_edges != first_node and edge_list:
                write_service_file(
                    path + test_file_name.format(current_service_with_edges),
                    self.gen_service_text(service_template, test_file_name, edge_list),
                )
                edge_list.clear()

            # (first_node < second_node) to make sure that the graph is directed acyclic graph
            if random.random() < self.__edge_probability and first_node < second_node:
                edge_list.append(edge)
                current_service_with_edges = first_node
                if self.__gen_dot:
                    self.__edge_list.append((first_node, second_node))

            # no need to process the rest of permutations if first node in the edge
            # is the last node, because all node numbers < the last node number.
            # the last node number has been generated as a parallel service already.
            if first_node == last_service_num:
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
    parser.add_argument(
        "-z",
        "--gen_graphviz_dot",
        action="store_true",
        help="generate graphviz dot file",
    )
    parser.add_argument(
        "-d",
        "--dot_dir",
        default=DEFAULT_GRAPHVIZ_DOT_OUTPUT_DIR,
        help="graphviz dot file output directory",
    )
    return parser.parse_args()


def main() -> None:
    """main"""
    if os.geteuid() != 0:
        fail("please run the script as root")

    if len(sys.argv) <= 1:
        fail("No options used, please use [-h|--help] to list the availbale options.")

    args = parse_args()

    if args.type:
        generator_types = [
            SinglePathServices(args.gen_graphviz_dot),
            ParallelServices(args.gen_graphviz_dot),
            DAGServices(args.edge_probability, args.gen_graphviz_dot),
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
                        obj.test_service_prefix, str(obj)
                    )
                    remove_test_services(
                        SYSTEMD_SYSTEM_PATH, obj.test_service_prefix, str(obj)
                    )
                    print(
                        f"disabled and removed {services_count} services of type {str(obj)}"
                    )

                else:
                    # generate by default
                    args_num_of_services = int(num_of_services.pop(0))
                    services_count = obj.gen_test_services(
                        SYSTEMD_SYSTEM_PATH, args_num_of_services
                    )
                    enable_test_services(obj.test_service_prefix, str(obj))
                    print(
                        f"generated and enabled {services_count} test services of type {str(obj)}"
                    )

                    if args.gen_graphviz_dot:
                        gen_dot_file(
                            obj.nodes,
                            obj.edges,
                            str(obj),
                            obj.test_service_prefix + " digraph",
                            obj.test_service_prefix + f"_{services_count}_services.dot",
                            args.dot_dir,
                            obj.test_service_prefix,
                        )

    if args.analyze:
        analyze_time()


if __name__ == "__main__":
    main()
