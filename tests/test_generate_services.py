import sys
import os
import contextlib
import io
import unittest
import subprocess
import types
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_services


class TestGenerateServicesFns(unittest.TestCase):
    def test_add_service_section(self):
        template = generate_services.DefaultTemplate().template
        template["Unit"]["Description"] = "single path test service 1"
        template["Unit"]["After"] = "test_single_path0.service"
        self.assertEqual(
            generate_services.add_service_section(template, "Unit"),
            [
                "[Unit]",
                "Description=single path test service 1",
                "After=test_single_path0.service",
            ],
        )

    def __redirect_stdout_to_str(
        self, fn_calling_print: types.FunctionType, *args, **kwargs
    ) -> str:
        with io.StringIO() as redirected_stdout, contextlib.redirect_stdout(
            redirected_stdout
        ):
            fn_calling_print(*args, **kwargs)
            return redirected_stdout.getvalue()

    def test_analyze_time(self):
        gs_analize_time_str = self.__redirect_stdout_to_str(
            generate_services.analyze_time
        )
        sd_analize_time_str = (
            subprocess.run(["systemd-analyze", "time"], capture_output=True)
            .stdout.decode()
            .split("\n")[0]
            .strip()
        )
        gs_analyze_time_sd_format = "Startup finished in "
        if "(firmware)" in sd_analize_time_str:
            gs_analyze_time_sd_format += (
                gs_analize_time_str.split("firmware_time = ")[1].split("\n")[0]
                + " (firmware) + "
            )
        if "(loader)" in sd_analize_time_str:
            gs_analyze_time_sd_format += (
                gs_analize_time_str.split("loader_time = ")[1].strip().split("\n")[0]
                + " (loader) + "
            )
        if "(kernel)" in sd_analize_time_str:
            gs_analyze_time_sd_format += (
                gs_analize_time_str.split("kernel_done_time = ")[1]
                .strip()
                .split("\n")[0]
                + " (kernel) + "
            )
        if "(initrd)" in sd_analize_time_str:
            gs_analyze_time_sd_format += (
                gs_analize_time_str.split("initrd_time = ")[1].strip().split("\n")[0]
                + " (initrd) + "
            )
        if "(userspace)" in sd_analize_time_str:
            gs_analyze_time_sd_format += (
                gs_analize_time_str.split("userspace_time = ")[1].strip().split("\n")[0]
                + " (userspace) = "
            )
        gs_analyze_time_sd_format += (
            gs_analize_time_str.split("startup_finish_time = ")[1]
            .strip()
            .split("\n")[0]
        )
        self.assertEqual(sd_analize_time_str, gs_analyze_time_sd_format)

    def __get_services_num(self, service_template: str) -> int:
        return len(glob.glob1(generate_services.SYSTEMD_SYSTEM_PATH, service_template))

    def test_gen_parallel_services(self):
        service_generator = generate_services.ParallelServices(gen_dot=False)
        self.__redirect_stdout_to_str(
            service_generator.gen_test_services,
            generate_services.SYSTEMD_SYSTEM_PATH,
            15,
        )
        self.__redirect_stdout_to_str(
            generate_services.enable_test_services,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.assertEqual(self.__get_services_num("test_parallel*.service"), 15)
        parallel_service_text_ref = """[Unit]
Description=parallel test service 5


[Service]
Type=simple
RemainAfterExit=yes
ExecStart=/usr/bin/sleep 1


[Install]
WantedBy=multi-user.target"""
        generated_parallel_service_text = ""
        with open(
            os.path.join(
                generate_services.SYSTEMD_SYSTEM_PATH, "test_parallel5.service"
            )
        ) as parallel_service:
            generated_parallel_service_text = parallel_service.read()
        self.assertEqual(generated_parallel_service_text, parallel_service_text_ref)
        self.__redirect_stdout_to_str(
            generate_services.disable_test_services,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.__redirect_stdout_to_str(
            generate_services.remove_test_services,
            generate_services.SYSTEMD_SYSTEM_PATH,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.assertEqual(self.__get_services_num("test_parallel*.service"), 0)

    def test_gen_single_path_services(self):
        service_generator = generate_services.SinglePathServices(gen_dot=False)
        self.__redirect_stdout_to_str(
            service_generator.gen_test_services,
            generate_services.SYSTEMD_SYSTEM_PATH,
            20,
        )
        self.__redirect_stdout_to_str(
            generate_services.enable_test_services,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.assertEqual(self.__get_services_num("test_single_path*.service"), 20)
        single_path_service_text_ref = """[Unit]
Description=single path test service 2
After=test_single_path1.service


[Service]
Type=simple
RemainAfterExit=yes
ExecStart=/usr/bin/sleep 1


[Install]
WantedBy=multi-user.target"""
        generated_single_path_service_text = ""
        with open(
            os.path.join(
                generate_services.SYSTEMD_SYSTEM_PATH, "test_single_path2.service"
            )
        ) as single_path_service:
            generated_single_path_service_text = single_path_service.read()
        self.assertEqual(
            generated_single_path_service_text, single_path_service_text_ref
        )
        self.__redirect_stdout_to_str(
            generate_services.disable_test_services,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.__redirect_stdout_to_str(
            generate_services.remove_test_services,
            generate_services.SYSTEMD_SYSTEM_PATH,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.assertEqual(self.__get_services_num("test_single_path*.service"), 0)

    def test_gen_dag_services(self):
        service_generator = generate_services.DAGServices(
            edge_probability_arg=0.5, gen_dot=False
        )
        self.__redirect_stdout_to_str(
            service_generator.gen_test_services,
            generate_services.SYSTEMD_SYSTEM_PATH,
            10,
        )
        self.__redirect_stdout_to_str(
            generate_services.enable_test_services,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.assertEqual(self.__get_services_num("test_DAG*.service"), 10)
        dag_service_text_ref = """[Unit]
Description=DAG test service 0
Before=test_DAG3.service test_DAG4.service test_DAG8.service


[Service]
Type=simple
RemainAfterExit=yes
ExecStart=/usr/bin/sleep 1


[Install]
WantedBy=multi-user.target"""
        generated_dag_service_text = ""
        with open(
            os.path.join(generate_services.SYSTEMD_SYSTEM_PATH, "test_DAG0.service")
        ) as dag_service:
            generated_dag_service_text = dag_service.read()
        self.assertEqual(generated_dag_service_text, dag_service_text_ref)
        self.__redirect_stdout_to_str(
            generate_services.disable_test_services,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.__redirect_stdout_to_str(
            generate_services.remove_test_services,
            generate_services.SYSTEMD_SYSTEM_PATH,
            service_generator.test_service_prefix,
            str(service_generator),
        )
        self.assertEqual(self.__get_services_num("test_DAG*.service"), 0)
