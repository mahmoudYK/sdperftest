import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_sd_load_tests

class TestRunSDLoadTestsFns(unittest.TestCase):
    def test_assemble_service_gen_cmd(self):
        cmd_ref = ["/usr/bin/python", "generate_services.py", "-t", "dag", "-n", "30", "-p", "0.5"]
        assembled_cmd = run_sd_load_tests.assemble_service_gen_cmd(["dag"],remove=False,services_num=30,dag_edge_probability=0.5)
        self.assertEqual(assembled_cmd, cmd_ref)

    def test_calc_rmse(self):
        test_dict = {"actual":[1,2,3,4,5], "predicted":[1.6,2.5,2.9,3,4.1]}
        rmse_ref = 0.6971
        rmse_calculated = round(run_sd_load_tests.ErrorStatsCalculator.calc_rmse(test_dict), 4)
        self.assertEqual(rmse_calculated, rmse_ref)