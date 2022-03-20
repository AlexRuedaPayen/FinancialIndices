import os,sys
sys.path.append(os.getcwd())

from Alexandre.Class.GCP import GCP
import unittest,pandas

class TestGCP(unittest.TestCase):
    def fun_A(self):
        fun_A_with_GCP=pandas.read_csv("./Alexandre/Data_GCP/fun_A_res.csv")
        from Alexandre.Script.fun_A import fun_A
        fun_A_without_GCP=fun_A()
        self.assertEqual(fun_A_with_GCP,fun_A_without_GCP)

if __name__ == '__main__':

    with GCP() as f:
        f.run("fun_A")

    unittest.main()