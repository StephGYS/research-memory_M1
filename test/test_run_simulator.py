# -*- coding: utf-8 -*-
"""
    This file is part of FLOWSIM.

    FLOWSIM is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    FLOWSIM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with FLOWSIM.  If not, see <https://www.gnu.org/licenses/>.

    Copyright (c) 2020 Bart Lamiroy
    e-mail: Bart.Lamiroy@univ-lorraine.fr
"""

import unittest
import filecmp
import os
import subprocess
import tempfile
from flowsim.labs.defaults import export_json, import_json, get_default_params
from flowsim.models.sir_h.simulator import run_sir_h


class TestRunSimulator(unittest.TestCase):

    def test_simulator_execution(self):
        param_pathname = './labs/test_data/default_parameters.json'
        series = 'SI'
        prefix = 'flowsim_run'
        reference_file = './labs/test_data/flowsim_run_SI_baseline.csv'

        self.assertTrue(os.path.exists(param_pathname),
                        f'{param_pathname} does not exist')
        self.assertTrue(os.path.exists(reference_file),
                        f'{reference_file} does not exist')

        with tempfile.TemporaryDirectory() as tmp_dirname:

            args = ['-m', 'flowsim.labs.run_simulator', '--noplot',
                    '--path', tmp_dirname, '-p', param_pathname, '-s']
            subprocess.call(['python3', *args])

            param_basename = os.path.splitext(
                os.path.basename(param_pathname))[0]
            result_file = f'{tmp_dirname}/{"_".join([prefix, series, param_basename])}.csv'

            self.assertTrue(os.path.exists(result_file),
                            f'{result_file} does not exist')

            with open(result_file) as f:
                l1 = [line for line in f]
            with open(reference_file) as f:
                l2 = [line for line in f]
            for a, b in zip(l1, l2):
                self.assertEqual(a, b)
            # self.assertTrue(filecmp.cmp(
                # result_file, reference_file), 'Files are not equal')


if __name__ == '__main__':
    unittest.main()
