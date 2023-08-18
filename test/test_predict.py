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


class TestPredict(unittest.TestCase):

    def test_default_execution(self):
        param_pathname = './labs/test_data/default_parameters.json'
        series = 'SI'
        prefix = 'flowsim_predict'

        self.assertTrue(os.path.exists(param_pathname),
                        f'{param_pathname} does not exist')

        with tempfile.TemporaryDirectory() as tmp_dirname:
            self.assertTrue(os.path.exists(tmp_dirname),
                            f'Process did not create {tmp_dirname}')

            args = ['-m', 'flowsim.labs.gaussian_processes.predict', '--silentplot',
                    '--path', tmp_dirname, '-s']
            r_object = subprocess.run(['python3', *args])
            self.assertEqual(r_object.returncode, 0, f'labs.gaussian_processes.predict exited with code {r_object.returncode} and {r_object.stderr}')

            default_result_path_prefix = f'{tmp_dirname}/{"_".join([prefix])}'
            default_csv_file = default_result_path_prefix + '.csv'
            default_png_file = default_result_path_prefix + '.png'
            default_file_list = [default_csv_file, default_png_file]

            for d_file in default_file_list:
                self.assertTrue(os.path.exists(d_file),
                                f'Process did not create {os.path.basename(d_file)}')


if __name__ == '__main__':
    unittest.main()
