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


class TestOptimise(unittest.TestCase):

    def test_default_execution(self):
        param_pathname = './labs/test_data/default_parameters.json'
        series = 'SI'
        model = 'disc'
        optim = 'least-squares'
        prefix = 'flowsim_fit'

        self.assertTrue(os.path.exists(param_pathname),
                        f'{param_pathname} does not exist')

        with tempfile.TemporaryDirectory() as tmp_dirname:
            args = ['-m', 'flowsim.labs.model_fit.optimise', '--noplot',
                    '--path', tmp_dirname, '-s']
            subprocess.call(['python3', *args])

            default_result_path_prefix = f'{tmp_dirname}/{"_".join([prefix, model, optim])}'
            default_json_file = default_result_path_prefix + '.json'
            default_opt_file = default_result_path_prefix + '_opt.json'
            default_res_file = default_result_path_prefix + '.res'
            default_file_list = [default_json_file, default_opt_file, default_res_file]

            args = ['-m', 'flowsim.labs.model_fit.optimise', '--noplot',
                    '--path', tmp_dirname, '-p', param_pathname, '-s', prefix + '2']
            subprocess.call(['python3', *args])

            param_result_path_prefix = f'{tmp_dirname}/{prefix}2'
            param_json_file = param_result_path_prefix + '.json'
            param_opt_file = param_result_path_prefix + '_opt.json'
            param_res_file = param_result_path_prefix + '.res'
            param_file_list = [param_json_file, param_opt_file, param_res_file]

            for (p_file, d_file) in zip(param_file_list, default_file_list):
                self.assertTrue(os.path.exists(d_file),
                                f'Process did not create {os.path.basename(d_file)}')
                self.assertTrue(os.path.exists(p_file),
                                f'Process did not create {os.path.basename(p_file)}')
                self.assertTrue(filecmp.cmp(d_file, p_file), f'Files {d_file} and {p_file} are not equal')
                '''
                with open(p_file) as f:
                    l1 = [line for line in f]
                with open(d_file) as f:
                    l2 = [line for line in f]
                for a, b in zip(l1, l2):
                    self.assertEqual(a, b, f'Files {d_file} and {p_file} are not equal')
                '''


if __name__ == '__main__':
    unittest.main()
