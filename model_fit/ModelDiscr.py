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

    SIR+H model based on discrete transitional states
"""

from typing import Dict, List
import numpy as np
from flowsim.models.sir_h.simulator import run_sir_h
from flowsim.models.rule import RuleChangeField
from flowsim.labs.defaults import get_default_params


def model_disc(model_params: Dict[str, any], series: List[str] = None, **kwargs: Dict[str, any]) -> Dict[str, any]:
    parameters = dict(model_params['parameters'])
    other_arguments = dict(kwargs)
    parameters.update(other_arguments)

    if 't_confinement' not in parameters.keys():
        t_confinement = get_default_params()['other']['confinement']
    else:
        t_confinement = parameters['t_confinement']

    if 't_end' not in parameters.keys():
        t_end = get_default_params()['other']['deconfinement']
    else:
        t_end = parameters['t_end']

    if 'beta_post' not in parameters.keys():
        parameters['beta_post'] = get_default_params(
        )['other']['r0_confinement'] / parameters['dm_r']
    if 'beta_end' not in parameters.keys():
        parameters['beta_end'] = 1.2 / parameters['dm_r']

    r0_start = parameters['beta'] * parameters['dm_r']
    r0_confinement = parameters['beta_post'] * parameters['dm_r']
    r0_end = parameters['beta_end'] * parameters['dm_r']

    '''
    beta_0 = parameters['beta'] if not 'R0_start' in parameters.keys() else parameters['R0_start']/parameters['dm_r']
    beta_post = r0_confinement/parameters['dm_r']
    beta_end = r0_end/parameters['dm_r']

    parameters['beta'] = beta_0
    if not 'beta_post' in parameters.keys() :
        parameters['beta_post'] = r0_confinement/parameters['dm_r']
    if not 'beta_end' in parameters.keys() :
        parameters['beta_end'] = r0_end/parameters['dm_r']
    '''

    rules = [RuleChangeField(t_confinement, 'beta', parameters['beta_post']),
             RuleChangeField(t_end, 'beta', parameters['beta_end']), ]

    lists = run_sir_h(parameters, rules, specific_series=series)
    t = np.linspace(0, parameters['lim_time'] - 1, parameters['lim_time'])

    def r0(t, k=1.0, R0_start=r0_start, t_confinement=t_confinement, R0_confinement=r0_confinement,
           t_end=parameters['lim_time'], R0_end=r0_end):
        # return 3.31
        # return R0_start if t < t_confinement else r0_confinement
        # if t<(t_confinement + t_end)/2:
        return (R0_start - R0_confinement) / (1 + np.exp(-k * (-t + t_confinement))) + R0_confinement
        # else:
        # return (r0_confinement-r0_end) / (1 + np.exp(-k*(-t + t_end))) + r0_end

    #  R0_over_time = [r0(i) for i in range(len(t))]

    return {'time': t, 'series': {k: np.array(v) for k, v in lists.items()}, }
