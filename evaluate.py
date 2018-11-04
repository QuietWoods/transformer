# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 15:32
# @Author  : QuietWoods
# @FileName: evaluate.py
# @Software: PyCharm

""" evaluation scripts"""
import re
import os
from os.path import join
import logging
import tempfile
import subprocess as sp

from cytoolz import curry

from pyrouge import Rouge155
from pyrouge.utils import log


try:
    _ROUGE_PATH = os.environ['ROUGE']
except KeyError:
    print('Warning: ROUGE is not configured')
    _ROUGE_PATH = None
def eval_rouge(dec_dir, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), "",
            join(tmp_dir, 'ref'), "",
            join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None


def eval_meteor(dec_file, ref_file):
    """ METEOR evaluation"""
    assert _METEOR_PATH is not None

    cmd = 'java -Xmx2G -jar {} {} {} -l en -norm'.format(
        _METEOR_PATH, dec_file, ref_file)
    output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output
