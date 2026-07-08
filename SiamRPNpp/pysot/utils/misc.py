# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from colorama import Fore, Style


__all__ = ['commit', 'describe']


def _exec(cmd):
    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return ""


def _bold(s):
    return "\033[1m%s\033[0m" % s


def _color(s):
    return f'{Fore.RED}{s}{Style.RESET_ALL}'


def _describe(model, lines=None, spaces=0):
    head = " " * spaces
    for name, p in model.named_parameters():
        if '.' in name:
            continue
        if p.requires_grad:
            name = _color(name)
        line = "{head}- {name}".format(head=head, name=name)
        lines.append(line)

    for name, m in model.named_children():
        space_num = len(name) + spaces + 1
        if m.training:
            name = _color(name)
        line = "{head}.{name} ({type})".format(
                head=head,
                name=name,
                type=m.__class__.__name__)
        lines.append(line)
        _describe(m, lines, space_num)


def commit():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    try:
        import subprocess
        # 获取commit hash
        result = subprocess.run('git log -1 --format="%H"', shell=True, capture_output=True, text=True, cwd=root)
        commit = result.stdout.strip().strip('"')
        # 获取commit log
        result = subprocess.run('git log -1 --oneline', shell=True, capture_output=True, text=True, cwd=root)
        commit_log = result.stdout.strip()
        return "commit : {}\n  log  : {}".format(commit, commit_log)
    except:
        return "commit : N/A\n  log  : N/A"


def describe(net, name=None):
    num = 0
    lines = []
    if name is not None:
        lines.append(name)
        num = len(name)
    _describe(net, lines, num)
    return "\n".join(lines)
