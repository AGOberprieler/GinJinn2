''' Test main
'''

import pytest
import subprocess

def test_main_simple():
    p = subprocess.Popen('ginjinn')
    r = p.communicate()
    rc = p.returncode

    assert rc == 0
