#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" xcp_d setup script """
from setuptools import setup

import versioneer


if __name__ == "__main__":
    setup(
        name="xcp_d",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )
