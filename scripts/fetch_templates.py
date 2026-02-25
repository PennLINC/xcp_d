#!/usr/bin/env python
"""
Standalone script to facilitate caching of required TemplateFlow templates.
"""

import argparse
import os


def fetch_MNI2009():
    template = 'MNI152NLin2009cAsym'
    tf.get(template, resolution=1, desc=None, suffix='T1w')
    tf.get(template, resolution=1, desc='carpet', suffix='dseg')
    tf.get(template, resolution=2, desc='brain', suffix='mask')
    tf.get(template, mode='image', suffix='xfm', extension='.h5', **{'from': 'MNI152NLin6Asym'})


def fetch_MNI6():
    template = 'MNI152NLin6Asym'
    tf.get(template, resolution=1, desc=None, suffix='T1w')
    tf.get(template, mode='image', suffix='xfm', extension='.h5', **{'from': 'MNI152NLin2009cAsym'})


def fetch_MNIInfant():
    tf.get('MNIInfant', resolution=1, desc=None, suffix='T1w')


def fetch_fsaverage():
    tf.get('fsaverage', density='164k', desc=None, suffix='sphere')
    tf.get('fsaverage', density='41k', desc=None, suffix='sphere')


def fetch_fsLR():
    tf.get('fsLR', density='32k', desc=None, suffix='midthickness')
    tf.get('fsLR', density='32k', desc='vaavg', suffix='midthickness')
    tf.get('fsLR', space=None, density='32k', suffix='sphere')


def fetch_dhcpAsym():
    tf.get('dhcpAsym', cohort='42', space='fsaverage', density='41k', desc='reg', suffix='sphere')
    tf.get('dhcpAsym', cohort='42', space=None, density='32k', desc=None, suffix='sphere')


def fetch_all():
    fetch_MNI2009()
    fetch_MNI6()
    fetch_MNIInfant()
    fetch_fsaverage()
    fetch_fsLR()
    fetch_dhcpAsym()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Helper script for pre-caching required templates to run XCP-D',
    )
    parser.add_argument(
        '--tf-dir',
        type=os.path.abspath,
        help=(
            'Directory to save templates in. '
            'If not provided, templates will be saved to `${HOME}/.cache/templateflow`.'
        ),
    )
    opts = parser.parse_args()

    if opts.tf_dir is not None:
        os.environ['TEMPLATEFLOW_HOME'] = opts.tf_dir

    import templateflow.api as tf

    fetch_all()
