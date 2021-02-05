#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""xcp_abcd postprocessing workflow."""

import os
import re
from pathlib import Path
import logging
import sys
import gc
import uuid
import warnings
import json
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from nipype import logging as nlogging, config as ncfg
from multiprocessing import cpu_count
from time import strftime

logging.addLevelName(25, 'IMPORTANT')  # Add a new level between INFO and WARNING
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')


def _warn_redirect(message, category, filename, lineno, file=None, line=None):
    logger.warning('Captured warning (%s): %s', category, message)


def check_deps(workflow):
    from nipype.utils.filemanip import which
    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and
            which(node.interface._cmd.split()[0]) is None))


def get_parser():
    """Build parser object."""
    from packaging.version import Version
    from ..__about__ import __version__
    from .version import check_latest, is_flagged

    verstr = 'xcp_abcd v{}'.format(__version__)


    parser = ArgumentParser(description='xcp_abcd postprocessing workflow of fmriprep outputs',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # important parameters required
    parser.add_argument('fmriprep_dir', action='store', type=Path,
                        help='the root folder of a fmriprep output  with sub-xxxx.')
    parser.add_argument('output_dir', action='store', type=Path,
                        help='the output path for the outcomes of preprocessing')
    parser.add_argument('analysis_level', choices=['participant'],
                        help='processing stage to be run, only "participant')

    # optional arguments
    parser.add_argument('--version', action='version', version=verstr)
     
    g_bidx = parser.add_argument_group('Options for filtering BIDS queries')
    g_bidx.add_argument('--participant_label', '--participant-label', action='store', nargs='+',
                        help='a space delimited list of participant identifiers or a single '
                             'identifier (the sub- prefix can be removed)')
    g_bidx.add_argument(
        '--bids-filter-file', action='store', type=Path, metavar='PATH',
        help='BIDS input filter using pybids ')

    g_bidx.add_argument('-t', '--task-id', action='store',
                        help='select a specific task to be selected for postprocessing')

    g_surfx = parser.add_argument_group('Options for cifti processing')
    g_surfx.add_argument('-s', '--surface', action='store_true', default=False,
                        help='post process surface instead of nifti')

    
    g_perfm = parser.add_argument_group('Options to for resource management ')
    g_perfm.add_argument('--nthreads',  action='store', type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--omp-nthreads', action='store', type=int, default=0,
                         help='maximum number of threads per-process')
    g_perfm.add_argument('--mem_mb', '--mem-mb', action='store', default=0, type=int,
                         help='upper bound memory limit for xcp_abcd processes')
    g_perfm.add_argument('--low-mem', action='store_true',
                         help='attempt to reduce memory usage (will increase disk usage '
                              'in working directory)')
    g_perfm.add_argument('--use-plugin', action='store', default=None,
                         help='nipype plugin configuration file')
    g_perfm.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                         help="increases log verbosity for each occurence, debug level is -vvv")

    g_param = parser.add_argument_group('parameters for postprocessing')
    g_param.add_argument(
        '--template', action='store', default='MNI152NLin2009cAsym',
        help=" template to be selected from anat to be processed or for normalization")

    g_param.add_argument('--lowpass', action='store', default=0.1, type=float,
                        help='low pass filter')

    g_param.add_argument('--highpass', action='store', default=0.01, type=float,
                        help='high pass filter')
    
    g_param.add_argument('--smoothing', nargs='?', const=5, default=False,
                             type=float, help='smoothing the postprocessed output (fhwm)')
    
    g_param.add_argument('-r','--head_radius',default=50,
                             type=float, help='head radius for computing FD, it is 40mm for baby')
    g_param.add_argument('-p','--params', required=True, default='24p',
                             type=str, help='nuissance parameters to be selected')
    g_param.add_argument('-c','--custom_conf', required=False,
                             type=Path, help='custom confound to be added to nuissance regressors')

    g_other = parser.add_argument_group('Other options')
    g_other.add_argument('-w', '--work-dir', action='store', type=Path, default=Path('work'),
                         help='path where intermediate results should be stored')
    g_other.add_argument('--clean-workdir', action='store_true', default=False,
                         help='Clears working directory of contents. Use of this flag is not'
                              'recommended when running concurrent processes of fMRIPrep.')
    g_other.add_argument(
        '--resource-monitor', action='store_true', default=False,
        help='enable Nipype\'s resource monitoring to keep track of memory and CPU usage')

    g_other.add_argument('--sloppy', action='store_true', default=False,
                         help='Use low-quality tools for speed - TESTING ONLY')

    return parser


def main():
    """Entry point"""
    from nipype import logging as nlogging
    from multiprocessing import set_start_method, Process, Manager
    set_start_method('forkserver')
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()

    #exec_env = os.name

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    logger.addHandler(logging.StreamHandler())
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    # Call build_workflow(opts, retval)
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(opts, retval))
        p.start()
        p.join()

        retcode = p.exitcode or retval.get('return_code', 0)

        #fmriprep_dir = Path(retval.get('fmriprep_dir'))
        #output_dir = Path(retval.get('output_dir'))
        #work_dir = Path(retval.get('work_dir'))
        plugin_settings = retval.get('plugin_settings', None)
        #subject_list = retval.get('subject_list', None)
        xcpabcd_wf = retval.get('workflow', None)
       


    retcode = retcode or int(xcpabcd_wf is None)
    if retcode != 0:
        sys.exit(retcode)

    # Check workflow for missing commands
    missing = check_deps(xcpabcd_wf)
    if missing:
        print("Cannot run xcp_abcd. Missing dependencies:", file=sys.stderr)
        for iface, cmd in missing:
            print("\t{} (Interface: {})".format(cmd, iface))
        sys.exit(2)
    # Clean up master process before running workflow, which may create forks
    gc.collect()

 
    xcpabcd_wf.run(**plugin_settings)
    

 



def build_workflow(opts, retval):
    """
    Create the Nipype Workflow that supports the whole execution
    graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows fmriprep to enforce
    a hard-limited memory-scope.

    """
    from bids import BIDSLayout
    from ..utils  import collect_participants
    from ..__about__ import __version__
    from ..workflow.base import init_xcpabcd_wf 

    build_log = logging.getLogger('nipype.workflow')

    INIT_MSG = """
    Running xcp_abcd version {version}:
      * fMRIPrep directory path: {fmriprep_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {uuid}.

    """.format

    fmriprep_dir = opts.fmriprep_dir.resolve()
    output_dir = opts.output_dir.resolve()
    work_dir = opts.work_dir.resolve()
    bids_filters = json.loads(opts.bids_filter_file.read_text()) if opts.bids_filter_file else None
   

    retval['return_code'] = 1
    retval['workflow'] = None
    retval['bids_dir'] = str(fmriprep_dir)
    retval['output_dir'] = str(output_dir)
    retval['work_dir'] = str(work_dir)

    if output_dir == fmriprep_dir:
        build_log.error(
            'The selected output folder is the same as the fmriprep directory. '
            'Please modify the output path ')
        retval['return_code'] = 1
        return retval

    if fmriprep_dir in work_dir.parents:
        build_log.error(
            'The selected working directory is a subdirectory of fmriprep directory. '
            'Please modify the output path.')
        retval['return_code'] = 1
        return retval

    # Set up some instrumental utilities
    run_uuid = '%s_%s' % (strftime('%Y%m%d-%H%M%S'), uuid.uuid4())
    retval['run_uuid'] = run_uuid

    # First check that bids_dir looks like a BIDS folder
    layout = BIDSLayout(str(fmriprep_dir), validate=False, derivatives=True)
    subject_list = collect_participants(
        layout, participant_label=opts.participant_label)
    retval['subject_list'] = subject_list

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml
        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        plugin_settings.setdefault('plugin_args', {})
    else:
        # Defaults
        plugin_settings = {
            'plugin': 'MultiProc',
            'plugin_args': {
                'raise_insufficient': False,
                'maxtasksperchild': 1,
            }
        }


    nthreads = plugin_settings['plugin_args'].get('n_procs')
    # Permit overriding plugin config with specific CLI options
    if nthreads is None or opts.nthreads is not None:
        nthreads = opts.nthreads
        if nthreads is None or nthreads < 1:
            nthreads = cpu_count()
        plugin_settings['plugin_args']['n_procs'] = nthreads

    if opts.mem_mb:
        plugin_settings['plugin_args']['memory_gb'] = opts.mem_mb / 1024

    omp_nthreads = opts.omp_nthreads
    if omp_nthreads == 0:
        omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)

    if 1 < nthreads < omp_nthreads:
        build_log.warning(
            'Per-process threads (--omp-nthreads=%d) exceed total '
            'threads (--nthreads/--n_cpus=%d)', omp_nthreads, nthreads)
    retval['plugin_settings'] = plugin_settings

    # Set up directories
    log_dir = output_dir / 'xcpabcd' / 'logs'
    # Check and create output and working directories
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Nipype config (logs and execution)
    ncfg.update_config({
        'logging': {
            'log_directory': str(log_dir),
            'log_to_file': True
        },
        'execution': {
            'crashdump_dir': str(log_dir),
            'crashfile_format': 'txt',
            'get_linked_libs': False,
        },
        'monitoring': {
            'enabled': opts.resource_monitor,
            'sample_frequency': '0.5',
            'summary_append': True,
        }
    })

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    # Build main workflow
    build_log.log(25, INIT_MSG(
        version=__version__,
        fmriprep_dir=fmriprep_dir,
        subject_list=subject_list,
        uuid=run_uuid)
    )
   
    retval['workflow'] = init_xcpabcd_wf (
              layout=layout,
              omp_nthreads=omp_nthreads,
              fmriprep_dir=str(fmriprep_dir),
              subject_list=subject_list,
              work_dir=str(work_dir),
              bids_filters=bids_filters,
              lowpass=opts.lowpass,
              task_id=opts.task_id,
              highpass=opts.highpass,
              smoothing=opts.smoothing,
              params=opts.params,
              surface=opts.surface,
              output_dir=str(output_dir),
              head_radius=opts.head_radius,
              template=opts.template,
              custom_conf=opts.custom_conf,
              name='xcpabcd_wf'
              )
    
    retval['return_code'] = 0

    #logs_path = Path(output_dir) / 'xcpabcd' / 'logs'
    
    return retval


if __name__ == '__main__':
    raise RuntimeError("xcp_abcd/cli/run.py should not be run directly;\n"
                       "Please `pip install` xcp_abcd and use the `xcp_abcd` command")