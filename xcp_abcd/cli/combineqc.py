# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 1; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""aggregate qc of all the subjects"""
import os
import glob as glob 
import pandas as pd
from pathlib import Path 

def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    parser.add_argument('xcpabcd_dir', action='store', type=Path, help='xcp_abcd output dir')
    
    parser.add_argument('output_prefix', action='store', type=str, help='output prefix for group')

    parser.add_argument('--cifti', action='store_true', default=False,help=' add cifti qc files')


    return parser


def main():
    """Entry point"""
    
    opts = get_parser().parse_args()

    allsubj_dir = os.path.abspath(opts.aslprep_dir)
    outputfile  = os.getcwd() + '/' + str(opts.output_prefix) + '_allsubjects_qc.csv'
    
    qclist=[]
    if opts.cifti:
	    for r, d, f in os.walk(allsubj_dir):
		    for filex in f:
			    if filex.endswith("space-fsLR_desc-qc_den-91k_bold.csv"):
				    qclist.append(r+ '/'+ filex)
    else:
	    for r, d, f in os.walk(allsubj_dir):
		    for filex in f:
			    if filex.endswith("desc-qc_bold.csv"):
				    qclist.append(r+ '/'+ filex)

    datax = pd.read_csv(qclist[0])
    for i in range(1,len(qclist)):    
        dy = pd.read_csv(qclist[i])
        datax = pd.concat([datax,dy])
    
    datax.to_csv(outputfile,index=None)

if __name__ == '__main__':
    raise RuntimeError("this should be run after xcp_abcd;\n"
                       " run xcp-abcd first")