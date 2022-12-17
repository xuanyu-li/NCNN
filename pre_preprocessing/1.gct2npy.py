#!/usr/bin/env python

import sys

import numpy as np
import cmapPy.pandasGEXpress.parse_gctx

def main():
    information_data = cmapPy.pandasGEXpress.parse_gctx.parse('./bgedv2_QNORM.gctx',
                                                              convert_neg_666=True, ridx=None, cidx=None,
                                                              row_meta_only=False,
                                                              col_meta_only=False, make_multiindex=False)

    outfile = 'bgedv2_float64.npy'

    data = information_data.data_df.values

    
    np.save(outfile, data)
    
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
