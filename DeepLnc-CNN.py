#!/usr/bin/env python
# -*- coding:utf-8-*-
# author: TianyangZhang time:2023/3/26 QQ:980557945 e-mail:Tianyang.Zhang819@outlook.com
# ----------------------------------------------------------------------------
'''
This is the main program of DeepLnc-CNN. Using this program you can identify lncRNAs in human and mouse.
Required modules to be installed:
pytorch==1.8.2+cpu
numpy==1.21.5
 '''
# ----------------------------------------------------------------------------
import argparse
from bin.prediction_process import preprocess, lncRNAdeep

def parse_args():
    '''Parameters.'''
    description = "DeepLnc-CNN is able to identify long non-coding RNAs in Human and Mouse.\n" \
                  "Example: python DeepLnc-CNN.py -i Example.txt -o output.html -s Human -ts 0.5"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--inputFile',
                        help='-i Example.txt (The input file is a complete Fasta format sequence.)')
    parser.add_argument('-o', '--outputFile',
                        help='-o output.html (Results of predicting lncRNA are saved under results folder.)')
    parser.add_argument('-s', '--species',
                        help='-s h/m (Choose Human/Mouse from two species to use.)')
    parser.add_argument('-ts', '--threshold',
                        help='-ts 0.5(Prediction result threshold)')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    preprocess(args.inputFile, args.outputFile, args.species, float(args.threshold))
