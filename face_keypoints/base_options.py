### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import os
import sys

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        self.parser.add_argument('--load_epoch', type=str, help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--load_size', type=int, default = 300, help='how big the face img is to load')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--batch_size', type=int, default = 128, help='train batch size')
        self.parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
        self.parser.add_argument('--symbols', type=str, 
                                 default='BigResNet', 
                                 help='symbol list to train. segment with ,')

        # examples
        # self.parser.add_argument('--name', type=str, default='label2dancer_512p', help='name of the experiment. It decides where to store samples and models')        
        # self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')     
        # self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')    
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized: self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.symbols = self.opt.symbols.split(',')
        if self.opt.load_epoch and self.opt.load_epoch != 'latest': self.opt.load_epoch = int(self.opt.load_epoch)

        return self.opt
