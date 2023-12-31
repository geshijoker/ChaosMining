import sys
import os
import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='Parse Argument for Simulation Experiment')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to the file saving the formulas')
    parser.add_argument('--experiment', '-e', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--name', '-n', type=str, required=True, 
                        help='Name of run', )
    parser.add_argument('--seed', '-s', type=int, default=None, 
                        help='which seed for random number generator to use')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='which GPU to use, negative value denotes cpu will be used')
    parser.add_argument('--loss', type=str, default='MSELoss', choices=['MSELoss', 'L1Loss'], 
                        help='Loss function to perform regression')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of test to train split')
    parser.add_argument('--num_noises', type=int, default=0,
                        help='Number of noisy features')
    parser.add_argument('--num_data', type=int, default=10000,
                        help='Number of data generated')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of steps to calculate integrated gradients')
    parser.add_argument('--X_var', type=float, default=0.33,
                        help='Variance of X distribution')
    parser.add_argument('--ny_var', type=float, default=0.0,
                        help='Variance of noise added to y distribution')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='the number of epochs for training')
    parser.add_argument('--arc_depth', type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('--arc_width', type=int, default=100,
                        help='Number of neurons in each layer')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], 
                        help='Type of optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (float in the range [0, 1])')
    parser.add_argument('--deterministic', action='store_true',
                        help='Using deterministic mode and disable benchmark algorithms')
    parser.add_argument('--debug', action='store_true',
                        help='Using debug mode')

    args = parser.parse_args()
    
    return args