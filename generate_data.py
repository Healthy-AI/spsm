import sys
sys.path.append('../')

import pickle as pkl
import argparse

from data.synthetic import *

if __name__ == "__main__":  #entrypoint
    """
    Argument parsing
    """
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    parser.add_argument('-se', '--setting', type=str, default='A', dest='setting', help='setting help')
    parser.add_argument('-s', '--seed', type=int, default=0, dest='seed', help='seed help')
    parser.add_argument('-o', '--outfile', type=str, default='./SYNTH_%s.pkl', dest='outfile', help='setting help')

    args = parser.parse_args()

    np.random.seed(args.seed)

    print('Generating data according to setting: %s' % args.setting)

    if args.setting.upper() == 'A':
        X, Xm, y, M, A, cfg = synth_setting_A()
    elif args.setting.upper() == 'B':
        X, Xm, y, M, A, cfg = synth_setting_B()
    elif args.setting.upper() == 'A1':
        X, Xm, y, M, A, cfg = synth_setting_A(n=10000)
    elif args.setting.upper() == 'B1':
        X, Xm, y, M, A, cfg = synth_setting_B(n=10000)
    else:
        raise Exception('Unknown setting: %s' % args.setting)

    fname = args.outfile % args.setting.upper()
    pkl.dump({'X': Xm, 'y': y, 'cfg': cfg, 'seed': args.seed}, open(fname, 'wb'))

    print('Saved to %s' % fname)
