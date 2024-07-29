import os
import glob as gb
import sys
from pathlib import Path

def run(rgb_path, save_path, batch_size=1, gap=[1, 2], reverse=[0, 1]):
    # data_path = '/path/to/dataset/'
    # rgb_path = data_path + '/JPEGImages'
    # # '/JPEGImages/480p' for DAVIS-related datasets and '/JPEGImages' for others

    # gap = [1]
    # reverse = [0, 1]
    # batch_size = 4

    save_path = Path(save_path)
    rgb_path = Path(rgb_path)
    folder = rgb_path.glob('*.png')
    if next(iter(folder)).is_file():
        folder = [rgb_path]
    for r in reverse:
        for g in gap:
            for f in folder:
                print('===> Runing {}, gap {}'.format(f, g))
                mode = 'raft-things.pth'  # model
                if r==1:
                    raw_outroot = save_path / 'Flows_gap-{}/'.format(g)  # where to raw flow
                    outroot = save_path / 'FlowImages_gap-{}/'.format(g)  # where to save the image flow
                elif r==0:
                    raw_outroot = save_path / 'Flows_gap{}/'.format(g)   # where to raw flow
                    outroot = save_path / 'FlowImages_gap{}/'.format(g)   # where to save the image flow

                os.system("python predict.py "
                            "--gap {} --mode {} --path {} --batch_size {} "
                            "--outroot {} --reverse {} --raw_outroot {}".format(g, mode, f, batch_size, outroot, r, raw_outroot))


if __name__ == '__main__':
    args = sys.argv
    run(args[1], args[2], int(args[3]))
