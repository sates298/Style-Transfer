from loader import get_images
from transfer import Transferer
from plotting import plotting
import sys

content_weight = 2_000_000

style_layers = [0, 1, 2]
content_layers = [2]


def main():
    folder = sys.argv[1]
    cont, sty, inp = get_images(folder)
    transf = Transferer(cont, sty, inp)
    transf.fit(content_weight=content_weight,
               cont_layers=content_layers,
               style_layers=style_layers)
    out = transf.get_output()
    plotting(out['target'], out['style'], out['content'], dirname=folder)


if __name__ == '__main__':
    main()