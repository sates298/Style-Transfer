#!/home/steve/Dokumenty/style-transfer/venv/bin/python

from loader import get_images
from transfer import Transferer
from plotting import plotting
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


content_weight = 100_000
if rank % 2 == 0:
    content_weight *= 10

style_layers = [0, 1, 2]
content_layers = [0, 1, 2]

if rank in (0, 1):
#     style_layers = [0, 1]
    content_layers = [2]
elif rank in (2, 3):
    style_layers = [0, 1]
    content_layers = [1, 2]
elif rank in (4, 5):
    style_layers = [0, 1]
elif rank in (6, 7):
    content_layers = [1, 2]


def main():
    folder = sys.argv[1]
    cont, sty, inp = get_images(folder)
    transf = Transferer(cont, sty, inp)
    transf.fit(content_weight=content_weight,
               cont_layers=content_layers,
               style_layers=style_layers)
    out = transf.get_output()
    plotting(out['target'], out['style'], out['content'], rank=rank)


if __name__ == '__main__':
    main()

while x <= polowa and liczba!=1:
    if x in liczbypierwsze or (x > max(liczbypierwsze) and czypierwsza(x)):

    if czypierwsza(x) == 1 and liczba%x == 0:
        liczbypierwsze.add(x)
        tablica.append(x)
        liczba = liczba/x
        print(x)
        x = 2
    x+=1