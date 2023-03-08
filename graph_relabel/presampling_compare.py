import torch
import sys
import numpy

if __name__ == '__main__':
    a = torch.load(sys.argv[1]).numpy()
    b = torch.load(sys.argv[2]).numpy()

    assert (a.size == b.size)
    print("Compared reulst for", a, "and", b)
    print(
        "Accessed times >= 1 nodes' intersect size:",
        numpy.intersect1d(
            numpy.arange(a.size)[a >= 1],
            numpy.arange(b.size)[b >= 1]).size)

    max = min(numpy.max(a), numpy.max(b))
    step = 24
    threshold = step
    while threshold < max:
        print(
            "Accessed times > {} nodes' intersect size:".format(threshold),
            numpy.intersect1d(
                numpy.arange(a.size)[a > threshold],
                numpy.arange(b.size)[b > threshold]).size)
        threshold += step
