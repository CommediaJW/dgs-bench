import torch
import sys
import numpy

if __name__ == '__main__':
    tensor = torch.load(sys.argv[1]).numpy()

    print("Analysis result for", sys.argv[1])
    print("Average accessed times:", numpy.mean(tensor))
    print("Max accessed times:", numpy.max(tensor))
    print("Min accessed times:", numpy.min(tensor))
    print("#Nodes total:", tensor.size)
    print("#Nodes accessed times >= 1:", tensor[tensor >= 1].size)
    step = 24
    threshold = numpy.min(tensor) + step
    while threshold < numpy.max(tensor):
        print("#Nodes accessed times > {}:".format(threshold),
              tensor[tensor > threshold].size)
        threshold += step