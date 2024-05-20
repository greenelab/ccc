from ccc.coef import ccc
import numpy as np

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config


def main():
    random_feature1 = np.random.rand(1000)
    random_feature2 = np.random.rand(1000)

    config = Config(max_depth=10)
    with PyCallGraph(output=GraphvizOutput(), config=config):
        res = ccc(random_feature1, random_feature2)
        print(res)


if __name__ == "__main__":
    main()




