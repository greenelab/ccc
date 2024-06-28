from ccc.coef import ccc
import numpy as np


def main():
    random_feature1 = np.random.rand(100)
    random_feature2 = np.random.rand(100)

    res = ccc(random_feature1, random_feature2, n_jobs=2)
    print(res)


if __name__ == "__main__":
    main()




