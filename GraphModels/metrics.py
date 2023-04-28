from scipy.stats import hmean

def TP(x, y, z, w):
    return x
def TN(x, y, z, w):
    return y
def FP(x, y, z, w):
    return z
def FN(x, y, z, w):
    return w
def FDR(x, y, z, w):
    return z / (x + z) if x + z > 0 else 0
def TPR(x, y, z, w):
    return x / (x + w) if x + w > 0 else 1
def F1(x, y, z, w):
    precision = 1 - FDR(x, y, z, w)
    recall = TPR(x, y, z, w)
    return hmean([precision, recall])

def TNR(x, y, z, w):
    return y / (y + z) if y + z > 0 else 1

def FOR(x, y, z, w):
    return w / (y + w) if y + w > 0 else 0