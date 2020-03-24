import numpy as np

# let's import a dataset to test the regression
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# Forward Stagewise Regression

def get_residuals(model, x, y):
    return y - model.predict(x)

def find_max_correlation(res, x, verbose = False):
    im, corr = 0, 0
    for i in range(x.shape[1]):
        cf = np.corrcoef(x[:,i], res)[0,1]
        if np.abs(cf) > np.abs(corr): im, corr = i, cf
        if verbose: print ("[+] %d: %f" % (im, cf))
    return im, corr

def get_coeff(x,y):
    return np.dot(x,y) / np.dot(x,x)

def update_model(model, idx, value):
    model.coef_[idx]+=value

def stagewise_regression(x, y, tolerance = 1e-4, max_iterations = 1e3, verbose = 0):
    model = LinearRegression()
    model.coef_ = np.zeros(x.shape[1])
    model.intercept_ = np.mean(y, axis = 0)

    it, corr = 0, tolerance * 2
    while abs(corr) > tolerance:
        it+=1
        res = get_residuals(model, x, y)
        ix, corr = find_max_correlation(res, x)
        cf = get_coeff(x[:,ix], res)
        if cf == 0:
            print("[!!] Coefficient not being updated")
            break
        update_model(model, ix, cf)
        if verbose == 2:
            print("[+] Residuals: %f. Max corr: %f in cord %d, coeff: %f" % (np.dot(res, res), corr, ix, cf))
        if it > max_iterations:
            print("[!!] Max iterations")
            break
    if verbose == 1:
        print("[+] Residuals: %f. Max corr: %f in cord %d, coeff: %f" % (np.dot(res, res), corr, ix, cf))
    return model

def main():
    # import dataset
    data = load_diabetes()
    x = data['data']
    y = data['target']

    # Let's get the model using stagewise regression
    srm = stagewise_regression(x, y, tolerance = 1e-5, max_iterations = 1e4, verbose = 1)

    # let's compare the output with a regular regression
    lr = LinearRegression()
    lr.fit(x,y)
    res = lr.predict(x) - y
    print("[+] Residues norm: %f" % np.dot(res,res))

    # let's check the coefficients
    coeff_diff = (lr.coef_ - srm.coef_) / lr.coef_ * 100
    print("[+] Max difference in coefficient: %f" % np.max(coeff_diff))


if __name__ == "__main__":
    main()
