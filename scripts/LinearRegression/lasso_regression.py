import numpy as np

# let's import a dataset to test the regression
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso

def get_residuals(model, x, y):
    return y - model.predict(x)

def main():
    # import dataset
    data = load_diabetes()
    x = data['data']
    y = data['target']

    # Let's get the model using lasso regression
    # Lasso regression is like ridge but with L1 regularization instead of L2
    rrm01 = Lasso(alpha = 0.01) # increasing alpha, we increase the constrain in the coefficients
    rrm01.fit(x,y)

    # Let's compare with a higher alpha
    rrm1 = Lasso(alpha=0.1)
    rrm1.fit(x,y)

    # Let's print the residuals
    res = get_residuals(rrm01, x, y)
    print("[+] Residuals norm for alpha = 0.01: %f" % np.dot(res, res))

    res = get_residuals(rrm1, x, y)
    print("[+] Residuals norm for alpha = 0.1: %f" % np.dot(res, res))

    # let's compare the output with a regular regression
    lr = LinearRegression()
    lr.fit(x,y)
    res = get_residuals(lr, x, y)
    print("[+] Residues norm: %f" % np.dot(res,res))

    # let's check the coefficients
    coeff_diff = (lr.coef_ - rrm01.coef_) / lr.coef_ * 100
    print("[+] Max difference in coefficient (alpha 0.01): %f" % np.max(coeff_diff))


    coeff_diff = (lr.coef_ - rrm1.coef_) / lr.coef_ * 100
    print("[+] Max difference in coefficient (alpha 0.1): %f" % np.max(coeff_diff))


if __name__ == "__main__":
    main()
