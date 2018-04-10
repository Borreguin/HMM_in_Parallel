import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from hmmlearn.hmm import GaussianHMM
import numpy as np

#samples:
X = np.array([[-1.03573482, -1.03573482],
       [ 6.62721065, 11.62721065],
       [ 3.19196949,  8.19196949],
       [ 0.38798214,  0.38798214],
       [ 2.56845104,  7.56845104],
       [ 5.03699793, 10.03699793],
       [ 5.87873937, 10.87873937],
       [ 4.27000819, -1.72999181],
       [ 4.02692237, -1.97307763],
       [ 5.7222677 , 10.7222677 ]])


# Trainning a new model over samples:
model = GaussianHMM(n_components=3, covariance_type="diag").fit(X)

# Create a new copy of the trained model:
new_model = GaussianHMM(n_components=3, covariance_type="diag")
new_model.startprob_ = model.startprob_
new_model.transmat_ = model.transmat_
new_model.means_ = model.means_
m = model._covars_
n = model.covars_
p = model.get_params()
new_model.covars_ = model._covars_

# Predict from X:
X_N = new_model.predict(X)

print(X_N)