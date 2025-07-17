library(Tvar)
library(MASS)
library(reticulate)

use_condaenv('python3.12', required=TRUE)

spam = 
"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
spam = pd.read_csv('./TVS_real_data/spambase.csv')
X = spam.iloc[:, :-2].to_numpy()
Y = spam.ham.to_numpy() * 1
seed = 4
np.random.seed(seed)
X_shuffle, Y_shuffle = shuffle(X, Y, random_state=seed)
x_train, x_test, y_train, y_test = train_test_split(X_shuffle, Y_shuffle, test_size=0.01, random_state=seed) # 0.3 for metrics, 0.01 for metrics_cv
scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
"
dat = py_run_string(python_code)


crime = 
"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

dat = pd.read_csv('./TVS_real_data/crimedata.csv')

Y = dat.iloc[:, -1].to_numpy()
X = dat.iloc[:, :-1].to_numpy()

seed = 4
np.random.seed(seed)
X_shuffle, Y_shuffle = shuffle(X, Y, random_state=seed)
x_train, x_test, y_train, y_test = train_test_split(X_shuffle, Y_shuffle, test_size=0.01, random_state=seed) # 0.3 for metrics, 0.01 for metrics_cv

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train[:, np.newaxis]).ravel()
"
dat = py_run_string(crime)



madelon = 
  "
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

dat = pd.read_csv('./TVS_real_data/madelon.csv')

Y = dat.iloc[:, -1].to_numpy()
X = dat.iloc[:, :-1].to_numpy()

seed = 4
np.random.seed(seed)
X_shuffle, Y_shuffle = shuffle(X, Y, random_state=seed)
x_train, x_test, y_train, y_test = train_test_split(X_shuffle, Y_shuffle, test_size=0.01, random_state=seed) # 0.3 for metrics, 0.01 for metrics_cv
"
dat = py_run_string(madelon)



x_train = dat$x_train
y_train = dat$y_train

X = dat$X
Y = dat$Y



rm(madelon, dat)

start_time = Sys.time()
TVS_output <- TVS(y_train, x_train, selector= 'bart', maxnrep = 10000, stop_crit = 100, niter = 500, ntree = 10)
end_time = Sys.time()
print(end_time - start_time)

prob = posterior(TVS_output)

rm(TVS_output)  # 删除临时对象

gc()  # 强制垃圾回收


source_python('python_code.py')

orders = order(prob, decreasing=TRUE)


tvs_madelon_scale = matrix(nrow=5, ncol=50)

plot_data = vector(length = 50)
for (num in 1:50) {
  idx = as.integer(orders[1:num] - 1)
  # print(idx)
  plot_data[num] = mean(metrics_cv(as.integer(idx), X, Y))
}  


plot(1:50, plot_data)


tvs_madelon_scale[1, ] = plot_data

rm(X, Y, x_train, y_train)

# save_code = 
# "
# np.save('home/Desktop/TVS/tvs_spam.npy', np.array(tvs_spam))
# "
# py_run_string(save_code)

write.table(tvs_madelon_scale, file = "tvs_madelon_scale.txt", sep = " ", row.names = FALSE, col.names = FALSE, quote = TRUE)
