library(Tvar)
library(MASS)


# ===================== linear model =====================================================
generate_data = function(m, n, signal, sigma, num_support)
{
  beta_star = rep(0, n)
  beta_star[1:num_support] = signal
  
  X = matrix(rnorm(m*n), m, n)
  Y = X %*% beta_star + rnorm(m, 0, sigma)
  
  return (list(X=X, Y=Y))
}


# ===================== covariates with correlations ====================================
generate_data = function(m, n, signal, sigma, num_support)
{
  mean = runif(n, -5, 5)
  cov = diag(1, n)
  for (i in 1:n) {
    for (j in 1:n) {
     cov[i, j] = 0.9^(abs(i-j))
    }
  }
  X = mvrnorm(m, mean, cov)
  
  beta_star = rep(0, n)
  beta_star[1:num_support] = signal
  
  Y = X %*% beta_star + rnorm(m, 0, sigma)
  
  return (list(X=X, Y=Y))
}

# ===================== linear model with intersection =====================================================
generate_data = function(m, n, signal, sigma, num_support)
{
  beta_star = rep(0, n)
  beta_star[1:6] = signal
  
  X = matrix(rnorm(m*n), m, n)
  Y = beta_star[1] * X[, 1] + beta_star[2] * X[, 2] * X[, 3] +
      beta_star[3] * X[, 4] + beta_star[4] * X[, 5] * X[, 6] + 
      beta_star[5] * X[, 7] + beta_star[6] * X[, 8] + rnorm(m, 0, sigma)
  
  return (list(X=X, Y=Y))
}


# ===================== Hierarchical model =====================================================
generate_data = function(m, n, signal, sigma, num_support)
{
  
  X = matrix(rnorm(m*n), m, n)
  X_s = X[, 1:num_support]
  
  W1 = matrix(rnorm(num_support*32), num_support, 32)
  W2 = rnorm(32)
  
  Y = pmax(X_s %*% W1, 0) %*% W2 + rnorm(m, 0, sigma)
  
  return (list(X=X, Y=Y))
}


# ===================== Binary classification =====================================================
generate_data = function(m, n, signal, sigma, num_support)
{
  beta_star = rep(0, n)
  beta_star[1:num_support] = signal
  
  X = matrix(rnorm(m*n), m, n)
  logits = X %*% beta_star + rnorm(m, 0, sigma)
  p = 1/(1 + exp(-logits))
  Y = (logits > 0.5) * 1
  
  return (list(X=X, Y=Y))
}

# ===================== Multi-class classification =====================================================
library(reticulate)
use_condaenv('python3.12', required=TRUE)


python_code=
'
from sklearn.datasets import make_classification
def multi_classes(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, seed=1):
  X, Y = make_classification(n_samples=n_samples, 
                             n_features=n_features, 
                             n_informative=n_informative, 
                             n_redundant=n_redundant,
                             n_repeated = n_repeated,
                             n_classes=n_classes, 
                             shuffle=False, random_state=seed)
  return X, Y
'

datasets = import('sklearn.datasets')


dat = datasets$make_classification(n_samples=as.integer(200), n_features=as.integer(50), n_informative=as.integer(8), 
                                   n_redundant=as.integer(0),
                                   n_repeated = as.integer(0),
                                   n_classes=as.integer(5), 
                                   shuffle=TRUE, random_state=as.integer(0))



m = 200
n = 300
signal = 1
sigma = 0.5
num_support = 8

# results = rep(0, 50)

start_time = Sys.time()
for (i in 1:50){
  set.seed(i)
  print(i)
  dat = generate_data(m, n, signal, sigma, num_support)
  # dat = datasets$make_classification(n_samples=as.integer(200), n_features=as.integer(50), n_informative=as.integer(8), 
  #                                    n_redundant=as.integer(0),
  #                                    n_repeated = as.integer(0),
  #                                    n_classes=as.integer(5), 
  #                                    shuffle=FALSE, random_state=as.integer(i))
  X = dat[[1]]
  Y = dat[[2]]
  
  TVS_output <- TVS(Y, X, selector= 'bart', maxnrep = 10000, stop_crit = 100, niter = 500, ntree = 10)
  
  prob = posterior(TVS_output)
  # print(prob[1:10])

  y_pred = (prob > 0.9) * 1

  write.table(t(y_pred), file = "m200_n300_intersect.txt", sep = " ", row.names = FALSE, col.names = FALSE, append=TRUE, quote = TRUE)
  
  # results = rbind(results, y_pred)
  
  print(paste('seed', i, 'completed'))
  
  rm(TVS_output, prob, dat, X, Y)  # 删除临时对象
  gc()  # 强制垃圾回收
}


end_time = Sys.time()
print(end_time - start_time)





