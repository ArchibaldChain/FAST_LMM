# [FaST linear mixed models for genome-wide association studies](1)

Position:  'FAST_LMM.FaST_LMM.FASTLMM'

Type: Class

Model:

$$
Y_i = \sum_{j = 1}^{p} \beta_j X_{i,j} + u_i +\epsilon_i \ \text{where } i = 1, 2,3,\dots,n
$$
where $(u_1, u_2, \dots, u_n)^\top \sim MVN(0, \sigma_g^2 K)$ and $\epsilon_i \sim ^{i.i.d} N(0, \sigma_e^2)$, and  $K =  WW^\top$, $\sigma_e^2 = \delta  \sigma_g^2$ . 

### Parameters:
- `lowRank`: boolean, whether to use the low rank methods shown in the paper;
- `REML`: boolean, whether to use the REML methods or MLE methods.
### Attributes:
- `sigma_g2`: `float`, estimated $\sigma_g^2$
- `sigma_e2`: `float`, estimated $\sigma_e^2$
- `delta`: `float`, estimated $\frac{\sigma_e^2}{\sigma_g^2}$
- `beta`: `float`, estimated $\hat \beta$
- `X`: `np.array`, the fixed effects term of shape `(n, p)`
- `y`: `np.array`, the phenotype data of shape `(n, 1)`
- `W`: `np.array`, the random effects indicator matrix of shape `(n, sc)`. If it is set as None, `W = 1/ np.sqrt(n) X`.
- `rank`: `int`, the rank of `W`
- `U`: `np.array`, eigenvalues matrix calculated by using `svd(W)` of shape `(n, rank)` if `lowRank` is set as true.
- `S`: the eigenvalues array of `W`  of shape `(rank,)` if `lowRank` is set as true.

### Methods:

##### `fit(X:np.array, y:np.array, W=None)`:
> fitting the model
parameters:
- `X`: `np.array` with shape of `(n, p)`
- `y`: `np.array` with shape of `(n,1)`, if it is shape of `(n,)`, it will be reshape to `(n,1)`
- `W`: `np.array`, the random effects indicator matrix of shape `(n, sc)`. If it is set as None, `W = 1/ np.sqrt(n) X`.

return: None

##### `summary`:
> print the summary statistics

##### `V(W=None,  sigma_g2=None, sigma_e2=None)`:
> Get the $\text{var}(y, y)$. If all parameters are None then using the estimated values, otherwise using the parameter values.

##### `V_inv(self, W=None, sigma_g2=None, sigma_e2=None)`:
> Get the $\text{var}(y, y)^{-1}$. If all parameters are None then using the estimated values, otherwise using the parameter values.

##### `plot_likelihood(REML=True)`:
> Plotting the log-likelihood v.s. $log(\delta)$. Parameters is to determine whether to plot log-likelihood or restricted log-likelihood. The restricted log-likelihood is plot when bot `REML` and `self.REML` are set as True.

## Example
Example is available in `./test_FAST_LMM.py`.

[1]: https://www.nature.com/articles/nmeth.1681
