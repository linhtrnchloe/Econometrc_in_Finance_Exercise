[price] = readmatrix("Gold_Platinum_LBMA.xlsx");
log_return_gold = price(:,3);
log_return_platinum = price(:,5);

% Ex 1
% Ex 2
[acf_gold, lags_gold, bounds_gold] = autocorr(log_return_gold, 'NumLags', 10, 'NumSTD', 2);
[acf_platinum, lags_platinum, bounds_platinum] = autocorr(log_return_platinum, 'NumLags', 10, 'NumSTD', 2);

disp('Gold ACF for lags 1, 2, 3:');
disp(acf_gold(2:4));
disp('Platinum ACF for lags 1, 2, 3:');
disp(acf_platinum(2:4));

% Define maximum lags to test
maxLags = [2, 8, 10];

% Perform Ljung-Box test for each lag
for i = 1:length(maxLags)
    lag = maxLags(i);
    
    % Perform test for gold
    [h_gold, pValue_gold] = lbqtest(log_return_gold, 'Lags', lag, 'Alpha', 0.1);
    
    % Perform test for platinum
    [h_platinum, pValue_platinum] = lbqtest(log_return_platinum, 'Lags', lag, 'Alpha', 0.1);
    
    % Display results
    fprintf('Ljung-Box Test Results for Gold at lag %d:\n', lag);
    fprintf('Reject H0: %d, p-value: %.4f\n', h_gold, pValue_gold);
    
    fprintf('Ljung-Box Test Results for Platinum at lag %d:\n', lag);
    fprintf('Reject H0: %d, p-value: %.4f\n', h_platinum, pValue_platinum);
end

% Perform Augmented Dickey-Fuller (ADF) test with 3 lags and constant, no trend
%[h, pValue, stat, critVal] = adftest(log_return, 'Lags', 3);

% p-value is 0.001 < 0.05, reject H0 -> stationary

% ADF test on gold price
%gold_price = price(:,2);
%[h, pValue, stat, critVal] = adftest(gold_price, 'Lags', 3); % p-value = 0.9458 -> cannot reject H0 -> not stationary -> stochastic trends


% ex 2
% Estimate MA(1), AR(1), and ARMA(1,1) models
modelMA1 = arima('MALags', 1, 'Constant', 0);
modelAR1 = arima('ARLags', 1, 'Constant', 0);
modelARMA11 = arima('ARLags', 1, 'MALags', 1, 'Constant', 0);

% Fit models to the log returns of platinum
[fitMA1, ~, logL_MA1] = estimate(modelMA1, log_return_platinum);
[fitAR1, ~, logL_AR1] = estimate(modelAR1, log_return_platinum);
[fitARMA11, ~, logL_ARMA11] = estimate(modelARMA11, log_return_platinum);

% Compute Bayesian Information Criterion (BIC)
bicMA1 = aicbic(logL_MA1, 2, length(log_return_platinum));  % MA(1) has 2 parameters
bicAR1 = aicbic(logL_AR1, 2, length(log_return_platinum));  % AR(1) has 2 parameters
bicARMA11 = aicbic(logL_ARMA11, 3, length(log_return_platinum)); % ARMA(1,1) has 3 parameters

% Display BIC values
fprintf('BIC for MA(1): %.4f\n', bicMA1);
fprintf('BIC for AR(1): %.4f\n', bicAR1);
fprintf('BIC for ARMA(1,1): %.4f\n', bicARMA11);

% Select the best model based on the smallest BIC
[~, bestModelIdx] = min([bicMA1, bicAR1, bicARMA11]);
bestModelName = {'MA(1)', 'AR(1)', 'ARMA(1,1)'};
fprintf('Best model based on BIC: %s\n', bestModelName{bestModelIdx});

%----
% Get residuals from the best model
residuals = infer(fitARMA11, log_return_platinum);
% Compute squared residuals
squared_residuals = residuals.^2;

% Perform Ljung-Box test on squared residuals for lag up to 20
[~, pValue_McLeodLi] = lbqtest(squared_residuals, 'Lags', 20, 'Alpha', 0.05);

fprintf('McLeod-Li Test p-value: %.4f\n', pValue_McLeodLi);
% Perform Engle's ARCH test with max lag 20
[h_ARCH, pValue_ARCH] = archtest(residuals, 'Lags', 20);

fprintf('Engle’s ARCH Test: Reject H0: %d, p-value: %.4f\n', h_ARCH, pValue_ARCH);
% -> have both correlation between squared residual and volatility

% Ex 3
% Estimate AR(1)-GARCH(1,1) with Student's t-distribution
Mdl1 = arima('ARLags', 1, 'Variance', garch(1, 1), 'Distribution', 't');
[EstMdl1, EstParamCov1, LogLik1] = estimate(Mdl1, log_return_gold);

% Estimate AR(2)-GARCH(1,2) with Student's t-distribution
Mdl2 = arima('ARLags', 1:2, 'Variance', garch(1, 2), 'Distribution', 't');
[EstMdl2, EstParamCov2, LogLik2] = estimate(Mdl2, log_return_gold);

% Perform Likelihood Ratio Test
LR_stat = 2 * (LogLik2 - LogLik1);
pValue_LR = 1 - chi2cdf(LR_stat, 2); % df = 2 (additional parameters in AR(2)-GARCH(1,2))
fprintf('Likelihood Ratio Test: LR stat = %.4f, p-value = %.4f\n', LR_stat, pValue_LR);

% Estimate AR(1)-EGARCH(1,1)
Mdl3 = arima('ARLags', 1, 'Variance', egarch(1, 1), 'Distribution', 't');
[EstMdl3, EstParamCov3, LogLik3] = estimate(Mdl3, log_return_gold);

% Estimate AR(2)-GJRGARCH(1,1)
Mdl4 = arima('ARLags', 1:2, 'Variance', gjr(1, 1), 'Distribution', 't');
[EstMdl4, EstParamCov4, LogLik4] = estimate(Mdl4, log_return_gold);

% AIC and BIC Calculations
Models = {'AR(1)-GARCH(1,1)', 'AR(2)-GARCH(1,2)', 'AR(1)-EGARCH(1,1)', 'AR(2)-GJRGARCH(1,1)'};
LogLik = [LogLik1, LogLik2, LogLik3, LogLik4];
NumParams = [4, 7, 4, 6]; % Adjust based on model specifications
AIC = -2 * LogLik + 2 * NumParams;
BIC = -2 * LogLik + NumParams * log(length(log_return_gold));

% Display Results
for i = 1:length(Models)
    fprintf('%s: AIC = %.4f, BIC = %.4f\n', Models{i}, AIC(i), BIC(i));
end

% Engle's ARCH Test for residuals of the best model
bestModelResiduals = infer(EstMdl3, log_return_gold);
[h_ARCH, pValue_ARCH] = archtest(bestModelResiduals, 'Lags', 20);
fprintf('Engle’s ARCH Test: Reject H0: %d, p-value: %.4f\n', h_ARCH, pValue_ARCH);

% EX 4
EstMdl1 = estimate(Mdl1, log_return_gold);
[residuals, ~] = infer(EstMdl1, log_return_gold);
[~, condVar] = infer(EstMdl1, log_return_gold); % Calculate the conditional variance (volatility)
condStdDev = sqrt(condVar); % Conditional standard deviation
standardizedResiduals = residuals ./ condStdDev; % Standardized residuals
% Test for autocorrelation in standardized residuals
[h_res, pValue_res] = lbqtest(standardizedResiduals, 'Lags', 20, 'Alpha', 0.05);

% Test for autocorrelation in squared standardized residuals
squaredStdResiduals = standardizedResiduals.^2;
[h_sq, pValue_sq] = lbqtest(squaredStdResiduals, 'Lags', 20, 'Alpha', 0.05);

% Display results
fprintf('Ljung-Box Test for Standardized Residuals: Reject H0: %d, p-value: %.4f\n', h_res, pValue_res);
fprintf('Ljung-Box Test for Squared Standardized Residuals: Reject H0: %d, p-value: %.4f\n', h_sq, pValue_sq);


%-----
% Ex 5
% Set the random number generator for reproducibility
rng(12);

% Parameters
r21 = 0.04;  % initial return (r^2) for the first time point
sigma21 = 0.04;  % initial volatility (sigma^2)
lambda = 0.05;  % decay factor for EWMA
n = 200;  % number of samples

% Preallocate arrays to store returns and volatilities
returns = zeros(1, n);
volatility = zeros(1, n);

% Initial values
returns(1) = sqrt(r21) * randn;  % first return generated from the given r21
volatility(1) = sqrt(sigma21);  % first volatility generated from the given sigma21

% Simulate the returns and volatility using the EWMA process
for t = 2:n
    % Calculate the volatility for the current time point using the EWMA formula
    volatility(t) = sqrt(lambda * returns(t-1)^2 + (1 - lambda) * volatility(t-1)^2);
    
    % Generate the return for the current time point using the volatility
    returns(t) = volatility(t) * randn;
end

% Forecast the 1-day-ahead volatility for the next observation (observation no. 201)
forecast_volatility = sqrt(lambda * returns(n)^2 + (1 - lambda) * volatility(n)^2);

% Find the maximum in-sample return (from observations 1 to 200)
max_return = max(returns);

% Display results
fprintf('1-day-ahead volatility forecast for observation 201: %.4f\n', forecast_volatility);
fprintf('Maximum in-sample return (from obs 1 to 200): %.4f\n', max_return);


% VaR and Backtesting
rng(12);
log_returns = randn(1000, 1) * 0.02; 
alpha = 0.95;
z = norminv(1 - alpha); % Z-score for normal distribution
VaR = -z * std(log_returns); % Forecasted VaR
% Identify Exceedances
exceedances = log_returns < -VaR; % 1 if return < -VaR, 0 otherwise
numExceedances = sum(exceedances);
expectedExceedances = length(log_returns) * (1 - alpha);
% Unconditional Coverage Test
p_hat = numExceedances / length(log_returns); % Observed exceedance rate
LR_UC = -2 * (expectedExceedances * log(1 - alpha) + numExceedances * log(alpha)) ...
          + 2 * (expectedExceedances * log(1 - p_hat) + numExceedances * log(p_hat));
pValue_UC = 1 - chi2cdf(LR_UC, 1);
% Print Results
fprintf('Unconditional Coverage Test:\n');
fprintf('LR_UC Statistic: %.4f, p-value: %.4f\n', LR_UC, pValue_UC);

% Independence Test (Markov approach)
transitions = diff([0; exceedances]);
p00 = sum(transitions == 0 & ~exceedances(1:end-1)) / sum(~exceedances(1:end-1));
p11 = sum(transitions == 0 & exceedances(1:end-1)) / sum(exceedances(1:end-1));

LR_Ind = -2 * (log((1 - p_hat)^numExceedances * p_hat^(1 - numExceedances)) - ...
          log(p00^numExceedances * p11^(1 - numExceedances)));
pValue_Ind = 1 - chi2cdf(LR_Ind, 1);

% Print Results
fprintf('Independence Test:\n');
fprintf('LR_Ind Statistic: %.4f, p-value: %.4f\n', LR_Ind, pValue_Ind);

% -- estimate GARCH model for normal distribution and t-distribution
% Define GARCH(1,1) with Normal Distribution
Mdl_Normal = garch(1, 1);

% Define GARCH(1,1) with Student's t-Distribution
Mdl_t = garch(1, 1, 'Distribution', 't');
% Estimate GARCH(1,1) with Normal Distribution
EstMdl_Normal = estimate(Mdl_Normal, log_returns);

% Estimate GARCH(1,1) with Student's t-Distribution
EstMdl_t = estimate(Mdl_t, log_returns);
% Loglikelihood
fprintf('Log-likelihood (Normal): %.4f\n', EstMdl_Normal.LogLikelihood);
fprintf('Log-likelihood (t): %.4f\n', EstMdl_t.LogLikelihood);
% Number of observations and parameters
n_obs = length(log_returns);
n_params_Normal = 3; % omega, alpha, beta (Normal)
n_params_t = 4; % omega, alpha, beta, DoF (Student's t)

% AIC and BIC for Normal
AIC_Normal = -2 * EstMdl_Normal.LogLikelihood + 2 * n_params_Normal;
BIC_Normal = -2 * EstMdl_Normal.LogLikelihood + log(n_obs) * n_params_Normal;

% AIC and BIC for Student's t
AIC_t = -2 * EstMdl_t.LogLikelihood + 2 * n_params_t;
BIC_t = -2 * EstMdl_t.LogLikelihood + log(n_obs) * n_params_t;
fprintf('AIC (Normal): %.4f, BIC (Normal): %.4f\n', AIC_Normal, BIC_Normal);
fprintf('AIC (t): %.4f, BIC (t): %.4f\n', AIC_t, BIC_t);

% Forecast 1-day-ahead volatility
[v_Normal, ~] = forecast(EstMdl_Normal, 1, 'Y0', log_returns);
[v_t, ~] = forecast(EstMdl_t, 1, 'Y0', log_returns);
fprintf('1-Day Ahead Volatility (Normal): %.4f\n', sqrt(v_Normal));
fprintf('1-Day Ahead Volatility (t): %.4f\n', sqrt(v_t));



