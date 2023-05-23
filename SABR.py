from functools import partial

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.stats import norm
import warnings
#lgt
from scipy.interpolate import interp1d
import pandas as pd
import os
from datetime import datetime, timedelta

#bs模型计算call
def bs_call(S, K, T, r, vol):
    N = norm.cdf
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

#bs模型计算vega
def bs_vega(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# S, K, r, T, target_value = forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, bid
def blsimpv(S, K, r, T, target_value, *args):
    result = []
    for _ in range(len(S)):
        MAX_ITERATIONS = 200
        PRECISION = 1.0e-5
        sigma = 0.5
        for i in range(0, MAX_ITERATIONS):
            price = bs_call(S[_], K[_], T[_], r[_], sigma)
            vega = bs_vega(S[_], K[_], T[_], r[_], sigma)
            diff = target_value[_] - price  # our root
            if abs(diff) < PRECISION:
                result.append(sigma)
                break
            else:
                sigma = sigma + diff / vega  # f(x) / f'(x)
    return np.array(result)

#从对应文件夹中调取数据
def get_data(date,symbol,timestamp):
    cwd = os.getcwd()
    files = os.listdir(cwd+"\\"+date+"_"+symbol+"\\"+date)
    df_all = pd.DataFrame()
    for file in files:
        #file = files[0]
        strike = int(file.split("-")[3])
        if file.split("-")[4][0] == "c":
            put = 0
        elif file.split("-")[4][0] == "p":
            put = 1
        else:
            print("表名不规范")
            # return
        df = pd.read_csv(cwd+"\\"+date+"_"+symbol+"\\"+date+"\\"+file)
        df = df[["datetime","ask_price1","bid_price1"]]
        df = df.sort_values("datetime")
        df = df[df["datetime"]<=timestamp].tail(1)
        df["datetime"] = timestamp
        df["put"] = put
        df["strike"] = strike
        df_all = pd.concat([df_all,df],axis = 0)
    underlying_data = pd.read_csv(cwd + "\\" + date + "_" + symbol + "\\CZCE-" + symbol + "-" + date + ".tick.csv")
    underlying_data = underlying_data.sort_values("datetime")
    underlying_data = underlying_data[underlying_data["datetime"] <= timestamp].tail(1)
    underlying_data["mid"] = (underlying_data["ask_price1"]+underlying_data["bid_price1"])/2
    df_all["mid"] = list(underlying_data["mid"])[0]
    df_all["maturity"] = 24/250
    return df_all

#处理数据，计算bs模型的参数
def process_data(date,symbol,timestamp):
    data = get_data(date,symbol,timestamp)
    data = data.reset_index(drop = True)
    data["interest_rate"] = 0.015
    data["dividend_yield"] = 0.015
    data = data.applymap(lambda x:[x])
    put, strike, bid, ask, close, maturity,interest_rate,dividend_yield= list(data["put"]), list(data["strike"]), list(data["bid_price1"]), list(data["ask_price1"]),list(data["mid"]), list(data["maturity"]),list(data["interest_rate"]),list(data["dividend_yield"])
    put, strike, bid, ask, close, maturity,interest_rate,dividend_yield = np.array(put),np.array(strike),np.array(bid),np.array(ask),np.array(close),np.array(maturity),np.array(interest_rate),np.array(dividend_yield)

    # 删除不可用数据：非正买卖价差，买入价低于3/8，实值期权
    forward = close * np.exp((interest_rate - dividend_yield) * maturity)  # 计算远期价格
    log_moneyness = np.log(strike / forward)  # 期权价位
    pos_remove = (ask <= bid) | (bid < 3 / 8) | ((strike < forward) & (put == 0)) | (
            (forward <= strike) & (put == 1))  # 删除不符合条件的数据
    ask = ask[~pos_remove]  # 删除不符合条件的ask价格
    bid = bid[~pos_remove]  # 删除不符合条件的bid价格
    forward = forward[~pos_remove]  # 删除不符合条件的前继价格
    maturity = maturity[~pos_remove]  # 删除不符合条件的期限
    put = put[~pos_remove]  # 删除不符合条件的put数据
    strike = strike[~pos_remove]  # 删除不符合条件的strike数据
    interest_rate = interest_rate[~pos_remove]  # 删除不符合条件的interest_rate数据
    dividend_yield = dividend_yield[~pos_remove]  # 删除不符合条件的dividend_yield数据
    log_moneyness = log_moneyness[~pos_remove]

    bid[put == 1] = bid[put == 1] + np.exp(-interest_rate[put == 1] * maturity[put == 1]) * (
            forward[put == 1] - strike[put == 1])  # 计算put期权的买入价格
    ask[put == 1] = ask[put == 1] + np.exp(-interest_rate[put == 1] * maturity[put == 1]) * (
            forward[put == 1] - strike[put == 1])  # 计算put期权的卖出价格

    mid = (bid + ask) / 2  # 计算中间价格
    return ask,bid,forward,maturity,put,strike,interest_rate,dividend_yield,log_moneyness,mid


##以下为优化sabr

def sabr_raw(f, k, t, param):
    """
    Calculate the SABR implied volatility.

    Parameters:
    alpha (float): Initial volatility level
    beta (float): Power for the forward price (between 0 and 1)
    rho (float): Correlation between the asset price and its volatility (between -1 and 1)
    nu (float): Volatility of volatility
    f (float): Forward price
    k (float): Strike price
    t (float): Time to maturity

    Returns:
    float: Implied volatility
    """
    assert len(param) == 4, "参数必须有四个元素: alpha, beta, rho, nu"
    alpha, beta, rho, nu = param
    
#    if f == k:
#        fk_term = (1 - beta) ** 2 / 24 * alpha ** 2 / f ** (2 - 2 * beta)
#        rho_term = rho * beta * nu * alpha / (4 * f ** (1 - beta))
#        nu_term = (2 - 3 * rho ** 2) * nu ** 2 / 24
#        vol = alpha * (1 + (fk_term + rho_term + nu_term) * t) / f ** (1 - beta)
#    else:
    z = nu / alpha * (f * k) ** ((1 - beta) / 2) * np.log(f / k)
    x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    fk_term1 = (1 - beta) ** 2 / 24 * (np.log(f / k)) ** 2
    fk_term2 = (1 - beta) ** 4 / 1920 * (np.log(f / k)) ** 4
    vol = alpha * (1 + (fk_term1 + fk_term2) * t) * z / (x * (f * k) ** ((1 - beta) / 2))

    return vol

##最优化函数
#根据上下界生成参数初始值
def generate_random_start_values(lb, ub):
        lb[np.isinf(lb)] = -1000
        ub[np.isinf(ub)] = 1000
        param0 = lb + np.random.rand(len(lb)) * (ub - lb)

        return param0

#代价函数
def fit_function_sabr(x, f, k, t,  implied_vol):
    param = x
    model_vol = sabr_raw(f, k, t, param)
    #print(model_vol)
    value = np.linalg.norm(
        implied_vol - model_vol.reshape([len(model_vol), ]))#矩阵元素平方和的平方根，类似于rmse
    
    return value

#拟合SABR曲面
def fit_sabr(implied_vol, f, k, t):
        lb = np.array([0.1, 0, -1, 0])
        ub = np.array([0.9, 1, 1, 10])
        #lb = np.array([0.0001,0,-1, 0])
        #ub = np.array([np.inf,np.inf,1,np.inf])
        # constraints = {'type': 'ineq', 'fun': lambda x: mycon(x, phifun)}

        targetfun = partial(fit_function_sabr, f=f, k=k, t=t, 
                            implied_vol=implied_vol)

        N = 100
        parameters = np.zeros((len(lb), N))
        fun_value = np.zeros(N)

        for n in range(N):
            param0 = generate_random_start_values(lb, ub)
            # lgt
            #minimize_method = "Powell" #运行速度慢，很多点nan，因为无法考虑限制条件
            minimize_method = "SLSQP" #还可以
            #minimize_method = 'L-BFGS-B' #效果很离谱
            #minimize_method = 'trust-constr' #全是报错，输出nan
            #minimize_method = 'BFGS' #不考虑边界条件
            #minimize_method = 'TNC' #效果很差
            #minimize_method = 'Nelder-Mead' #效果很差
            #minimize_method = 'CG' #全是报错，输出nan
            #minimize_method = 'Newton-CG' #需要梯度函数（导函数）
            #minimize_method = 'COBYLA' #运行速度非常慢，会卡死
            res = minimize(targetfun, param0, method=minimize_method, bounds=Bounds(lb, ub),
                           options={'disp': False})
            res.fun
            
            # print(res.fun)
            parameters[:, n] = res.x
            fun_value[n] = res.fun

        idx = np.argmin(fun_value)
        parameters = parameters[:, idx]

        return parameters

def sabr_fit_all(date,symbol,timestamp):
    ask, bid, forward, maturity, put, strike, interest_rate, dividend_yield, log_moneyness, mid = process_data(date,symbol,timestamp)
    # 估计隐含波动率
    # 计算买入价格的隐含波动率
    implied_volatility_bid = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, bid)
    # 计算卖出价格的隐含波动率
    implied_volatility_ask = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, ask)
    # 计算中间价格的隐含波动率
    implied_volatility = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, mid)

    # 绘制总隐含波动率
    #total_implied_volatility = implied_volatility ** 2 * maturity  # 计算总隐含波动率
    #total_implied_variance = total_implied_volatility

    parameters= fit_sabr(implied_volatility, forward, strike, maturity)  # 拟合SABR曲面
    print(parameters)
    total_implied_variance = implied_volatility ** 2 * maturity  # 计算总隐含方差


    model_vol= sabr_raw(forward, strike, maturity, parameters)

    df_result = pd.DataFrame()
    df_result["log_moneyness"] = log_moneyness
    df_result["iv_ask"] = implied_volatility_ask
    df_result["iv_bid"] = implied_volatility_bid
    df_result["iv_sabr"] = model_vol
    df_result["timestamp"] = timestamp
    return df_result


## Example usage:
#alpha = 0.2
#beta = 0.5
#rho = -0.3
#nu = 0.4
#f = 100
#k = 110
#t = 1
#
#implied_vol = sabr_vol(alpha, beta, rho, nu, f, k, t)
#sabr_raw(f, k, t, (alpha, beta, rho, nu))
#print("Implied volatility:", implied_vol)


# time_interval = 3600
date = "20230428"
symbol = "SR307"
timestamp = '2023-04-28 09:42:00.0'
result = sabr_fit_all(date,symbol,timestamp)
plt.plot(result["log_moneyness"], result["iv_ask"], 'xb')
plt.plot(result["log_moneyness"], result["iv_bid"], 'xb')
plt.plot(result["log_moneyness"], result["iv_sabr"], 'xr')

plt.show()

ask, bid, forward, maturity, put, strike, interest_rate, dividend_yield, log_moneyness, mid=process_data(date,symbol,timestamp)


fit_function_sabr((0.9,0.84046222,0.33309287,3.09134478), forward, strike, maturity,  
                  blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, mid))

