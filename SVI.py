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

def bs_call(S, K, T, r, vol):
    N = norm.cdf
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

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

##以下为优化svi
def svi_raw(k, param, tau=None):


    k = np.asarray(k)

    assert len(param) == 5, "参数必须有五个元素: a, b, m, rho, sigma"

    a, b, m, rho, sigma = param

    # 计算总方差
    totalvariance = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    if tau is not None:
        # 如果需要，计算隐含波动率
        impliedvolatility = np.sqrt(totalvariance / tau)
    else:
        impliedvolatility = None

    return totalvariance, impliedvolatility
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





    # 估计隐含波动率
    # 计算买入价格的隐含波动率
    implied_volatility_bid = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, bid)
    # 计算卖出价格的隐含波动率
    implied_volatility_ask = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, ask)
    # 计算中间价格的隐含波动率
    implied_volatility = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, mid)

    # 绘制总隐含波动率
    total_implied_volatility = implied_volatility ** 2 * maturity  # 计算总隐含波动率
    maturities = np.sort(np.unique(maturity))  # 获取期限的不同值
    T = len(maturities)  # 计算期限的数量

    # fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    # for i in range(3):
    #     for j in range(4):
    #         t = i * 3 + j
    #         pos = maturity == maturities[t]  # 获取期限为t的数据的位置
    #         idx = np.argsort(log_moneyness[pos])  # 对数据的货币价值的对数排序
    #         x = log_moneyness[pos][idx]
    #         y = total_implied_volatility[pos]  # 获取期限为t的总隐含波动率
    #         y = y[idx]  # 根据排序的结果重新排列总隐含波动率
    #         axs[i, j].plot(x, y)  # 绘制图形

    # 拟合SSVI曲面
    phifun = 'power_law'  # 定义函数形式
    parameters, _, _ = fit_svi_surface(implied_volatility, maturity, log_moneyness, phifun)  # 拟合SSVI曲面

    total_implied_variance = implied_volatility ** 2 * maturity  # 计算总隐含方差
    model_total_implied_variance = np.zeros(total_implied_variance.shape)  # 初始化模型总隐含方差
    model_implied_volatility = np.zeros(total_implied_variance.shape)  # 初始化模型隐含波动率

    for t in range(T):
        pos = maturity == maturities[t]  # 获取期限为t的数据的位置

        # 计算模型总隐含方差和模型隐含波动率

        model_total_implied_variance[pos], model_implied_volatility[pos] \
            = svi_jumpwing(log_moneyness[pos], parameters[:, t], maturities[t])
        model_total_implied_variance, model_implied_volatility \
            = svi_jumpwing(log_moneyness2, parameters[:, t], maturities[t])

    df_result = pd.DataFrame()
    df_result["log_moneyness"] = list(map(lambda x:x[0],log_moneyness2))
    df_result = df_result.drop_duplicates().reset_index(drop =True)
    df_result["iv_ask"] = 0
    df_result.loc[25-len(implied_volatility_ask):,"iv_ask"] = implied_volatility_ask
    df_result["iv_bid"] = 0
    df_result.loc[25-len(implied_volatility_bid):,"iv_bid"] = implied_volatility_bid
    df_result["iv_ssvi"] = model_implied_volatility[::2]
    df_result["timestamp"] = timestamp

##最优化函数
def generate_random_start_values(lb, ub):
        lb[np.isinf(lb)] = -1000
        ub[np.isinf(ub)] = 1000
        param0 = lb + np.random.rand(len(lb)) * (ub - lb)

        return param0

def fit_function_svi(x, log_moneyness,  total_implied_variance):
    param = x
    model_total_implied_variance, _ = svi_raw(log_moneyness, param, tau=None)
    value = np.linalg.norm(
        total_implied_variance - model_total_implied_variance.reshape([len(model_total_implied_variance), ]))#矩阵元素平方和的平方根，类似于rmse

    return value


    # 拟合SVI曲面

def fit_svi(total_implied_variance, log_moneyness):
        lb = np.array([0,0,-1,-np.inf, 0])
        ub = np.array([np.inf,np.inf,1, np.inf,np.inf])
        # constraints = {'type': 'ineq', 'fun': lambda x: mycon(x, phifun)}

        targetfun = partial(fit_function_svi, log_moneyness=log_moneyness,
                            total_implied_variance=total_implied_variance)

        N = 100
        parameters = np.zeros((len(lb), N))
        fun_value = np.zeros(N)

        for n in range(N):
            param0 = generate_random_start_values(lb, ub)
            # lgt
            minimize_method = "Powell"
            res = minimize(targetfun, param0, method=minimize_method, bounds=Bounds(lb, ub),
                           options={'disp': False})
            # print(res.fun)
            parameters[:, n] = res.x
            fun_value[n] = res.fun

        idx = np.argmin(fun_value)
        parameters = parameters[:, idx]

        return parameters

def svi_fit_all(date,symbol,timestamp):
    ask, bid, forward, maturity, put, strike, interest_rate, dividend_yield, log_moneyness, mid = process_data(date,symbol,timestamp)
    # 估计隐含波动率
    # 计算买入价格的隐含波动率
    implied_volatility_bid = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, bid)
    # 计算卖出价格的隐含波动率
    implied_volatility_ask = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, ask)
    # 计算中间价格的隐含波动率
    implied_volatility = blsimpv(forward * np.exp(-interest_rate * maturity), strike, interest_rate, maturity, mid)

    # 绘制总隐含波动率
    total_implied_volatility = implied_volatility ** 2 * maturity  # 计算总隐含波动率
    total_implied_variance =total_implied_volatility

    parameters= fit_svi(total_implied_variance, log_moneyness)  # 拟合SSVI曲面

    total_implied_variance = implied_volatility ** 2 * maturity  # 计算总隐含方差


    model_total_implied_variance, model_implied_volatility= svi_raw(log_moneyness, parameters, maturity[0])

    df_result = pd.DataFrame()
    df_result["log_moneyness"] = log_moneyness
    df_result["iv_ask"] = implied_volatility_ask
    df_result["iv_bid"] = implied_volatility_bid
    df_result["iv_svi"] = model_implied_volatility
    df_result["timestamp"] = timestamp
    return df_result

# time_interval = 3600
date = "20230428"
symbol = "SR307"
timestamp = '2023-04-28 09:42:00.0'
result = svi_fit_all(date,symbol,timestamp)
plt.plot(result["log_moneyness"], result["iv_ask"], 'xb')
plt.plot(result["log_moneyness"], result["iv_bid"], 'xb')
plt.plot(result["log_moneyness"], result["iv_svi"], 'xr')

plt.show()