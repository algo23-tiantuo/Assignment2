# 隐含波动率曲面模型梳理
## 摘要
在这一项目中，我对于一些常见的隐含波动率曲面拟合模型进行了测试，包括SVI模型，SABR模型和Wing模型。发现SVI和Wing模型拟合表现较好，SABR模型在平值期权附近稍差。SVI模型形式最为简单，相比而言也更有理论依据，是一个比较好的波动率拟合方法。
## 数据来源
数据为白糖SR307在2023年4月28日全天的tick数据。
## 拟合方法
从20230428——SR307文件夹中提取数据，提取买一卖一价格，在log-moneyness为正时计算call iv, 在log-moneyness为负时计算put iv，合成到一张图上。在模型拟合时，使用scipy.optimize里的minimize函数进行参数估计。
## SVI拟合结果
![](https://github.com/algo23-tiantuo/Assignment2/blob/main/svi.png)
## Wing拟合结果
![](https://github.com/algo23-tiantuo/Assignment2/blob/main/wing.png)
## SABR拟合结果
![](https://github.com/algo23-tiantuo/Assignment2/blob/main/sabr.png)
