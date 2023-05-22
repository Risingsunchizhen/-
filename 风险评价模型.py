import tushare as ts
import pandas as pd
pro = ts.pro_api('7c9670aec368eaf509ab283a3526853ba257993895389b2ded0e62f9')
import warnings
warnings.filterwarnings('ignore')
import backtrader as bt
import datetime
import numpy as np
from scipy.optimize import minimize
import streamlit as st
import streamlit as st
import datetime
import tushare as ts
import pandas as pd
pro = ts.pro_api('7c9670aec368eaf509ab283a3526853ba257993895389b2ded0e62f9')
import warnings
warnings.filterwarnings('ignore')
import backtrader as bt
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
plt.rcParams['xtick.direction']='in' #####刻度向内
plt.rcParams['ytick.direction']='out'
# font_manager.fontManager.addfont(r"D:\anaconda\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\{}.ttf".format(myfont))
# plt.rc('font',family='Alibaba PuHuiTi 2.0',color='#005493ff')
# plt.rcParams['font.weight']='bold'
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
plt.rcParams['axes.spines.right']=False###去掉图右边轴
plt.rcParams['axes.spines.top']=False #####去掉图上轴
plt.rcParams['axes.edgecolor']='#2F466E' ####使得边缘线是灰色#C0C0C0
plt.rcParams['axes.labelcolor']='#172A4B'
plt.rcParams["xtick.labelcolor"]='#172A4B'
plt.rcParams["ytick.labelcolor"]='#172A4B'
plt.rcParams['legend.labelcolor']='#172A4B'

start=st.date_input("开始日期是:")
st.write("开始日期:",start)
end=st.date_input("结束日期是:")
st.write("结束日期:",end)
code=st.text_input("Which code do you know best")
st.write("选取的股票代码:",code)
@st.cache_data
def get_data(code,start,end):
    data=pro.daily(ts_code=code,start_date=start,end_date=end)
    data['openinterest']=0
    data.rename(columns={'vol':'volume'},inplace=True)
    data['trade_date']=pd.to_datetime(data['trade_date'])
    data.sort_values(by='trade_date',inplace=True)
    data.rename(columns={'trade_date':'date'},inplace=True)
    return data[['date',"open","high","low","close","volume","openinterest"]]
data=get_data(code,start.strftime("%Y%m%d"),end.strftime("%Y%m%d"))

option = st.selectbox(
    'Which number do you like best?',
     data.columns)
st.write('You selected: ', option)
fig = plt.figure(figsize=(6,4))
plt.plot(data['date'],data[option])
plt.xticks(rotation=90)
st.pyplot(fig)

codes=st.text_input("你愿意用来做风险评价模型的股票代码:")
codes=str(codes)
codes_list=codes.split(',')

def get_ret(code,start,end):
    data=pro.daily(ts_code=code,start_date=start,end_date=end)
    data['openinterest']=0
    data.rename(columns={'vol':'volume'},inplace=True)
    data['trade_date']=pd.to_datetime(data['trade_date'])
    data.sort_values(by='trade_date',inplace=True)
    data.rename(columns={'trade_date':'date'},inplace=True)
    ret=np.log(data['close']/data['close'].shift(1))
    ret.name=code
    return ret
def risk_budget_objective(weights,cov):
    weights = np.array(weights) #weights为一维数组
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights))) #获取组合标准差
    #sigma = np.sqrt(weights@cov@weights)
    MRC = np.dot(cov,weights)/sigma  #MRC = cov@weights/sigma
    #MRC = np.dot(weights,cov)/sigma
    TRC = weights * MRC
    delta_TRC = [sum((i - TRC)**2) for i in TRC]
    return sum(delta_TRC)
def total_weight_constraint(x):
    return np.sum(x)-1.0

ret = pd.DataFrame()
for code in codes_list:
    ret_new=get_ret(code,start.strftime("%Y%m%d"),end.strftime("%Y%m%d"))
    ret = pd.concat([ret,ret_new],axis=1)
ret = ret.dropna()
R_cov = ret.cov() #计算协方差
cov= np.array(R_cov)
x0 = np.ones(cov.shape[0]) / cov.shape[0]
bnds = tuple((0,None) for x in x0)
cons = ({'type': 'eq', 'fun': total_weight_constraint})
#cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
options={'disp':False, 'maxiter':1000, 'ftol':1e-20}
solution = minimize(risk_budget_objective, x0,args=(cov), bounds=bnds, constraints=cons, method='SLSQP', options=options)
final_weights=solution.x
for i in range(len(final_weights)):
    st.write(f'{final_weights[i]:.1%}投资于{R_cov.columns[i]}')