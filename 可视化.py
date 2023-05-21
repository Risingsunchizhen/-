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
myfont='AlibabaPuHuiTi-2-55-Regular'
import matplotlib.patches as mpatches
plt.rcParams['xtick.direction']='in' #####刻度向内
plt.rcParams['ytick.direction']='out'
font_manager.fontManager.addfont(r"D:\anaconda\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\{}.ttf".format(myfont))
# plt.rc('font',family='Alibaba PuHuiTi 2.0',color='#005493ff')
plt.rcParams["font.sans-serif"]=[myfont] #设置字体
# plt.rcParams['font.weight']='bold'
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
plt.rcParams['axes.spines.right']=False###去掉图右边轴
plt.rcParams['axes.spines.top']=False #####去掉图上轴
plt.rcParams['axes.edgecolor']='#2F466E' ####使得边缘线是灰色#C0C0C0
plt.rcParams['axes.labelcolor']='#172A4B'
plt.rcParams["xtick.labelcolor"]='#172A4B'
plt.rcParams["ytick.labelcolor"]='#172A4B'
plt.rcParams['legend.labelcolor']='#172A4B'

st.subheader('沪深300的初步研究')
@st.cache_data
def get_index(code,start,end):
    index= pro.index_daily(ts_code=code,start_date=start,end_date=end)
    index['trade_date']=pd.to_datetime(index['trade_date'])
    index=index.sort_values(by='trade_date',ascending=True)
    index['openinterest']=0
    index.rename(columns={'vol':'volume','trade_date':'date'},inplace=True)
    return index[['date',"open","high","low","close","volume","openinterest"]]
start=st.date_input("开始日期是:")
st.write("开始日期:",start)
end=st.date_input("结束日期是:")
st.write("结束日期:",start)
index_300=get_index(code='399300.SZ', start=start.strftime("%Y%m%d"), end=end.strftime('%Y%m%d'))
index_300['date']=pd.to_datetime(index_300['date'])
option = st.selectbox(
    'Which number do you like best?',
     index_300.columns)

st.write('You selected: ', option)
fig = plt.figure(figsize=(6,4))
plt.plot(index_300['date'],index_300[option])
plt.xticks(rotation=90)
st.pyplot(fig)


if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)