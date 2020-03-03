import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import tensorflow as tf
print(tf.__version__)
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

train_weather_path = "./data/train_weather.csv"
train_rice_path = "./data/train_rice.csv"
testA_path = "./data/testA.csv"
testB_path = "./data/testB.csv"

r_df = pd.read_csv(train_rice_path, encoding='gbk')

is_zao = False    # True   or    False

def get_ycr(cid,year):
    '''获取某年·某县的早/晚水稻产量'''
    r_ci = r_df[r_df.区县id == cid]
    if is_zao:
        year = str(year)+'年早稻'  
    else:
        year = str(year)+'年晚稻'    
    
    return r_ci[year].values[0]

dtypes = {   '区县id': object,
             '站名id': np.int64,
             '年份':   np.int64,
             '月份':   np.int64,
             '日期':   np.int64,
             '日照时数（单位：h)': np.float64,
             '02时风向': object,
             '08时风向': object,
             '14时风向': object,
             '20时风向': object,
             '日平均风速(单位：m/s)':   np.float64,
             '日降水量（mm）':          np.float64,
             '日最高温度（单位：℃）':  np.float64,
             '日最低温度（单位：℃）':  np.float64,
             '日平均温度（单位：℃）':  np.float64,
             '日相对湿度（单位：%）':   np.float64,
             '日平均气压（单位：hPa）': np.float64}
na_values =['*','/']
w_df = pd.read_csv(train_weather_path,dtype=dtypes, na_values=na_values,encoding='gbk').fillna(0)
w_df.columns = ['CountyID',  'StationID',   'Year',    'Month',    'Day',           'Hours',
                'F02',        'F08',          'F14',     'F20',      'WindS',
                'Rain',      'HighT',       'LowT',    'MeanT',    'Humidity',      'Pressure']

def z_map(column):
    cmax = column.max(axis=0)
    cmin = column.min(axis=0)
    new_column = (column - cmin) / (cmax - cmin)
    return new_column

need_scale_cloumns = ['Hours',  'WindS','Rain','HighT','LowT','MeanT','Humidity', 'Pressure' ]


for column in  need_scale_cloumns:
    w_df[column] = z_map(w_df[column])


def parse_countyID(x,need_onehot=True): 
    '''把县的ID转换成onehot编码,县的ID由1到88中的81个数组成，在实现onehot编码时使用编码长度88'''
    length = len('county')
    cid = x[length:]
    cid_num = int(cid)
    if need_onehot:        
        cid_vec = [0]*88
        cid_vec[cid_num-1] = 1
        return cid_vec
    else:
        return cid_num

print(parse_countyID('county87'))


def fx_to_vec(x):
    '''把风向由字符转换成二维的向量，第一个维度代表水平方向，第二个维度代表竖直方向'''
    angle = np.pi / 8.

    fx_string_list = [     'E',   'ENE',   'NE',   'NNE',
                           'N',   'NNW',   'NW',   'WNW',
                           'W',   'WSW',   'SW',   'SSW',
                           'S',   'SSE',   'SE',   'ESE',   'C', 0        ]  
    # 从E(东)方向开始，逆时针转动的16个方向,无风时：‘C’   
    index = fx_string_list.index(x)
    if index < 16:
        horizontal, vertical = np.cos(index*angle), np.sin(index*angle)
    else:
        horizontal, vertical = 0., 0.
    if horizontal-0 < 1e-4:
        horizontal=0
    if vertical-0 < 1e-4:
        vertical=0
    return horizontal, vertical

def get_ycw(year,countyID):
    '''获取某年·某县的天气特征
    1 把每个风向由字符串表示成一个二维向量
    2 把县ID、站名ID和年月日丢弃，用归一化的索引代表时间，并只取每年的前360天
    3 保留降水量等其余特征
    结果：某年·某县的天气特征为：360*17'''
    w_ci = w_df[w_df.CountyID == countyID]
    w_ci_yi = w_ci[w_ci.Year == year]
        
    w_ci_yi = w_ci_yi.reset_index(drop=True)
    w_ci_yi = w_ci_yi[w_ci_yi.index < 360]

    for column in [ 'F02','F08','F14','F20']:
        f_s = w_ci_yi.pop(column).values          # 把 F02 的每个元素由‘N’转换成，（0，1）

        fx = []
        fy = []
        for item in f_s:
            item_vec = fx_to_vec(item)
            fx.append(item_vec[0])                 # 每个（0，1）中的第一个数字，代表风向在X方向的分量
            fy.append(item_vec[1])

        data = np.array([fx,fy])                  # 【2，360】
        data = np.transpose(data)                 # 【360，2】
        f_pd = pd.DataFrame(data=data,columns=[column+'_X',column+'_Y'])
        w_ci_yi = pd.concat([w_ci_yi,f_pd],axis=1,ignore_index=False)

    CountyID_onehot = w_ci_yi.CountyID.map(parse_countyID).values
    CountyID_onehot = np.stack(CountyID_onehot)
    w_ci_yi_17features = w_ci_yi.drop(columns=['CountyID','StationID','Year','Month','Day'])    
    w_ci_yi_17features = w_ci_yi_17features.reset_index(drop=False)
    w_ci_yi_17features['index'] = w_ci_yi_17features['index'] / 360. -0.5
    return w_ci_yi_17features.values

valid_county_ID = r_df.区县id.values

weather_list = []
countyID_list = []
rice_list = []
for year in range(2015,2018+1):
    for countyID in valid_county_ID:
        ycw = get_ycw(year,countyID)                      #  【360， 17】  
        weather_list.append(ycw)
        
        ycc = parse_countyID(countyID)                    #  【88，】     
        countyID_list.append(ycc)
        
        if year < 2018:
            ycr = get_ycr(countyID,year)           #  【1，】
            rice_list.append(ycr)
        
weather = np.stack(weather_list)
countyID = np.stack(countyID_list)
rice = np.stack(rice_list)

idx15 = list(range(81*0, 81*1))
idx16 = list(range(81*1, 81*2))
idx17 = list(range(81*2, 81*3))
idx18 = list(range(81*3, 81*4))

train_index = idx15 + idx16*2                      # 训练集：      15年 + 2倍16年
valid_index = idx17                                # 验证集：      17年
all_index   = idx15 + idx16*2 + idx17*4            # 所有数据集 ： 15年 + 2倍16年 + 4倍17年
test_index  = idx18                                # 测试集：      18年

bs = 81

train_dt = tf.data.Dataset.from_tensor_slices(
    ((weather[train_index],countyID[train_index]),rice[train_index])).repeat().shuffle(10000).batch(bs)
valid_dt = tf.data.Dataset.from_tensor_slices(
    ((weather[valid_index],countyID[valid_index]),rice[valid_index])).repeat().batch(bs)
all_dt = tf.data.Dataset.from_tensor_slices(
    ((weather[all_index],  countyID[all_index]),  rice[all_index]  )).repeat().shuffle(10000).batch(bs)

def build_and_compile_model():
    inputs1 = tf.keras.Input(shape=(360,17),name='inputs1')
    inputs2 = tf.keras.Input(shape=(88,),name='inputs2')
    L2 = tf.keras.regularizers.l2(1e-4)
    ##############################################################################################################
    lstm1 = keras.layers.Bidirectional(
           layers.LSTM(32,  kernel_regularizer=L2,
                            recurrent_regularizer=L2,
                            bias_regularizer=L2,
                            recurrent_initializer='glorot_uniform',
                            return_sequences=True,
                            name='lstm1'),
                                        merge_mode='concat',
                                        name='Bid_1')(inputs1)
    lstm2 = keras.layers.Bidirectional(
           layers.LSTM(32,  kernel_regularizer=L2,
                            recurrent_regularizer=L2,
                            bias_regularizer=L2,
                            recurrent_initializer='glorot_uniform',
                            return_sequences=True,
                            name='lstm2'),
                                        merge_mode='concat',
                                        name='Bid_2')(lstm1)
    
    lstm3 = keras.layers.Bidirectional(
           layers.LSTM(64,  kernel_regularizer=L2,
                            recurrent_regularizer=L2,
                            bias_regularizer=L2,
                            recurrent_initializer='glorot_uniform',
                            return_sequences=False,
                            name='lstm3'),
                                        merge_mode='concat',
                                        name='Bid_3')(lstm2)    
    ##############################################################################################################
    pre = layers.Dense(     units = 1,                      
                            kernel_initializer='glorot_uniform',
                            kernel_regularizer=L2,                 
                            name='predict')(lstm3)
    ##############################################################################################################
    county_condition1 = layers.Dense(32,
                                    activation='tanh', 
                                    name='county_condition1')(inputs2)
    county_condition2 = layers.Dense(1,
                                    activation='sigmoid', 
                                    name='county_condition2')(county_condition1)
    contrl_pre = tf.multiply(pre,   county_condition2, name='contrl_pre',)   # 使用countyID 约束 预测的产量
    
    
    merge_model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=contrl_pre, name='merge_model')
    ##############################################################################################################
    merge_model.compile( optimizer=tf.keras.optimizers.Adam(),
                         loss=tf.keras.losses.Huber(),
                         metrics = [tf.keras.metrics.mse])
    return merge_model

merge_model = build_and_compile_model()

# merge_model.load_weights('../user_data/merge_model.h5')
cbk = tf.keras.callbacks.EarlyStopping(     monitor='val_mean_squared_error',
                                            min_delta=0,   # 忽略
                                            patience=100,
                                            verbose=1,
                                            mode='min',
                                            baseline=None,
                                            restore_best_weights=True,)


merge_model_h = merge_model.fit(train_dt,     epochs=1000,         steps_per_epoch  = len(train_index)//bs,
                                validation_data=valid_dt,          validation_steps = 1,
                                callbacks=[cbk] )

merge_model.evaluate(train_dt,steps =  2*6)
merge_model.evaluate(valid_dt,steps =  1*4)

def plot_learning_history(history,y_max):
    pd.DataFrame(history.history).plot(figsize=(20,5))
    plt.grid(True)
    plt.gca().set_ylim(0, y_max)
    plt.show()
    
try:
    plot_learning_history(merge_model_h,1)
except NameError as Error:
    print(Error)