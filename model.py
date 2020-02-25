from keras.layers import Input, LSTM, Bidirectional, Dense, Multiply
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import Huber
from keras.metrics import mse
from keras.utils import plot_model


def construct_model():
    # 天气预测
    weather_in = Input(shape=(360, 17), name='weather_in')
    # 使用L2正规化防止过拟合
    rglrz = l2(1e-4)

    # 三层 Bidirectional LSTM堆叠
    lstm = Bidirectional(
        LSTM(32, 
        kernel_regularizer = rglrz,
        recurrent_regularizer=rglrz,
        bias_regularizer=rglrz,
        recurrent_initializer='glorot_uniform',
        return_sequences=True,
        name='lstm1'),
        merge_mode='concat',
        name='Bid_1'
    )(weather_in)

    lstm = Bidirectional(
        LSTM(32, 
        kernel_regularizer = rglrz,
        recurrent_regularizer=rglrz,
        bias_regularizer=rglrz,
        recurrent_initializer='glorot_uniform',
        return_sequences=True,
        name='lstm2'),
        merge_mode='concat',
        name='Bid_2'
    )(lstm)

    lstm = Bidirectional(
        LSTM(32, 
        kernel_regularizer = rglrz,
        recurrent_regularizer=rglrz,
        bias_regularizer=rglrz,
        recurrent_initializer='glorot_uniform',
        return_sequences=False,
        name='lstm3'),
        merge_mode='concat',
        name='Bid_3'
    )(lstm)

    # 全联接网络产生预测结果
    prediction = Dense(1, 
    kernel_initializer='glorot_uniform', 
    kernel_regularizer=rglrz, 
    name='prediction')(lstm)

    # 地理位置情况
    county_in = Input(shape=(88,), name='county_in')
    county = Dense(32, activation='tanh', name='county1')(county_in)
    county = Dense(1, activation='sigmoid', name='county2')(county)

    # 合并两个网络的结果，使用乘
    pre_county = Multiply(name='pre_county')([prediction, county])
    merge_model = Model(inputs=[weather_in, county_in], outputs=pre_county, name='merge_model')

    # 编译
    merge_model.compile(
        optimizer=Adam(),
        loss=Huber(),
        metrics=[mse]
    )
    return merge_model
    
    
if __name__ == '__main__':
    print('begin')
    model = construct_model()
    print(model.summary)
    plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True)