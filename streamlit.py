import pandas as pd
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

def welcome():
    return 'welcome all'

def prediction(p, d, q,NodesInLayer,epochs):  
    df = pd.read_csv("VNINDEX.csv")
    ColumnToPredict = 'CLOSE'
    df.dropna(subset=[ColumnToPredict], inplace=True)
    df.DATE = pd.to_datetime(df.DATE)
    df[ColumnToPredict] = pd.to_numeric(df[ColumnToPredict])
    model = ARIMA(df[ColumnToPredict], order=(p,d,q))
    model_fit = model.fit()
    prediction = model_fit.forecast(5, alpha=0.05)
    print(prediction)
    
    from sklearn.preprocessing import MinMaxScaler
    import numpy
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

    dataset = df[['DATE' ,ColumnToPredict]]
    dataset = dataset.set_index('DATE')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.9975)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(NodesInLayer, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs ,batch_size=1, verbose=1)
    testPredict=  model.predict(testX)
    testPredict = scaler.inverse_transform(testPredict)
    
    ensembleModel = np.mean([prediction.values,testPredict.reshape((5,))],axis=0)
    
    return prediction , testPredict, ensembleModel

def main():
      # giving the webpage a title
    st.title("ARIMA & LSTM model forecasting")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit ARIMA & LSTM model forecasting App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    p = st.number_input("P Value",0,10)
    d = st.number_input("D Value",0,10)
    q = st.number_input("Q Value",0,10)
    
    NodesInLayer = st.number_input("Nodes In Layer",1,10)
    epochs = st.number_input("epochs",1,100)
    result =""
    result2=""
    result3= ""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result,result2,result3 = prediction(p,d,q,NodesInLayer,epochs)
    st.success('The output from ARIMA is {}'.format(result))
    st.success('The output from LSTM is {}'.format(result2)) 
    st.success('The output from Ensemble(combine) model is {}'.format(result3)) 
    
if __name__=='__main__':
    main()