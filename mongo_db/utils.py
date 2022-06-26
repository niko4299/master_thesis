from pymongo import MongoClient
import os
import mongo_db.config as config
import plotly.graph_objects as go
from torchvision import transforms
from PIL import Image
import io
WINDOW_SIZE = 20
transformer = transforms.ToTensor()

def create_tensor_image(symbol, current_ndarray_result_map):
    current_ndarray,result_map = current_ndarray_result_map
    sub_array = current_ndarray[(current_ndarray[:,5] == symbol)]
    if sub_array.shape[0] == WINDOW_SIZE:
        fig = go.Figure(data=[go.Candlestick(x = sub_array[:,0],
                open=sub_array[:,1], high=sub_array[:,2],
                low=sub_array[:,4], close=sub_array[:,3],
                increasing = {'fillcolor': '#3D9970',"line":{"color" : '#3D9970'}},decreasing = {'fillcolor' : '#FF4136',"line":{"color" : '#FF4136'}}),],layout = config.IMAGES_LAYOUT)

        fig_bytes = fig.to_image(format = 'png')
        buf = io.BytesIO(fig_bytes)
        img = Image.open(buf).convert('RGB')

        result_map[symbol] = transformer(img)


def create_history_file_images(ndarray,symbol,step):
    dir_path = f'{config.DATA_DIR_PATH}/{symbol}/'
    if os.path.exists(dir_path):
        print('SYMBOL:',symbol,'DONE!')
        return 
    os.makedirs(dir_path)
    for start in range(ndarray.shape[0]-step):
        end = start+step
        sub_array = ndarray[start:end]
        fig = go.Figure(data=[go.Candlestick(x = sub_array[:,0],
                    open=sub_array[:,1], high=sub_array[:,2],
                    low=sub_array[:,4], close=sub_array[:,3],
                    increasing = {'fillcolor': '#3D9970',"line":{"color" : '#3D9970'}},decreasing = {'fillcolor' : '#FF4136',"line":{"color" : '#FF4136'}}),],layout = config.IMAGES_LAYOUT)

        fig.write_image(f'{dir_path}/{start}_{end}.png')
    print('SYMBOL:',symbol,'DONE!')

def get_db_collection():
    uri = "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(config.USERNAME, config.PASSWORD, config.HOSTNAME, config.PORT, config.DB_NAME)
    client = MongoClient(uri,authSource="admin")
    db = client['diplomski_db']

    return db['stocks']
