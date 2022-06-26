from mongo_db.utils import create_tensor_image
import numpy as np
from multiprocessing import Manager
from functools import partial
import torch
from autoencoder.model import CAEC
from sklearn.metrics.pairwise import cosine_similarity
import autoencoder.config as aec_config
from igraph import *
WINDOW_SIZE = 20

class PortfolioConstruct():
    def __init__(self,portfolio_stock_num, step, autoencoder_path = 'autoencoder/model_weights_13.pth'):
        self.portfolio_stock_num = portfolio_stock_num
        self.step = step
        self.aec_model = CAEC(in_channels=3, encoder_arhitecture_layers_size=aec_config.encoder_arhitecture_layers_size,decoder_arhitecture_layers_size=aec_config.decoder_arhitecture_layers_size)  
        self.aec_model.load_state_dict(torch.load(autoencoder_path,map_location=torch.device('cpu')))

    def create_images(self,current_ndarray):
        symbols = np.unique(current_ndarray[:,5])
        np.random.shuffle(symbols)
        symbols = symbols[:100]
        manager = Manager()
        all_images = manager.dict()
        with manager.Pool(processes = 10) as pool:
            pool.map(partial(create_tensor_image, current_ndarray_result_map=(current_ndarray,all_images)), symbols )
        
        return np.array(all_images.keys()), all_images.values()

    def calculate_sharpes(self,partition_symbol_indexes,current_ndarray,symbols, stocks_to_take,take_from_each):
        cluster_stock_symbols = symbols[partition_symbol_indexes]
        sharpes_ratios = []
        for symbol in cluster_stock_symbols:
            sub_array = current_ndarray[current_ndarray[:,5] == symbol]
            r = np.log((sub_array[:,3]/sub_array[:,1]).astype(float))
            r_mean = np.mean(r)
            r_std = np.std(r)
            sharpes_ratios.append(r_mean/r_std)

        indexes_stocks = np.argsort(sharpes_ratios)[::-1][:take_from_each] 
        stocks_to_take.update({cluster_stock_symbols[index]:sharpes_ratios[index] for index in indexes_stocks})

    def get_partitions(self,extracted_features):
        weights = cosine_similarity(extracted_features)
        graph = Graph.Weighted_Adjacency(weights, mode=ADJ_UNDIRECTED, attr="weight", loops=False)
        
        return graph.community_leiden( weights=graph.es['weight'],resolution_parameter=0.8,n_iterations = -1)

    def choose_stocks(self,partitions,current_ndarray, symbols):
        take_from_each = int(self.portfolio_stock_num/len(partitions))
        manager = Manager()
        stocks_to_take = manager.dict()
        with manager.Pool(processes = 25) as pool:
            pool.map(partial(self.calculate_sharpes, current_ndarray = current_ndarray,symbols = symbols,stocks_to_take = stocks_to_take,take_from_each = take_from_each), partitions)

        return stocks_to_take
    
    def remove_n_minimums(self,d, n):
        for _ in range(n):
            min_key = min(d.keys(), key=lambda k: d[k])
            del d[min_key]

    def construct_portfolio(self,current_ndarray):
        symbols, images = self.create_images(current_ndarray)
        with torch.no_grad():
            X = torch.stack(images)
            extracted_features = self.aec_model.encoder_forward(X).cpu().numpy()
        leiden_partitions = self.get_partitions(extracted_features)
        portfolio_stocks = self.choose_stocks(leiden_partitions,current_ndarray,symbols)
        current_portfolio_len = len(portfolio_stocks)
        if current_portfolio_len == self.portfolio_stock_num:
            return portfolio_stocks.keys()
        elif current_portfolio_len > self.portfolio_stock_num:
            self.remove_n_minimums(portfolio_stocks,current_portfolio_len-self.portfolio_stock_num)
            return portfolio_stocks.keys()
        else:
            missing_for_portoflio = self.portfolio_stock_num - current_portfolio_len
            filtered_current_ndarray = current_ndarray[np.isin(current_ndarray[:,5],portfolio_stocks, invert=True)]
            left_symbols = np.unique(filtered_current_ndarray[:,5])
            sharpes_ratios = []
            for symbol in left_symbols:
                symbol_array = filtered_current_ndarray[filtered_current_ndarray[:,5] == symbol]
                r = np.log((symbol_array[:,3]/symbol_array[:,1]).astype(float))
                r_mean = np.mean(r)
                r_std = np.std(r)
                sharpes_ratios.append(r_mean/r_std)
            indexes = np.argsort(sharpes_ratios)[::-1][:missing_for_portoflio] 
            portfolio = portfolio_stocks.keys()
            portfolio.extend(left_symbols[indexes])
            
            return portfolio
