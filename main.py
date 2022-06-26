from pathlib import Path
import torch
from helpers import make_model
from losses.batch_loss import Batch_Loss
from losses.test_loss import Test_Loss
from losses.optimizer import NoamOpt
from environment.env import StockTradingEnv
import json
import argparse
from train_test import train_net,test_net

def run_main(args):

    with open(args.config_file,"r") as fhandle:
        ctx = json.load(fhandle)

    launch_train(ctx, args.data_file_path)


def launch_train(ctx: dict, data_file_path):
    if not (Path(ctx["model_dir"]) / ctx["model_name"] / str(ctx["model_index"])).exists():
        (Path(ctx["model_dir"]) / ctx["model_name"] / str(ctx["model_index"])).mkdir(parents=True)

    lr_model_sz = 5120
    factor = ctx["learning_rate"] 
    warmup = 0  

    total_step = ctx["total_step"]
    x_window_size = ctx["x_window_size"]

    batch_size = ctx["batch_size"]
    coin_num = ctx["selected_symbols"] 
    feature_number = ctx["selected_features"]  
    trading_consumption = ctx["trading_consumption"]  
    variance_penalty = ctx["variance_penalty"]  
    cost_penalty = ctx["cost_penalty"] 
    output_step = ctx["output_step"]
    local_context_length = ctx["local_context_length"]
    model_dim = ctx["model_dim"]
    weight_decay = ctx["weight_decay"]
    interest_rate = ctx["daily_interest_rate"] / 24 / 2

    device = ctx["device"]
    model = make_model(batch_size, coin_num, x_window_size, feature_number,
                       N=1, d_model_Encoder=ctx["multihead_num"] * model_dim,
                       d_model_Decoder=ctx["multihead_num"] * model_dim,
                       d_ff_Encoder=ctx["multihead_num"] * model_dim,
                       d_ff_Decoder=ctx["multihead_num"] * model_dim,
                       h=ctx["multihead_num"],
                       dropout=0.01,
                       local_context_length=local_context_length,device = device)
    model.to(device)

    # model.load_state_dict(torch.load('leiden_rat_models/2020-04-28.pkl',map_location=device))
            
    model_opt = NoamOpt(lr_model_sz, factor, warmup,
                        torch.optim.Adam(model.parameters(), lr=ctx["learning_rate"], betas=(0.9, 0.98), eps=1e-9,
                                         weight_decay=weight_decay))
    loss_function = Batch_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty, device=device)

    env_train = StockTradingEnv(start_date=ctx['train']['start'],end_date=ctx['train']['end'],trading_window_size = x_window_size,path = f"portfolio_files/train/portfolios_{ctx['train']['start']}_{ctx['train']['end']}.txt",number_of_days_trading=ctx['number_of_trading_days'],assets_number=coin_num,bias  = ctx['buffer_bias_ratio'], batch_size = batch_size, device=device, data_file_path = data_file_path)
    env_val = StockTradingEnv(start_date = ctx['val']['start'],end_date = ctx['val']['end'],trading_window_size = x_window_size, path = f"portfolio_files/val/portfolios_{ctx['val']['start']}_{ctx['val']['end']}.txt",number_of_days_trading=ctx['number_of_trading_days'],assets_number=coin_num,bias  = ctx['buffer_bias_ratio'], batch_size = batch_size, device=device, data_file_path = data_file_path)
    tst_loss, tst_portfolio_value = train_net(env_train,env_val,total_step, output_step, x_window_size,
                                              local_context_length, model,
                                              ctx["model_dir"]+'/'+ ctx["model_name"] +'/'+ str(ctx["model_index"]) , ctx["model_index"], loss_function,
                                              model_opt, device = device)
    model.eval()
    env_test = StockTradingEnv(start_date=ctx['test']['start'],end_date = ctx['test']['end'],trading_window_size = x_window_size, path = f"portfolio_files/test/portfolios_{ctx['test']['start']}_{ctx['test']['end']}.txt",number_of_days_trading=ctx['number_of_trading_days'],assets_number=coin_num,bias  = ctx['buffer_bias_ratio'], batch_size = batch_size, random = False, device=device, data_file_path = data_file_path)      
    test_loss = Test_Loss(trading_consumption, interest_rate, variance_penalty, cost_penalty,size_average=False, device=device)
    test_net(env_test,x_window_size=x_window_size,model= model,local_context_length=local_context_length,loss_func = test_loss, device = device)
    
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type = str,default='default_configs.json',help='config file')
    parser.add_argument('--data_file_path',type = str,default='mongo_db/total_df.pkl',help='config file')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    run_main(args)
