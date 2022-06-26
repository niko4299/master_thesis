import time
import torch
from helpers import make_std_mask


def train_net(env_train, env_test, total_step, output_step, x_window_size, local_context_length, model, model_dir,
              model_index,
              loss_func, optimizer, device):
    start = time.time()
    total_loss = 0
    max_val_portfolio_value = 0
    min_val_loss = 0
    for i in range(total_step):
        model.train()
        out, trg_y = run_one(env_train, x_window_size, model, local_context_length, device)
        loss, portfolio_value = loss_func(out, trg_y)
        loss.backward()
        optimizer.step()
        optimizer.optimizer.zero_grad()
        total_loss += loss.item()
        if i % output_step == 0:
            tst_total_loss = 0
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                  (i, loss.item(), portfolio_value.item(), output_step / elapsed))
            print(time.time() - start)
            start = time.time()
            with torch.no_grad():
                model.eval()
                tst_out, tst_trg_y = run_one(env_test, x_window_size, model, local_context_length, device)
                tst_loss, tst_portfolio_value = loss_func(tst_out, tst_trg_y)
                tst_total_loss += tst_loss.item()
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
                      (i, tst_loss.item(), tst_portfolio_value.item(), 1 / elapsed))
                start = time.time()

                if tst_portfolio_value > max_val_portfolio_value:
                    max_val_portfolio_value = tst_portfolio_value
                    min_val_loss = tst_loss
                    torch.save(model.state_dict(), model_dir + '/' + str(model_index) + ".pkl")
                    print("save model!")

        

    return min_val_loss, max_val_portfolio_value

def test_net(env_test,x_window_size, model, local_context_length,loss_func,device):
    start = time.time()
    with torch.no_grad():
        model.eval()
        tst_long_term_w, tst_trg_y = test_run(env_test, x_window_size, model, local_context_length, device)
        tst_loss, tst_portfolio_value, SR, CR = loss_func(tst_long_term_w,tst_trg_y)
        elapsed = time.time() - start
        print("Test Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f |testset per Sec: %f" %
              (tst_loss.item(), tst_portfolio_value.item(), SR, CR, 1 / elapsed))
  
       
def test_run(env_test, x_window_size, model, local_context_length, device):
    tst_batch_input,tst_batch_y, _ = env_test.get_all()
    tst_previous_w = torch.ones((1,tst_batch_y.shape[1]),  dtype= torch.float, device = device) * 1/tst_batch_y.shape[1]
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)
    tst_batch_input = tst_batch_input.permute((1, 0, 3, 2))

    tst_src_mask = (torch.ones(tst_batch_input.size()[1], 1, x_window_size) == 1)

    long_term_tst_currt_price = tst_batch_input.permute((3, 1, 2, 0))

    tst_trg_mask = make_std_mask(long_term_tst_currt_price[:, :, 0:1, :], tst_batch_input.size()[1])

    tst_long_term_w = []

    for j in range(tst_batch_y.shape[0]): 
        if j % 20 == 0:
            tst_previous_w = torch.ones((1,25),  dtype= torch.float, device = 'cpu') * 1/25
            tst_previous_w = torch.unsqueeze(tst_previous_w, 1)

        tst_src = tst_batch_input[:,j,:, :]
        tst_src = tst_src.unsqueeze(1)
        tst_currt_price = long_term_tst_currt_price[:, j, -1, :]
        tst_currt_price = tst_currt_price.unsqueeze(1).unsqueeze(2)
        if local_context_length > 1:
            padding_price = tst_src[:, :, -local_context_length * 2 + 1:-1, :]
            padding_price = padding_price.permute((3, 1, 2, 0))    
        else:
            padding_price = None
        out = model(tst_src.to(device), tst_currt_price.to(device), tst_previous_w.to(device),tst_src_mask.to(device), tst_trg_mask.to(device), padding_price.to(device))
        if j == 0:
            tst_long_term_w = out.unsqueeze(0) 
        else:
            tst_long_term_w = torch.cat([tst_long_term_w, out.unsqueeze(0)], 0)
        out = out[:, :, 1:] 
        tst_previous_w = out
    tst_long_term_w = tst_long_term_w.permute(1, 0, 2, 3) 
    
    return tst_long_term_w, tst_batch_y.unsqueeze(-1)

def run_one(env, x_window_size, model, local_context_length,device):
    batch_input,batch_y, batch_last_w, indexes = env.get_batch()

    previous_w = torch.unsqueeze(batch_last_w, 1)  
    batch_input = batch_input.permute((1, 0, 3, 2))
    price_series_mask = (torch.ones(batch_input.size()[1], 1, x_window_size) == 1) 
    currt_price = batch_input.permute((3, 1, 2, 0)) 
    if local_context_length > 1:
        padding_price = currt_price[:, :, -local_context_length * 2 + 1:-1, :]
    else:
        padding_price = None

    currt_price = currt_price[:, :, -1:, :]
    trg_mask = make_std_mask(currt_price, batch_input.size()[1])
    out = model(batch_input.to(device), currt_price.to(device), previous_w.to(device),
                        price_series_mask.to(device), trg_mask.to(device), padding_price.to(device))

    new_w = out[:, 0, 1:].detach()
    env.set_new_weigts(indexes,new_w)

    return out, batch_y.unsqueeze(-1)

