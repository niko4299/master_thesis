import torch
from torch import nn
import empyrical

class Test_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, beta=0.1, size_average=True, device="cpu"):
        super(Test_Loss, self).__init__()
        self.gamma = gamma 
        self.beta = beta
        self.size_average = size_average
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate
        self.device = device

    def forward(self, w, close_price):  
        close_price = torch.cat([torch.ones(close_price.shape[0],1,1).to(self.device),close_price],dim = 1).to(self.device) 
        reward = torch.matmul(w, close_price)  
        close_price = close_price.view(close_price.size()[0], close_price.size()[2], close_price.size()[1])  
        element_reward = w * close_price
        interest = torch.zeros(element_reward.size(), dtype=torch.float).to(self.device)
        interest[element_reward < 0] = element_reward[element_reward < 0]
        interest = torch.sum(interest, 3).unsqueeze(3) * self.interest_rate
        future_omega = w * close_price / reward  
        wt = future_omega[:, :-1] 
        wt1 = w[:, 1:]  
        pure_pc = 1 - torch.sum(torch.abs(wt - wt1), -1) * self.commission_ratio  
        pure_pc = pure_pc.to(self.device)
        pure_pc = torch.cat([torch.ones([pure_pc.size()[0], 1, 1]).to(self.device), pure_pc],1)  
        pure_pc = pure_pc.view(pure_pc.size()[0], pure_pc.size()[1], 1, pure_pc.size()[2])  
        # cost_penalty = torch.sum(torch.abs(wt - wt1), -1) 
        reward = reward * pure_pc 
        
        reward = reward + interest
        x = reward.squeeze()
        print(torch.cumprod(x,axis = 0))
        if not self.size_average:
            tst_pc_array = reward.squeeze()
            sr_reward = tst_pc_array - 1
            sr_reward = sr_reward[:211]
            SR = empyrical.sharpe_ratio(sr_reward.cpu().numpy(),annualization=len(sr_reward))
            CR = empyrical.calmar_ratio(sr_reward.cpu().numpy(),annualization=len(sr_reward))
            print(SR,CR)
            SR=sr_reward.mean()/sr_reward.std()
#            print("SR:",SR.size(),"reward.mean():",reward.mean(),"reward.std():",reward.std())
            SN=torch.prod(reward,1) #[1,1,1,1]
            SN=SN.squeeze() #
#            print("SN:",SN.size())
            St_v=[]
            St=1.            
            MDD=max_drawdown(tst_pc_array)
            for k in range(reward.size()[1]):  #2808-31
                St*=reward[0,k,0,0]
                St_v.append(St.item())
            CR=SN/MDD            
            
            
        portfolio_value = torch.prod(reward, 1)  
        batch_loss = -torch.log(portfolio_value)  

        if self.size_average:
            loss = batch_loss.mean()
            return loss, portfolio_value.mean()
        else:
            loss = batch_loss.mean()
            return loss, portfolio_value[0][0][0], SR, CR

def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)