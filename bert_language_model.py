import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_traj_model import Bert_Traj_Model 


class NextSentencePredict(nn.Module):

    def __init__(self, d_model):
        super(NextSentencePredict, self).__init__()

        self.Linear = nn.Linear(d_model, 2)

    def forward(self, x):
        
        x = self.Linear(x[:, 0])
        return F.softmax(x, dim=-1)


class Masked_LM(nn.Module):

    def __init__(self, token_size, d_model):
        super().__init__()

        self.Linear = nn.Linear(d_model, token_size)

    def forward(self, x):
        x = self.Linear(x)
        return F.softmax(x, dim=-1)


class Predict_Model(nn.Module):

    def __init__(self, Bert_Traj_Model, token_size, head_n=12, d_model=768, N_layers=12, dropout=0.1):
        super(Predict_Model, self).__init__()

        self.bert = Bert_Traj_Model

        self.place_outemb = nn.Embedding(token_size, d_model-10, padding_idx=0)
        # self.time_outemb = nn.Embedding(49, 10)

        # self.place_Linear = nn.Linear(d_model-10, token_size)
        # self.time_Linear = nn.Linear(10, 49)

        # self.place_Linear.weight = Bert_Traj_Model.Embed.token_embed.token_embed.weight
        # self.time_Linear.weight = Bert_Traj_Model.Embed.token_embed.time_embed.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, input_time, max_len, pos=None, neg=None):

        is_target = pos.ne(0)

        x = self.bert(x, input_time, max_len)
        x = x[:,max_len:]

        # logit: [batch_size, seq_size, d_]
        # print("neg", neg.size())

        pos_emb = self.place_outemb(pos).unsqueeze(-2)
        neg_emb = self.place_outemb(neg)

        # print("pos", pos_emb.size(), "* x[:,:,:-10]", x[:,:,:-10].size())
        # print("neg", neg_emb.size(), "* x[:,:,:-10].unsqueeze(2) ", x[:,:,:-10].unsqueeze(2).size())
        # print("x[:,:,:-10]",x[:,:,:-10].size())
        
# #////////////////////////////////////////////////////
        # pos_logits = torch.sum(pos_emb * x[:,:,:-10].unsqueeze(2), -1)
        # neg_logits = torch.sum(neg_emb * x[:,:,:-10].unsqueeze(2), -1)
        # # print(neg_logits)
        # print("pos", pos_logits.size())
        # print("neg", neg_logits.size())

        # loss_pos = torch.sum(- torch.log(torch.sigmoid(pos_logits) + 1e-24) * is_target.unsqueeze(-1)) 
        # loss_neg = torch.sum( - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * is_target.unsqueeze(-1))
        # # ) / torch.sum(is_target)
        # loss_neg2 = torch.sum( - torch.log( torch.sigmoid(-neg_logits) + 1e-24) * is_target.unsqueeze(-1))
        # loss = loss_pos + loss_neg

        # print("loss1, loss_pos, loss_neg", loss.item(), loss_pos.item(), loss_neg.item())
        # print("loss_neg2", loss_neg2)
# #////////////////////////////////////////////////////////////

        # print("exit in predict model forward")
        # exit()

        # [batch_size, seq_size, embed_size]      x
        # [batch_size, seq_Size, embed_size]      pos
        # [batch_size, K, seq_size, embed_size]   neg

        #向量乘法
        x = x[:,:,:-10].unsqueeze(-1) # [batch_size, embed_size, 1],新增一个维度用于向量乘法
        # print("")

        # pos_emb = pos_emb.unsqueeze(-2)

        # print("x", x.size())
        # print("pos", pos_emb.size())
        # print("neg", neg_emb.size())
        # print(torch.matmul(pos_emb, x).squeeze().size())
        # exit()
        pos_dot = torch.matmul(pos_emb, x).squeeze() # [batch_size, seq_size ]z只保留前两维
        neg_dot = torch.matmul(neg_emb, x).squeeze() # [batch_size, K, seq_size)]z只保留前两维


        pos_dot = pos_dot * is_target
        neg_dot = neg_dot * is_target.unsqueeze(-1)


        log_pos = F.logsigmoid(pos_dot).sum() #按照公式计算
        log_neg = F.logsigmoid(-neg_dot).sum()

        loss = (-(log_pos + log_neg)) / torch.sum(is_target)
        # print(loss)
        # exit()
          
        return loss

    
    def predict(self, x, input_time, max_len, pos=None, neg=None):
        # forward(self, x, input_time, max_len, pos=None, neg=None):

        # is_target = pos.ne(0)

        x = self.bert(x, input_time, max_len)

        x = x[:,max_len:]


        logit = torch.matmul(x[:,:,:-10],self.place_outemb.weight.T)
        # logit2 = self.time_Linear(x[:,:,-10:])

        return logit.contiguous().view(-1, logit.size(-1))



