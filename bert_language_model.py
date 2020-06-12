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

        neg = pos+1
        is_target = pos.ne(0)
        # print("target", is_target)

        x = self.bert(x, input_time, max_len)

        x = x[:,max_len:]
        # print("x size", x.size())
        # print("exit in preddict")
        # eixt()

        # logit: [batch_size, seq_size, d_]

        pos_emb = self.place_outemb(pos)
        neg_emb = self.place_outemb(neg)

        # print("pos", pos_emb.size())
        # print("neg", neg_emb.size())

        pos_logits = torch.sum(pos_emb * x[:,:,:-10], -1)
        neg_logits = torch.sum(neg_emb * x[:,:,:-10], -1)
        # print("pos",pos_logits.size())

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * is_target -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * is_target
        ) / torch.sum(is_target)
        # print("loss", loss)

        # print("exit in predict model forward")
        # exit()

        # logit2 = self.time_Linear(x[:,:,-10:])
        
        return loss
        # return loss, logit2.contiguous().view(-1, logit2.size(-1))
    
    def predict(self, x, input_time, max_len, pos=None, neg=None):
        # forward(self, x, input_time, max_len, pos=None, neg=None):

        # is_target = pos.ne(0)

        x = self.bert(x, input_time, max_len)

        x = x[:,max_len:]


        logit = torch.matmul(x[:,:,:-10],self.place_outemb.weight.T)
        # logit2 = self.time_Linear(x[:,:,-10:])

        return logit.contiguous().view(-1, logit.size(-1))



