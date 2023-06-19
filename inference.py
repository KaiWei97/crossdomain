import torch
from torch.optim import optimizer
from torch.utils.data import DataLoader
from data_loader import *
from lea_model_pt import LEA_GCN
from eval_fn import eval_score
from LEA_Setting import Settings
torch.autograd.set_detect_anomaly(True)

args = Settings()


train_data = LEADataset('data\Douban', train=True)
test_data = LEADataset('data\Douban', train=False)

matrix_form_graph = train_data.get_matrix_form_graph()

train_loader = DataLoader(train_data, args.batch_size,
                          shuffle=True, drop_last=True, collate_fn=collect_fn)
test_loader = DataLoader(test_data, args.batch_size,
                         shuffle=False, drop_last=True, collate_fn=collect_fn)

dic = torch.load('best.pt')

model = LEA_GCN(n_items_A=dic['n_items_A'], n_items_B=dic['n_items_B'],
                n_users=dic['n_users'], graph_matrix=dic['graph_matrix'])
model.load_state_dict(dic['weights'])

model.to(args.device)
RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A = 0, 0, 0, 0, 0, 0
RC_5_B, RC_10_B, RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B = 0, 0, 0, 0, 0, 0
NDCG_5_A, NDCG_10_A, NDCG_20_A, NDCG_5_B, NDCG_10_B, NDCG_20_B = 0, 0, 0, 0, 0, 0
best_score_domain_A = -1
best_score_domain_B = -1

for epoch in range(args.epochs):
 for elem in test_loader:
    pred_A, pred_B = model(elem[:-2])
    # pred_A = torch.argmax(pred_A, -1)
    # pred_B = torch.argmax(pred_B, -1)
    target_A, target_B = elem[-2:]
    test_len =len(elem[2]+elem[3])

    RC_A, MRR_A, NDCG_A = eval_score(pred_A.cpu().detach().numpy(), target_list=target_A.cpu().numpy(), options=[5, 10, 20])

    RC_5_A += RC_A[0]
    RC_10_A += RC_A[1]
    RC_20_A += RC_A[2]
    MRR_5_A += MRR_A[0]
    MRR_10_A += MRR_A[1]
    MRR_20_A += MRR_A[2]
    NDCG_5_A += NDCG_A[0]
    NDCG_10_A += NDCG_A[1]
    NDCG_20_A += NDCG_A[2]



    RC_B, MRR_B, NDCG_B = eval_score(pred_B.cpu().detach().numpy(), target_list=target_B.cpu().numpy(),options= [5, 10, 20])
    RC_5_B += RC_B[0]
    RC_10_B += RC_B[1]
    RC_20_B += RC_B[2]
    MRR_5_B += MRR_B[0]
    MRR_10_B += MRR_B[1]
    MRR_20_B += MRR_B[2]
    NDCG_5_B += NDCG_B[0]
    NDCG_10_B += NDCG_B[1]
    NDCG_20_B += NDCG_B[2]


    if RC_5_A >= best_score_domain_A or RC_5_B >= best_score_domain_B:
        best_score_domain_A = RC_5_A
        best_score_domain_B = RC_5_B
        print("Recommender performs better, saving current model....")
        print(f'predA RC={RC_A}\tMRR={MRR_A}\tNDCG={NDCG_A}')
        print(f'predB RC={RC_B}\tMRR={MRR_B}\tNDCG={NDCG_B}')









    

