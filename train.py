import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_loader import LEADataset, collect_fn
from lea_model_pt import LEA_GCN
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


n_items_A = len(train_data.dict_A)
n_items_B = len(train_data.dict_B)
n_users = len(train_data.dict_U)
model = LEA_GCN(n_items_A=n_items_A, n_items_B=n_items_B,
                n_users=n_users, graph_matrix=matrix_form_graph)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer_A = torch.optim.Adam(model.parameters(), lr=args.lr_A)
optimizer_B = torch.optim.Adam(model.parameters(), lr=args.lr_B)


min_test_loss = 1e5
model.to(args.device)

for epoch in range(args.epochs):
    print(f'Epochs {epoch+1}/{args.epochs}')
    model.train()
    avg_loss_A = 0
    avg_loss_B = 0
    print('start Train.')
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        pred_A, pred_B = model(batch[:-2])

        target_A, target_B = batch[-2:]
        loss_A1 = loss_fn(pred_A, target_A)
        loss_A2 = args.l2_regular_rate * \
            (torch.sum(model.seq_embed_A**2) + torch.sum(model.user_embed**2))
        loss_A = loss_A1 + loss_A2

        loss_B1 = loss_fn(pred_B, target_B)
        loss_B2 = args.l2_regular_rate * \
            (torch.sum(model.seq_embed_B**2) + torch.sum(model.user_embed**2))
        loss_B = loss_B1 + loss_B2

        gradients = optimizer_A.zero_grad()
        gradients = optimizer_B.zero_grad()
        loss_A.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        loss_B.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer_A.step()
        optimizer_B.step()

        avg_loss_A += loss_A
        avg_loss_B += loss_B

    train_batch_num = len(train_loader)

    rec_loss_A = avg_loss_A / train_batch_num
    rec_loss_B = avg_loss_B / train_batch_num
    print(f'total_train_loss={rec_loss_A+rec_loss_B}\ttrain_loss_A={rec_loss_A}\t train_loss_B={rec_loss_B}\n')

    # test
    model.eval()
    avg_loss_A = 0
    avg_loss_B = 0
    print('start Test.')
    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            pred_A, pred_B = model(batch[:-2])

            target_A, target_B = batch[-2:]
            loss_A1 = loss_fn(pred_A, target_A)
            loss_A2 = args.l2_regular_rate * \
                (torch.sum(model.seq_embed_A**2) + torch.sum(model.user_embed**2))
            loss_A = loss_A1 + loss_A2

            loss_B1 = loss_fn(pred_B, target_B)
            loss_B2 = args.l2_regular_rate * \
                (torch.sum(model.seq_embed_B**2) + torch.sum(model.user_embed**2))
            loss_B = loss_B1 + loss_B2

        avg_loss_A += loss_A
        avg_loss_B += loss_B

    test_batch_num = len(test_loader)

    rec_loss_A = avg_loss_A / test_batch_num
    rec_loss_B = avg_loss_B / test_batch_num
    test_loss = rec_loss_A+rec_loss_B
    print(f'total_test_loss={test_loss}\ttest_loss_A={rec_loss_A}\t test_loss_B={rec_loss_B}\n')

    # 保存最优模型
    if test_loss < min_test_loss:
        print('save best checkpoint!')
        min_test_loss = test_loss
        torch.save({
            'weights': model.state_dict(),
            'n_items_A': model.n_items_A,
            'n_items_B': model.n_items_B,
            'n_users': model.n_users,
            'graph_matrix': model.graph_matrix,
        }, 'best.pt')