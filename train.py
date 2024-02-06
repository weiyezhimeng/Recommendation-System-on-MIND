import torch
from utils import loss
import gc
from GPU import GPU
def train(tokenizer,model,model_bert,device,lr,EPOCH,loader,batch):
    # ========== setup optimizer and scheduler ========== #
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr,weight_decay=0,eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)
    for epoch in range(EPOCH):
        step = 0
        print('Epoch:', epoch + 1, 'Training...')
        for step, (history, label) in enumerate(loader):  # 每一步loader释放一小批数据用来学习
            loss_all=loss(history,label,batch,model,model_bert,tokenizer,device)
            print("loss:", loss_all, "step:", step)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            scheduler.step()
            del  loss_all ; gc.collect();torch.cuda.empty_cache()
            if step==150:
                break
    torch.save(model, 'user.pth')
    torch.save(model_bert, 'bert-news.pth')







