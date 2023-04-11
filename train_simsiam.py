from SimSiam.simsiam.datapreparation import *
from SimSiam.simsiam.simsiam import *
from SimSiam.simsiam.utils import *
from SimSiam.simsiam.evaluation import *
from SimSiam.simsiam.monitoring import *
import argparse
import torch
from tqdm import tqdm


if __name__=="__main__":
    # python3 train_simsiam.py --epochs 1 --lr 0.03 --momentum 0.9 --weight_decay 0.0005 --output_path 'test.pth'

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize for dataloader')
    parser.add_argument('--lr', type=float, default=0.03, help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay')
    parser.add_argument('--output_path', type=str, default='simsiam.pth')

    opt = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SimSiam().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    train_loader, memory_loader, test_loader = prepare_data(batch_size=opt.batch_size)

    accuracy = 0 
    # Start training
    global_progress = tqdm(range(0, opt.epochs), desc=f'Training')
    for epoch in global_progress:
        model.train()
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{opt.epochs}')
        for idx, data in enumerate(local_progress):
            images = data[0]
            optimizer.zero_grad()
            data_dict = model.forward(images[0].to(device, non_blocking=True), images[1].to(device, non_blocking=True))
            loss = data_dict['loss'].mean()

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            
            local_progress.set_postfix(data_dict)

        #if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
        accuracy = knn_monitor(model.encoder, memory_loader, test_loader, device, k=min(25, len(memory_loader.dataset))) 
        
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)

    PATH = opt.output_path
    torch.save(model.state_dict(), PATH)