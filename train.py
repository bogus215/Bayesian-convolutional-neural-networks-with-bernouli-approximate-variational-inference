#%%
import warnings
warnings.filterwarnings("ignore")
from loader import *
import argparse
from Lenet import *
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import gc
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random


# %% Train
def train(args):

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    writer = SummaryWriter(f'./runs/{args.experiment}')
    print(f'start : {args.experiment}')
    optimizer = optim.SGD(args.model.parameters(), args.learning_rate , momentum=0.9,
                          weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    test_acc = []
    test_error = []
    best_ = -np.inf


    iter = 0
    for e in range(args.epoch):
        print("\n===> epoch %d" % e)
        running_loss = 0
        args.model.train()
        for i, data in enumerate(tqdm(args.loader.train_iter, desc='train')):
            iter += 1
            inputs, labels = data
            inputs = inputs.cuda(args.gpu_device)
            labels = labels.cuda(args.gpu_device)
            optimizer.zero_grad()
            outputs = F.softmax(args.model(inputs) , dim = 1 )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            optimizer.param_groups[0]['lr'] = args.learning_rate * np.power( 1 + 0.0001 * iter, -0.75)

        writer.add_scalar('train_loss',running_loss / len(args.loader.train_iter) , e + 1)
        print('Epoch {} loss: {}'.format(e+1,running_loss / len(args.loader.train_iter)))


        with torch.no_grad():

            args.model.eval()
            correct = 0
            total = 0
            for s, val_batch in enumerate(tqdm(args.loader.test_iter, desc='test')):

                feature = val_batch[0].cuda(args.gpu_device)
                target = val_batch[1].cuda(args.gpu_device)
                if args.experiment.split('_')[0] == 'Bayes':
                    temp = [F.softmax(args.model(feature),dim=1).view(1,feature.size()[0],10) for _ in range(args.T)]
                    pred = torch.cat(temp).mean(axis=0)
                else:
                    pred = F.softmax(args.model(feature), dim=1)

                _, predicted = torch.max(pred.data,1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            acc = (100 * correct) / total
            torch.save(args.model.state_dict(), f'./parameter/last_parameter_{args.experiment}.pth')
            if acc > best_:
                best_ = acc
                torch.save(args.model.state_dict() , f'./parameter/best_parameter_{args.experiment}.pth')


            writer.add_scalar('test_acc', acc, e + 1)
            test_acc.append(acc)
            writer.add_scalar('test_error',total-correct, e + 1)
            test_error.append(total-correct)






# %% main
def main():

    parser = argparse.ArgumentParser(description="-----[#]-----")

    # Model
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("--epoch", default=1500, type=int, help="number of max epoch")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="weight dacay")
    parser.add_argument("--conv_drop", default=0.5, type=float, help="conv dropout rate")
    parser.add_argument("--dense_drop", default=0.5, type=float, help="dense dropout rate")



    # Data and train
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset CIFAR or MNIST')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument("--gpu_device", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--printevery', default=100, type=int, help='log , print every % iteration')
    parser.add_argument('--data_size', default=50000, type=int, help='dataset size(n)')
    parser.add_argument('--image_size', default=28, type=int, help='image size')
    parser.add_argument('--image_channel', default=1, type=int, help='image channel')
    parser.add_argument('--T', default=50, type=int, help='stochastic forward')
    parser.add_argument('--experiment', type=str, default='Bayes_', help='experiment name')


    args = parser.parse_args()
    args.loader = c_loader(args)
    print('loader loaded')

    if args.experiment.split('_')[0] == 'Bayes':
        # Define bayes networks
        args.model = DropOutLenet_all(conv_dropout = args.conv_drop , dense_dropout = args.dense_drop,force_dropout = True)
        args.model = args.model.cuda(args.gpu_device)

    else:
        # Define normal network
        args.model = LeNet_all(droprate = 0.5)
        args.model = args.model.cuda(args.gpu_device)

    print('model created -- gpu version!')

    gc.collect()
    train(args)

# %% run
if __name__ == "__main__":
    main()

