import os
import errno
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

from resnet_32x32 import resnet as resnet_32x32
from resnet_64x64 import resnet as resnet_64x64
from resnet_std import resent as resnet_std
import pandas as pd
import argparse
import csv
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau
from dataLoader import DataLoader
from summaries import TensorboardSummary
import torchvision

# parameters setting
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', default='test_model', help='filename to output best model') #save output
parser.add_argument('--dataset', default='cifar-10',help="datasets")
parser.add_argument('--depth', default=20,type=int,help="depth of resnet model")
parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
parser.add_argument('--batch_size', default=64,type=int, help='batch size')
parser.add_argument('--epoch', default=200,type=int, help='epoch')
parser.add_argument('--exp_dir',default='./',help='dir for tensorboard')
parser.add_argument('--res', default='./result.txt', help="file to write best result")
parser.add_argument('--ifmask', default='True', type=str, help="whether use learnable mask (i.e. gate matrix)")
parser.add_argument('--optim', default='adam', type=str, help="optimizer: adam | agd")
parser.add_argument('--lr', default=0.1, type=float, help="learning rate for normal path")
parser.add_argument('--lr_reg', default=1, type=float, help='lr of the loss of regularization path')
parser.add_argument('--img_size', default=32, type=int, help="image size, input 32|64|128")
parser.add_argument('--lambda_reg', default=1e-3, type=float, help='regularization coefficient')
parser.add_argument('--frozen', default='True', type=str, help='freeze the lower layers')


args = parser.parse_args()
args.ifmask=True if args.ifmask == 'True' else False
args.frozen=True if args.frozen=='True' else False
print(args)

if os.path.exists(args.exp_dir):
    print ('Already exist and will continue training')
    # exit()
summary = TensorboardSummary(args.exp_dir)
tb_writer = summary.create_summary()

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()

    #best_model = model.state_dic()
    best_acc = 0.0
    best_train_acc = 0.0
    # Load unfinished model
    unfinished_model_path = os.path.join(args.exp_dir , 'unfinished_model_lastest.pt')
    if(os.path.exists(unfinished_model_path)):
        checkpoint = torch.load(unfinished_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']+1
        loss = checkpoint['loss']
    else:
        epoch = 0

    while epoch < num_epochs:
        epoch_time = time.time()
        print('-'*10)
        print('Epoch {}/{}'.format(epoch,num_epochs-1))

        #each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            ifmask = (epoch % 3 >= 1 and epoch >= 10 and args.ifmask and phase == 'train')
            print('ifmask =', ifmask)

            running_loss = 0.0
            running_corrects = 0.0
            top5_corrects = 0.0

            # change tensor to variable(including some gradient info)
            # use variable.data to get the corresponding tensor
            for iteration, data in enumerate(dataloaders[phase]):
                # ifmask = (iteration % 3 == 2 and args.ifmask and phase=='train')

                #782 batch,batch size= 64
                inputs,labels = data
                # print (inputs.shape)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                #zero the parameter gradients
                optimizer.zero_grad()

                #forward
                if ifmask:
                    outputs, regulization_loss = model(inputs, labels)
                    # print('outloss:',criterion(outputs, labels) * 0.7 , criterion(icnn_outputs, labels), regulization_loss)
                    loss = criterion(outputs, labels) + regulization_loss * args.lambda_reg
                    # loss *= args.multipler
                else:
                    # outputs = model(inputs)
                    outputs = model(inputs)
                    # print('outloss:',criterion(outputs, labels) * 0.7 , criterion(icnn_outputs, labels), regulization_loss)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)


                # _,top5_preds = torch.topk(outputs.data,k=5,dim=1)
                # print ('group loss:',group_loss[0])

                if phase == 'train':
                    loss.backward()
                    if ifmask:
                        optimizer_reg.step()
                    else:
                        optimizer.step()
                    # limit mask to positive values
                    # model.module.mask.data = torch.clamp(model.module.mask, min=0.0)
                    # import ipdb;
                    # ipdb.set_trace()
                    # call clip function in model to normalize the lmask
                    if ifmask:
                        model.module.lmask.clip_lmask()

                y = labels.data
                batch_size = labels.data.shape[0]
                # print(y.resize_(batch_size,1))
                running_loss += loss.item()
                running_corrects += torch.sum(preds == y)
                # top5_corrects += torch.sum(top5_preds == y.resize_(batch_size,1))

            if phase == 'train':
                if ifmask:
                    scheduler_reg.step(loss)
                else:
                    scheduler.step(loss)

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = float(running_corrects) /dataset_sizes[phase]
            # top5_acc = top5_corrects /dataset_sizes[phase]

            print('%s Loss: %.4f top1 Acc:%.4f'%(phase,epoch_loss,epoch_acc))
            if phase == 'train':
                tb_writer.add_scalar('train/total_loss_epoch', epoch_loss, epoch)
                tb_writer.add_scalar('train/acc_epoch', epoch_acc, epoch)
                if best_train_acc < epoch_acc:
                    best_train_acc = epoch_acc
            if phase == 'val':
                tb_writer.add_scalar('val/total_loss_epoch', epoch_loss, epoch)
                tb_writer.add_scalar('val/acc_epoch', epoch_acc, epoch)
                if args.ifmask:
                    mask_density = model.module.lmask.get_density()
                    tb_writer.add_scalar('val/mask_density', mask_density, epoch)
                    print('mask density %.4f' % mask_density)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = model.state_dict()
            cost_time = time.time() - epoch_time
            print('Epoch time cost {:.0f}m {:.0f}s'.format(cost_time // 60, cost_time % 60))
        # Save model periotically

        if (epoch % 5 == 0):
            checkpoint_path = os.path.join(args.exp_dir, 'unfinished_model_%d.pt' % epoch)
            lastest_checkpoint_path = os.path.join(args.exp_dir , 'unfinished_model_lastest.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            symlink_force(checkpoint_path, lastest_checkpoint_path)
        epoch += 1

    cost_time = time.time() - since
    print ('Training complete in {:.0f}m {:.0f}s'.format(cost_time//60,cost_time%60))
    print ('Best Train Acc is {:.4f}'.format(best_train_acc))
    print ('Best Val Acc is {:.4f}'.format(best_acc))
    model.load_state_dict(best_model)
    return model,cost_time,best_acc,best_train_acc


if __name__ == '__main__':
    print ('DataSets: '+args.dataset)
    print ('ResNet Depth: '+str(args.depth))
    loader = DataLoader(args.dataset,batch_size=args.batch_size)
    dataloaders,dataset_sizes = loader.load_data(args.img_size)
    num_classes = 10
    if args.dataset == 'cifar-10':
        num_classes = 10
    if args.dataset == 'cifar-100':
        num_classes = 100
    if args.dataset == 'VOCpart':
        num_classes = len(dataloaders['train'].dataset.classes)

    # model = torchvision.models.resnet152(pretrained=True)

    # if args.img_size == 64:
        # model = resnet_64x64(depth=args.depth, num_classes=num_classes, ifmask=args.ifmask)
    # elif args.img_size == 32:
        # model = resnet_32x32(depth=args.depth, num_classes=num_classes, ifmask=args.ifmask)
    print('args.ifmask =', args.ifmask)
    model = resnet_std(depth=args.depth, num_classes=num_classes, ifmask=args.ifmask, pretrained=True)
    # model = resnet_std(depth=args.depth, num_classes=num_classes, ifmask=False, pretrained=True)

    # frozeen lower layers
    if args.frozen:
        def freeze(unfrozen_layer_num):
            n_layer = len(list(model.children()))
            # print(n_layer)
            for idx, (layer_name, layer) in  enumerate( model.named_children() ):
                print(idx, layer_name)
                if idx < n_layer - unfrozen_layer_num : # 4
                    # print('frozeen', idx)
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
                        # print(name)
            # return list(filter(lambda p: p.requires_grad, model.parameters()))

        def select_param(top_n_layer):
            selected_param_names = []
            n_layer = len(list(model.children()))
            for idx, (layer_name, layer) in enumerate(model.named_children()):
                if idx >= n_layer - top_n_layer:  # 4
                    for param_name, param in layer.named_parameters():
                        selected_param_names.append(layer_name + '.' + param_name)
            return [param for name_param, param in model.named_parameters() if name_param in selected_param_names]

        # for name, param in model.named_parameters():
        #     print(name)

        train_params = select_param(5)
        train_params_reg = select_param(4)
        freeze(5)
    else:
        train_params = list(model.parameters())
        train_params_reg = list(model.parameters())

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        optimizer_reg = torch.optim.SGD(train_params_reg, lr=args.lr_reg, momentum=0.9, nesterov=True, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer_reg = torch.optim.Adam(train_params_reg, lr=args.lr_reg, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    else:
        raise

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[args.epoch*0.4, args.epoch*0.6, args.epoch*0.8], gamma=0.1)
    scheduler_reg = MultiStepLR(optimizer_reg, milestones=[args.epoch*0.4, args.epoch*0.6, args.epoch*0.8], gamma=0.1)

    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=10, cooldown=10, verbose=True)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        model = torch.nn.DataParallel(model) # device_ids=args.gpu_ids
        # patch_replication_callback(model)
        model = model.cuda()
    model,cost_time,best_acc,best_train_acc = train_model(model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            scheduler=scheduler,
                                            num_epochs=args.epoch)

    exp_name = 'resnet%d dataset: %s batchsize: %d epoch: %d bestValAcc: %.4f bestTrainAcc: %.4f \n' % (
    args.depth, args.dataset,args.batch_size, args.epoch,best_acc,best_train_acc)

    # os.system('rm ' + os.path.join(args.exp_dir , 'unfinished_model.pt') )
    torch.save(model.state_dict(), os.path.join(args.exp_dir , 'saved_model.pt'))
    with open(args.res,'a') as f:
        f.write(exp_name)
        f.close()

