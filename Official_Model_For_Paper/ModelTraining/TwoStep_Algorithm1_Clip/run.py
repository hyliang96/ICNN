import os

# rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['VOCpart']
depths = [20]
gpu_id = '0'
batchsize = 32
epoch = 300
# exp_dir = '/data/ouyangzhihao/Exp/ICNN/LearnableMask/tb_dir/learnable_mask_baseline/Debug'
ifmask = False
optim = 'adam'
exp_dir_root = '/mfs/haoyu/project/CDCNN/ICNN_exp/VOCPart_train0.7_64/'
lr = '0.01'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"



# os.system('rm -r ' + exp_dir)
for data in datasets:
    for depth in depths:
        exp_dir = exp_dir_root+'%sres%d_bs%d_%s_lr%s' % ('naive_' if not ifmask else '', depth, batchsize, optim, lr)
        res = exp_dir + '/res.txt'
        print('run ', exp_dir.split('/')[-1])

        # cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #                         %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        cmd = 'python3 ./train.py --dataset %s --depth %d --res %s --ifmask %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s --lr %s' \
              % (data, depth, res,ifmask, gpu_id, batchsize, epoch, exp_dir, lr)
        os.system(cmd)



