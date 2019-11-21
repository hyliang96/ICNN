import os

# rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['VOCpart']
depths = [152]
gpu_id = '0'
batchsize = 32
epoch = 300
# exp_dir = '/data/ouyangzhihao/Exp/ICNN/LearnableMask/tb_dir/learnable_mask_baseline/Debug'
optim = 'adam'
lr = '1e-5'  # finetune resnet152: 1e-5
lr_reg = '4.5e-4'
img_size = 128
lambda_reg = '1e-3' # reg. coef.
frozen = 'True'
ifmask = 'True'
train = 'False'

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

exp_dir_root = '/home/haoyu/mfs/project/CDCNN/ICNN_exp/VOCPart_train0.7_%d_pretrained/'%img_size


# os.system('rm -r ' + exp_dir)
for data in datasets:
    for depth in depths:
        exp_dir = exp_dir_root + '%sres%d_bs%d_%s_lr%s_lrreg%s_lmd%s_%s_testscore4' % ('naive_' if not ifmask == 'True' else '', depth, batchsize, optim, lr, lr_reg, lambda_reg,
           'frozen' if frozen == 'True' else ''
        )
        # model_name = 'naive_res152_bs32_adam_lr1e-5_lrreg1e-3_lmd1e-3_frozen_3:2'
        # model_name = 'res152_bs32_adam_lr1e-5_lrreg1e-3_lmd1e-3_frozen_frz5:4_pre10-3:2'
        # exp_dir = exp_dir_root+'/'+model_name
        res = exp_dir + '/res.txt'
        print('run ', exp_dir.split('/')[-1])

        # cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #                         %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        cmd = 'python3 ./train.py --dataset %s --depth %d --res %s --ifmask %s --gpu-ids %s --optim %s \
                --batch_size %d --epoch %d --exp_dir %s --lr %s --img_size %d --lambda_reg %s --frozen %s --lr_reg %s --train %s' \
              % (data, depth, res,ifmask, gpu_id, optim, batchsize, epoch, exp_dir, lr ,img_size, lambda_reg, frozen, lr_reg, train)
        os.system(cmd)




