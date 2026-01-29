from __future__ import division
import warnings

from network.thermalcc import ThermalCC

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import dataset
import math
from image import *
from utils import *

import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import time

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


from torch.nn import Module

class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32).cuda() + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list


class Bay_Loss(Module):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32).cuda()
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32).cuda()
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)
        return loss
        
        
        

def train_collate(batch):
    transposed_batch = list(zip(*batch))

    fnames = transposed_batch[0]

    images = torch.stack(transposed_batch[1], 0)

    fidt_maps = torch.stack(transposed_batch[2], 0)

    kpoints = torch.stack(transposed_batch[3], 0)

    keypoints = transposed_batch[4]

    targets = transposed_batch[5]
    
    st_sizes = torch.FloatTensor(transposed_batch[6])

    return fnames, images, fidt_maps, kpoints, keypoints, targets, st_sizes



def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'
    elif args['dataset'] == 'dronergbt':
        train_file = './npydata/dronergbt_train.npy'
        test_file = './npydata/dronergbt_test.npy'
    elif args['dataset'] == 'rgbtcc':
        train_file = './npydata/rgbtcc_train.npy'
        test_file = './npydata/rgbtcc_test.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    model = ThermalCC()
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    optimizer = torch.optim.Adam(
        [  #
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    criterion = nn.MSELoss(size_average=False).cuda()


    print(args['pre'])

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])
    print(args['best_pred'], args['start_epoch'])

    if args['preload_data'] == True:
        train_data = pre_data(train_list, args, train=True)
        test_data = pre_data(test_list, args, train=False)
    else:
        train_data = train_list
        test_data = test_list

    for epoch in range(args['start_epoch'], args['epochs']):

        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args)
        end1 = time.time()

        '''inference '''
        # if epoch % 1 == 0 and epoch >= 10:
        if epoch >= 10:
            prec1, visi = validate(test_data, model, args)

            end2 = time.time()

            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])
            

            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, visi, is_best, args['save_path'])

def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        # print(fname)
        img, fidt_map, kpoint = load_data_fidt(Img_path, args, train)

        if min(fidt_map.shape[0], fidt_map.shape[1]) < 256 and train == True:
            # ignore some small resolution images
            continue
        # print(img.size, fidt_map.shape)
        blob = {}
        blob['img'] = img
        blob['kpoint'] = np.array(kpoint)
        blob['fidt_map'] = fidt_map
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

    return data_keys

criterion2 = Bay_Loss(True, device=None)
post_prob = Post_Prob(sigma=8,
                      c_size=256,
                      stride=8,
                      background_ratio=1.0,
                      use_background=True,
                      device=None,)
                       
def train(Pre_data, model, criterion, optimizer, epoch, args):
    count_loss = nn.L1Loss().cuda()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),

                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        collate_fn=train_collate, batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()


    for i, (fname, img, fidt_map, kpoint, keypoints, targets, st_sizes) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()

        fidt_map = fidt_map.type(torch.FloatTensor).unsqueeze(1).cuda()
        
        keypoints = [p.cuda() for p in keypoints]
        targets = [t.cuda() for t in targets]
        st_sizes = st_sizes.cuda()
            
            
        out1, outputs = model(img, epoch/1000)
        
        prob_list = post_prob(keypoints, st_sizes)
        loss2 = criterion2(prob_list, targets, outputs)
        
        # =====
        target_count = torch.tensor([len(p) for p in keypoints], dtype=torch.float32).cuda()
        
        
        pred_density1 = get_count(out1)
        pred_count1 = pred_density1.view(pred_density1.shape[0], -1).sum(dim=1)
        c_loss1 = count_loss(pred_count1, target_count)
        
        pred_count2 = outputs.view(outputs.shape[0], -1).sum(dim=1)
        c_loss2 = count_loss(pred_count2, target_count)
        
        
        if out1.shape != fidt_map.shape:
            print("the shape is wrong, please check. Both of prediction and GT should be [B, C, H, W].")
            exit()
        loss1 = criterion(out1, fidt_map)
        
        loss = loss2 + loss1 + c_loss1 + c_loss2
        
        losses.update(loss.item(), img.size(0))
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t [info]:loss1={loss1:.4f}, loss2={loss2:.4f}, c_loss1={c_loss1:.4f}, c_loss2={c_loss2:.4f},'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss1=loss1, loss2=loss2, c_loss1=c_loss1, c_loss2=c_loss2))



def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    
    mae2 = 0
    mse2 = 0
    
    visi = []
    index = 0

    if not os.path.exists('./local_eval/loc_file'):
        os.makedirs('./local_eval/loc_file')

    '''output coordinates'''
    f_loc = open("./local_eval/A_localization.txt", "w+")

    for i, (fname, img, fidt_map, kpoint) in enumerate(test_loader):
        if img.shape == torch.Size([1, 3, 640, 480]):
            continue
        count = 0
        img = img.cuda()

        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(fidt_map.shape) == 5:
            fidt_map = fidt_map.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(fidt_map.shape) == 3:
            fidt_map = fidt_map.unsqueeze(0)

        with torch.no_grad():
            out1, outputs = model(img)
            

            '''return counting and coordinates'''
            count, pred_kpoint, f_loc = LMDS_counting(out1, i + 1, f_loc, args)
            point_map = generate_point_map(pred_kpoint, f_loc, rate=1)

            if args['visual'] == True:
                if not os.path.exists(args['save_path'] + '_box/'):
                    os.makedirs(args['save_path'] + '_box/')
                ori_img, box_img = generate_bounding_boxes(pred_kpoint, fname)
                show_fidt = show_map(out1.data.cpu().numpy())
                gt_show = show_map(fidt_map.data.cpu().numpy())
                res = np.hstack((ori_img, gt_show, show_fidt, point_map, box_img))
                cv2.imwrite(args['save_path'] + '_box/' + fname[0], res)

            
            count2 = torch.sum(outputs).item()

        gt_count = torch.sum(kpoint).item()
        
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)
        
        """
        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}:: {pred2}'.format(fname=fname[0], gt=gt_count, pred=count, pred2=count2))
            visi.append(
                [img.data.cpu().numpy(), d6.data.cpu().numpy(), fidt_map.data.cpu().numpy(),
                 fname])
            index += 1
        """

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)
    
    nni.report_intermediate_result(mae)
    print('\n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae, visi


def get_count(input):
    input_max = torch.max(input).item()


    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        input = input * 0

    # count = int(torch.sum(input).item())


    return input
    

def LMDS_counting(input, w_fname, f_loc, args):
    input_max = torch.max(input).item()

    ''' find local maxima'''
    if args['dataset'] == 'UCF_QNRF' :
        input = nn.functional.avg_pool2d(input, (3, 3), stride=1, padding=1)
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    else:
        keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    '''set the pixel valur of local maxima as 1 for counting'''
    input[input < 100.0 / 255.0 * input_max] = 0
    input[input > 0] = 1

    ''' negative sample'''
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    f_loc.write('{} {} '.format(w_fname, count))
    return count, kpoint, f_loc


def generate_point_map(kpoint, f_loc, rate=1):
    '''obtain the location coordinates'''
    pred_coor = np.nonzero(kpoint)

    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)

    for data in coord_list:
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))
    f_loc.write('\n')

    return point_map


def generate_bounding_boxes(kpoint, fname):
    '''change the data path'''
    Img_data = cv2.imread(
        '/home/dkliang/projects/synchronous/datasets/ShanghaiTech/part_A_final/test_data/images/' + fname[0])
    ori_Img_data = Img_data.copy()

    '''generate sigma'''
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    distances, locations = tree.query(pts, k=4)
    for index, pt in enumerate(pts):
        pt2d = np.zeros(kpoint.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if np.sum(kpoint) > 1:
            sigma = (distances[index][1] + distances[index][2] + distances[index][3]) * 0.1
        else:
            sigma = np.average(np.array(kpoint.shape)) / 2. / 2.  # case: 1 point
        sigma = min(sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.05)

        if sigma < 6:
            t = 2
        else:
            t = 2
        Img_data = cv2.rectangle(Img_data, (int(pt[0] - sigma), int(pt[1] - sigma)),
                                 (int(pt[0] + sigma), int(pt[1] + sigma)), (0, 255, 0), t)

    return ori_Img_data, Img_data


def show_map(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
