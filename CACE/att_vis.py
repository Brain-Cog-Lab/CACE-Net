import os
import time
import random
import json
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import h5py
import numpy as np
from configs.opts import parser
from model.main_model import supv_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset import AVEDataset
import torch.nn.functional as F
import cv2
# utils variable
global args, logger, writer, dataset_configs
# configs
dataset_configs = get_and_save_args(parser)
parser.set_defaults(**dataset_configs)
args = parser.parse_args()

# =================================  seed config ============================
SEED = args.seed
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_path = 'configs/main.json'
with open(config_path) as fp:
    config = json.load(fp)
print(config)
# =============================================================================

def video_frame_sample(frame_interval, video_length, sample_num):

    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num

def normlize(x, min = 0, max = 255):

    num, row, col = x.shape
    for i in range(num):
        xi = x[i, :, :]
        xi = max *(xi - np.min(xi))/(np.max(xi) - np.min(xi))
        x[i, :, :] = xi
    return x

def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.3):
    print(np.max(im_cloud))

    im_cloud[:, :, 1] = 0
    im_cloud[:, :, 2] = 0
    return (a1*im_map + a2*im_cloud).astype(np.uint8)

def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.snapshot_pref = args.snapshot_pref + "Seed{}".format(args.seed) + "_guide_{}".format(args.guide) + "_psai{}".format(args.psai) + "_Contrastive_{}".format(args.contrastive) + "_guide_{}".format(args.guide)  + "_eta_{}/".format(args.eta)
    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    '''Dataset'''
    train_dataloader = DataLoader(
        AVEDataset('/home/hexiang/EAVA/data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset('/home/hexiang/EAVA/data/', split='test'),
        batch_size=4020,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    '''model setting'''
    mainModel = main_model(config['model'], psai=args.psai, guide=args.guide, contrastive=args.contrastive, eta=args.eta)
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        mainModel.module.contrastive_switch = False
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    best_models = []
    '''Training and Testing'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',
                    epoch=epoch + 1,
                    best_models=best_models
                )
            logger.info("-----------------------------")
            logger.info("best acc and epoch: {} {}".format(best_accuracy, best_accuracy_epoch))
            logger.info("-----------------------------")
        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.
    model.double()
    optimizer.zero_grad()

    if epoch >= args.n_epoch / 8:
        model.module.contrastive_switch = True
    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # For a model in a float precision
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        is_event_scores, event_scores, audio_visual_gate, av_score, q, k = model(visual_feature, audio_feature)
        # is_event_scores, event_scores = model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()
        audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze().contiguous()

        labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        # _, labels_CAS = labels.max(-1)

        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        label_is_gate = criterion(audio_visual_gate, labels_BCE.cuda())
        loss_cas = criterion_event(av_score, labels_event.cuda())

        # 计算对比损失
        N, T, C = q.shape
        scores_event_ind = (is_event_scores.sigmoid() > 0.5).float()  # 注意labels_BCE与scores_event_ind的选择

        # 定义正样本: q和k中所有为event的样本, 都是正样本对
        mask = torch.ones(T, T).cuda() - torch.eye(T, T).cuda()  # [T, T]
        mask_expand = mask.unsqueeze(0).repeat(N, 1, 1)  # [N, T, T]
        mask_final = torch.ones(2 * T, 2 * T).cuda() - torch.eye(2 * T, 2 * T).cuda()  # [2T, 2T]
        mask_final = mask_final.unsqueeze(0).repeat(N, 1, 1)  # [N, 2T, 2T]

        contrast_feature = torch.cat([q, k], dim=1)  # [N, 2*T, C]
        logits = torch.einsum("ntc,nck->ntk", [contrast_feature, contrast_feature.permute(0, 2, 1)])  # [N, 2*T, 2*T]

        # apply temperature
        logits /= 0.01

        scores_pos_ind_q = labels_BCE.unsqueeze(-1) * labels_BCE.unsqueeze(1) * mask_expand # [N, T, T]
        scores_pos_ind_k = labels_BCE.unsqueeze(-1) * labels_BCE.unsqueeze(1) + torch.eye(T).cuda()  # [N, T, T]
        scores_pos_ind_k = torch.clamp(scores_pos_ind_k, 0, 1)
        scores_pos_ind = torch.zeros(N, 2*T, 2*T).cuda()
        scores_pos_ind[:, :T, :T] = scores_pos_ind_q
        scores_pos_ind[:, :T, T:2*T] = scores_pos_ind_k
        scores_pos_ind[:, T:2*T, :T] = scores_pos_ind_k
        scores_pos_ind[:, T:2*T, T:2*T] = scores_pos_ind_q

        num_pos_pairs = labels_BCE.sum(1).unsqueeze(-1)

        index = torch.log(torch.cat([labels_BCE * (2*num_pos_pairs - 2), labels_BCE * (2*num_pos_pairs - 2)], dim=1) + torch.ones(1, 2*T).cuda())
        log_prob_k = torch.log((torch.exp(logits) * scores_pos_ind.float()).sum(dim=2)) - torch.log((torch.exp(logits) * mask_final.float()).sum(dim=2)) - index
        log_prob_k = -log_prob_k
        # labels: positive key indicators
        loss_contrastive = log_prob_k.mean()

        loss = loss_is_event + loss_event_class + label_is_gate + loss_cas

        if args.contrastive:
            if epoch >= (args.n_epoch / 8):
                loss += loss_contrastive
            # binary_acc = (scores_event_ind == labels_BCE).sum()/ scores_event_ind.numel()
            # # print("binary_acc:{}".format(binary_acc))
            # if binary_acc >= 0.8:
            #     loss += loss_contrastive

        loss.backward()

        '''Compute Accuracy'''
        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg

# generate attention maps
att_map = torch.zeros((4020, 49, 1))
tanh_o = nn.Tanh()

def fun(m, i, o):
    att_map.copy_(tanh_o(o.data))  # output shape [640, 49, 1]

@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # For a model in a float type
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        bs = visual_feature.size(0)

        dir_order_test = "/home/hexiang/EAVA/data/test_order.h5"

        # access to original videos for extracting video frames
        raw_video_dir = "/home/hexiang/Encoders/data/AVE_Dataset/AVE/"  # videos in AVE dataset
        lis = os.listdir(raw_video_dir)
        f = open("/home/hexiang/EAVA/data/Annotations.txt", 'r')
        dataset = f.readlines()
        with h5py.File(dir_order_test, 'r') as hf:
            test_order = hf['order'][:]

        att_layer = model._modules['module'].spatial_channel_att.affine_v_s_att  # extract attention maps from the layer

        map = att_layer.register_forward_hook(fun)

        is_event_scores, event_scores, audio_visual_gate, _, _, _ = model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze()
        audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze()

        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_is_gate = criterion(audio_visual_gate, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = loss_is_event + loss_event_class + loss_is_gate

        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        print("acc:{}".format(acc))
        map.remove()
        z_t = att_map.squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))
        att_weight = alpha_t.view(402, 10, 7, 7).cpu().data.numpy()  # attention maps of all testing samples

        c = 0
        t = 10
        sample_num = 16  # 16 frames for 1-sec video segment
        extract_frames = np.zeros((160, 224, 224, 3))  # 160 224x224x3 frames for a 10-sec video

        method = "ours"
        if method != "ours":
            save_dir = "visual_att_baseline/attention_maps/"  # store attention maps
            original_dir = "visual_att_baseline/original/"  # store video frames
        else:
            save_dir = "visual_att_ours/attention_maps_CMGA/"  # store attention maps
            original_dir = "visual_att_ours/original/"  # store video frames

        for num in range(len(test_order)):
            print(num)
            data = dataset[test_order[num]]
            x = data.split("&")

            # extract video frames
            video_index = os.path.join(raw_video_dir, x[1] + '.mp4')
            vid = imageio.get_reader(video_index, 'ffmpeg')
            vid_len = sum(1 for _ in vid)
            frame_interval = int(vid_len / t)

            frame_num = video_frame_sample(frame_interval, t, sample_num)
            imgs = []
            for i, im in enumerate(vid):
                x_im = cv2.resize(im, (224, 224))
                imgs.append(x_im)
            vid.close()
            cc = 0
            for n in frame_num:
                extract_frames[cc, :, :, :] = imgs[n]
                cc += 1

            # process generated attention maps
            att = att_weight[num, :, :, :]
            att = normlize(att, 0, 255)
            att_scaled = np.zeros((10, 224, 224))
            for k in range(att.shape[0]):
                att_scaled[k, :, :] = cv2.resize(att[k, :, :], (224, 224))  # scaling attention maps

            att_t = np.repeat(att_scaled, 16,
                              axis=0)  # 1-sec segment only has 1 attention map. Here, repeat 16 times to generate 16 maps for a 1-sec video
            heat_maps = np.repeat(att_t.reshape(160, 224, 224, 1), 3, axis=-1)
            c += 1

            att_dir = save_dir + x[1]
            ori_dir = original_dir + x[1]
            if not os.path.exists(att_dir):
                os.makedirs(att_dir)
            if not os.path.exists(ori_dir):
                os.makedirs(ori_dir)
            for idx in range(160):
                heat_map = heat_maps[idx, :, :, 0]
                im = extract_frames[idx, :, :, :]
                im = im[:, :, (2, 1, 0)]
                heatmap = cv2.applyColorMap(np.uint8(heat_map), cv2.COLORMAP_JET)

                att_frame = heatmap * 0.2 + np.uint8(im) * 0.6
                n = "%04d" % idx
                vid_index = os.path.join(att_dir, 'pic' + n + '.jpg')
                cv2.imwrite(vid_index, att_frame)
                ori_frame = np.uint8(im)
                # ori_index = os.path.join(ori_dir, 'ori' + n + '.jpg')
                # cv2.imwrite(ori_index, ori_frame)


        accuracy.update(acc.item(), bs * 10)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    '''Add loss in an epoch to Tensorboard'''
    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    # labels = labels[:, :, :-1]  # 28 denote background
    _, targets = labels.max(-1)
    # pos pred
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1)  # foreground classification
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]
    # add mask
    pred[scores_mask] = 28  # 28 denotes bg
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc


def save_checkpoint(state_dict, top1, task, epoch, best_models):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model_psai_{args.psai}_eta_{args.eta}.pth.tar'
    torch.save(state_dict, model_name)
    best_models.append((top1, model_name))
    best_models.sort(key=lambda x: x[0], reverse=True)  # 按准确率降序排序

    # 如果保存的模型超过1个，则删除准确率最低的模型
    while len(best_models) > 1:
        _, oldest_model_path = best_models.pop()  # 获取准确率最低的模型
        os.remove(oldest_model_path)  # 删除该模型文件


if __name__ == '__main__':
    main()
