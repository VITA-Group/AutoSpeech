import time
import torch
import torch.nn.functional as F
import logging
import numpy as np
import matplotlib.pyplot as plt

from utils import compute_eer
from utils import AverageMeter, ProgressMeter, accuracy

plt.switch_backend('agg')
logger = logging.getLogger(__name__)


def train(cfg, model, optimizer, train_loader, val_loader, criterion, architect, epoch, writer_dict, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    alpha_entropies = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(
        len(train_loader), batch_time, data_time, losses, top1, top5, alpha_entropies,
        prefix="Epoch: [{}]".format(epoch), logger=logger)
    writer = writer_dict['writer']

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        if lr_scheduler:
            current_lr = lr_scheduler.set_lr(optimizer, global_steps, epoch)
        else:
            current_lr = cfg.TRAIN.LR

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        input_search, target_search = next(iter(val_loader))
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        # step architecture
        architect.step(input_search, target_search)

        alpha_entropy = architect.model.compute_arch_entropy()
        alpha_entropies.update(alpha_entropy.mean(), input.size(0))

        # compute output
        output = model(input)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer.add_scalar('arch_entropy', alpha_entropies.val, global_steps)

        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', top1.val, global_steps)
        writer.add_scalar('train_acc5', top5.val, global_steps)

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)


def train_from_scratch(cfg, model, optimizer, train_loader, criterion, epoch, writer_dict, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), batch_time, data_time, losses, top1, top5, prefix="Epoch: [{}]".format(epoch), logger=logger)
    writer = writer_dict['writer']

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        if lr_scheduler:
            current_lr = lr_scheduler.get_lr()
        else:
            current_lr = cfg.TRAIN.LR

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', top1.val, global_steps)
        writer.add_scalar('train_acc5', top5.val, global_steps)

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)


def validate_verification(cfg, model, test_loader):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(test_loader), batch_time, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()
    labels, distances = [], []

    with torch.no_grad():
        end = time.time()
        for i, (input1, input2, label) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True).squeeze(0)
            input2 = input2.cuda(non_blocking=True).squeeze(0)
            label = label.cuda(non_blocking=True)

            # compute output
            outputs1 = model(input1).mean(dim=0).unsqueeze(0)
            outputs2 = model(input2).mean(dim=0).unsqueeze(0)

            dists = F.cosine_similarity(outputs1, outputs2)
            dists = dists.data.cpu().numpy()
            distances.append(dists)
            labels.append(label.data.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        eer = compute_eer(distances, labels)
        logger.info('Test EER: {:.8f}'.format(np.mean(eer)))

    return eer


def validate_identification(cfg, model, test_loader, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader), batch_time, losses, top1, top5, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda(non_blocking=True).squeeze(0)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=0, keepdim=True)
            output = model.forward_classifier(output)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            loss = criterion(output, target)

            losses.update(loss.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)

        logger.info('Test Acc@1: {:.8f} Acc@5: {:.8f}'.format(top1.avg, top5.avg))

    return top1.avg

