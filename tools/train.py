import time

from helpers import AverageMeter, ProgressMeter, accuracy

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f', 's')
    data_time = AverageMeter('Data', ':6.3f', 's')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f', '%')
    top3 = AverageMeter('Acc@3', ':6.2f', '%')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top3, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc3 = accuracy(output, target, topk=(1, 3))
        # TODO: fix the size(0) bug
        # losses.update(loss.item(), input.size(0))
        # top1.update(acc1[0], input.size(0))
        # top3.update(acc3[0], input.size(0))
        losses.update(loss.item(), args.batch_size)
        top1.update(acc1[0], args.batch_size)
        top3.update(acc3[0], args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
