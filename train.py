import argparse
from config import cfg
import torch
from common.base import Trainer
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp


# torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='', help='for resuming train')
    parser.add_argument('--amp', dest='use_mixed_precision', action='store_true',
                        help='use automatic mixed precision training')
    parser.add_argument('--init_scale', type=float, default=1024., help='initial loss scale')
    parser.add_argument('--cfg', type=str, default='', help='experiment configure file name')
    # 以下是第二个模型训练的时候的参数
    parser.add_argument('--with_contrastive', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-4, help='learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--inter_weight', type=float, default=0.1)
    parser.add_argument('--intra_weight', type=float, default=0.1)
    parser.add_argument('--total_steps', type=int, default=1e10)

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set suitable gpu ids!"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train, exp_dir=args.exp_dir)
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    scaler = amp.GradScaler(init_scale=args.init_scale, enabled=args.use_mixed_precision)

    # train
    # 这里在训练第二个模型的时候，使用梯度累计的方式模拟大的batch_size
    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            with amp.autocast(args.use_mixed_precision):
                loss = trainer.model(inputs, targets, meta_info, 'train')
                # 这个是第二个模型的
                intra_nce = loss.pop('intra_nce_0', 0)
                inter_nce = loss.pop('inter_nce_0', 0)

                loss = {k: loss[k].mean() for k in loss}
                # 这个是第二个模型的
                # loss = trainer.awl(loss)
                _loss = sum(loss[k] for k in loss) + intra_nce * cfg.intra_weight + inter_nce * cfg.inter_weight
                # _loss = sum(loss[k] for k in loss)

            # backward
            with amp.autocast(False):
                _loss = scaler.scale(_loss)
                _loss.backward()
                scaler.step(trainer.optimizer)

                # # 更新学习率
                # trainer.set_lr(epoch)

            # 第一个模型要用的 改的时候不能删掉
            scaler.update(args.init_scale)

            trainer.gpu_timer.toc()
            if itr % 20 == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                # 这个是第二个模型的
                screen += ['intra_nce: %.4f' % intra_nce]
                screen += ['inter_nce: %.4f' % inter_nce]

                trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            # 这个是第二个模型的
            # "awl": trainer.awl.state_dict(),
        }, epoch)


if __name__ == "__main__":
    main()
