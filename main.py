import argparse
import datetime
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from lib.datasets.eshg_dataset import ESHGDataset
from lib.models.blind_deep_gestalt import BlindDeepGestalt
from lib.models.deep_gestalt import DeepGestalt

saved_model_dir = "saved_models"


# Function that helps set each work's seed differently (consistently)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Bone Age Test')
    parser.add_argument('--batch-size', type=int, default=280, metavar='N',
                        help='input batch size for training (default: 4)')

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',  # lr=1e-3
                        help='learning rate (default: 0.005)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')

    parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val-interval', type=int, default=100000,
                        help='how many batches to wait before validation is evaluated (and optimizer is stepped).')

    parser.add_argument('--session', type=int, dest='session',
                        help='Session used to distinguish model tests.')
    parser.add_argument('--model-type', default='BlindDeepGestalt', dest='model_type',
                        help='Model type to use. (Options: \'DeepGestalt\', \'BlindDeepGestalt\')')

    parser.add_argument('--in_channels', default=1, dest='in_channels',
                        help='Number of color channels of the images used as input (default: 1)')
    parser.add_argument('--num_classes', default=2, dest='num_classes', type=int)  # 10575 for casia
    parser.add_argument('--alpha', default=1.0, dest='alpha')
    parser.add_argument('--dataset', default='eshg', dest='dataset',
                        help='Which dataset to use. (Options: "eshg")')

    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use tensorboard for logging')

    return parser.parse_args()


def train(args, model, device, train_loader, sec_loader, optimizers, epochs=-1, val_loader=None, scheduler=None):
    model.train()

    # Alpha value used to decide the weight of the confusion loss
    alpha = args.alpha

    # Time measurements
    tick = datetime.datetime.now()

    # Tensorboard Writer
    if args.use_tensorboard:
        writer = SummaryWriter(
            comment=f"s{args.session}_{args.model_type}_bs{args.batch_size}")
    global_step = 0

    if epochs == -1:
        epochs = args.epochs

    # First time we train the secondary classifier for a set amount of epochs
    first = True

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.

        # Only run secondary classifier when using BlindDeepGestalt
        if isinstance(model, BlindDeepGestalt):
            # Freeze the representation layer; it should only update during the primary-part
            model.freeze_repr_layer(True)
            model.freeze_cls_layer('primary', True)
            model.freeze_cls_layer('secondary', False)

            # Train the secondary classifier
            best_loss = 100.
            epochs_worse_loss = 0
            for s_epoch in range(1, 100 + 1):
                # last_loss = 0.
                for batch_idx, (data, _, target_secondary) in enumerate(sec_loader):
                    data = data.to(device, dtype=torch.float32)
                    target_secondary = target_secondary.to(device, dtype=torch.int64).unsqueeze(1)

                    _, pred_s, _ = model(data)
                    loss_s = F.cross_entropy(pred_s, target_secondary.view(-1), weight=args.ce_weights_s)
                    loss_s.backward()

                    last_loss_secondary = loss_s.item()

                    # Clipping gradients here, if we get exploding gradients we should uncomment:
                    # nn.utils.clip_grad_value_(model.parameters(), 0.1)

                    optimizers[1].step()
                    optimizers[1].zero_grad()

                # Early stopping when loss doesn't decrease for a while
                if last_loss_secondary < best_loss:
                    best_loss = last_loss_secondary
                    epochs_worse_loss = 0
                else:
                    epochs_worse_loss += 1

                if epochs_worse_loss > 10 and not first:
                    print(f"Stopped secondary classifier training early at epoch {s_epoch}")
                    break
            first = False

            # Plot the secondary classifier performance on the validation set
            _, t_acc = validate(model, device, val_loader, args, secondary=True)
            if args.use_tensorboard:
                writer.add_scalar('Val/top_acc_secondary', t_acc, global_step)
                writer.add_scalar('Train/ce_loss_secondary', last_loss_secondary, global_step)

            # unfreeze the representation layer; it should update during this primary-part
            model.freeze_repr_layer(False)
            model.freeze_cls_layer('primary', False)
            model.freeze_cls_layer('secondary', True)

        # Train the primary and secondary classifiers, as well as the representation layer
        for batch_idx, (data, target_primary, _) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float32)
            target_primary = target_primary.to(device, dtype=torch.int64).unsqueeze(1)
            # target_secondary = target_secondary.to(device, dtype=torch.int64).unsqueeze(1)

            loss_s_conf = 0.
            if isinstance(model, BlindDeepGestalt):
                pred_p, pred_s, _ = model(data)

                # Confusion loss
                confusion_tensor = torch.FloatTensor(pred_s.size()).uniform_(0, 1).to(device)
                loss_s_conf = - (torch.sum(confusion_tensor * torch.log(F.softmax(pred_s, dim=1)))) / float(
                    F.softmax(pred_s, dim=1).size(0))
            elif isinstance(model, DeepGestalt):
                pred_p, _ = model(data)
            else:
                raise NotImplementedError

            # Classification loss
            loss_p = F.cross_entropy(pred_p, target_primary.view(-1), weight=args.ce_weights_p)

            loss = loss_p + alpha * loss_s_conf
            loss.backward()

            # Clipping gradients here, if we get exploding gradients we should uncomment
            # nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizers[0].step()
            optimizers[0].zero_grad()

            epoch_loss += loss.item()

        if (batch_idx + 1) % args.log_interval == 0:
            tock = datetime.datetime.now()
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(Elapsed time {:.1f}s)'.format(
                tock.strftime("%H:%M:%S"), epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                                                  100. * batch_idx / len(train_loader), loss.item(),
                (tock - tick).total_seconds()))
            tick = tock

            if args.use_tensorboard:
                writer.add_scalar('Train/ce_loss', loss.item(), global_step)

        if val_loader:
            if (batch_idx + 1) % args.val_interval == 0:
                avg_val_loss, t_acc = validate(model, device, val_loader, args)

                tick = datetime.datetime.now()

                if args.use_tensorboard:
                    writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
                    writer.add_scalar('Val/top_acc', t_acc, global_step)

        global_step += args.batch_size

        # Epoch is completed
        print(f"Overall average training loss: {epoch_loss / len(train_loader):.6f}")
        if args.use_tensorboard:
            writer.add_scalar('Train/ce_loss', epoch_loss / len(train_loader), global_step)

        if scheduler:
            scheduler.step()

        # Save model
        print(
            f"Saving model in: {saved_model_dir}/s{args.session}_{args.dataset}_adam_{args.model_type}_e{epoch}"
            f"_bs{args.batch_size}.pt")
        torch.save(model.state_dict(),
                   f"{saved_model_dir}/s{args.session}_{args.dataset}_adam_{args.model_type}_e{epoch}"
                   f"_bs{args.batch_size}.pt")

        # Plot the performance on the validation set
        avg_val_loss, t_acc = validate(model, device, val_loader, args)
        if args.use_tensorboard:
            writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
            writer.add_scalar('Val/top_acc', t_acc, global_step)

    if args.use_tensorboard:
        writer.flush()
        writer.close()


def validate(model, device, val_loader, args, out=False, secondary=False):
    model.eval()
    val_ce_loss = 0.
    top_acc = 0.

    tick = datetime.datetime.now()
    with torch.no_grad():
        for idx, (data, target_primary, target_secondary) in enumerate(val_loader):
            data = data.to(device, dtype=torch.float32)
            if not secondary:
                target_primary = target_primary.to(device, dtype=torch.int64).unsqueeze(1)
                target_secondary = target_secondary.to(device, dtype=torch.int64).unsqueeze(1)
            else:
                target_secondary, target_primary = (target_primary.to(device, dtype=torch.int64).unsqueeze(1),
                                                    target_secondary.to(device, dtype=torch.int64).unsqueeze(1))

            if isinstance(model, BlindDeepGestalt):
                pred_p, pred_s, pred_rep = model(data)
            elif isinstance(model, DeepGestalt):
                pred_p, pred_rep = model(data)
            else:
                raise NotImplementedError

            if not secondary:
                pred = pred_p
            else:
                pred = pred_s
            val_ce_loss += F.cross_entropy(pred, target_primary.view(-1)).item()

            # Used after training to save the representations to a file
            if out:
                # save the representations to txt file
                file = open(f"s{args.session}_{args.model_type}_{args.act_type}_bs{args.batch_size}.txt", "w+")
                for i in range(args.val_bs):
                    if secondary:
                        print(f"{target_secondary[i].item()},{target_primary[i].item()},{pred[i].tolist()}")
                    else:
                        print(f"{target_primary[i].item()},{target_secondary[i].item()},{pred[i].tolist()}")

                    line = f"{target_primary[i].item()},{target_secondary[i].item()},{pred_rep[i].tolist()}"
                    file.write(line)
                file.close()

            # extra stats
            _, max_idx = torch.max(pred, dim=-1)
            top_acc += torch.sum((target_primary.view(-1) == max_idx)).item()

            # Dirty, but easy, way to get the accuracy per class-pair ...
            counts = np.zeros(4)
            if secondary:
                indices = [int((p << 1) + s) for (p, s) in zip(target_secondary.view(-1), target_primary.view(-1))]
            else:
                indices = [int((p << 1) + s) for (p, s) in zip(target_primary.view(-1), target_secondary.view(-1))]

            for i, val in enumerate((target_primary.view(-1) == max_idx)):
                if val:
                    counts[indices[i]] += 1

            # magic number: 25 is the number of samples per class-pairs
            counts = counts / 25
            print(
                f"\tAccuracy on {'Primary' if not secondary else 'Secondary'} per class-pair: {counts} (ff, ft, tf, tt)")
            del counts
            del indices

    top_acc = torch.true_divide(top_acc, len(val_loader) * args.val_bs).item()

    model.train()

    print(f"Average BCE Loss ({val_ce_loss / len(val_loader)}) during validation")
    print(f"\tAverage accuracy on {'Primary' if not secondary else 'Secondary'}: {top_acc}")
    print(f"Elapsed time during validation: {(datetime.datetime.now() - tick).total_seconds():.1f}s")

    return val_ce_loss / len(val_loader), top_acc


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(f"Using {'GPU.' if use_cuda else 'CPU, as was explicitly requested, or as GPU is not available.'}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # torch.set_deterministic(True)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    kwargs = {}
    if use_cuda:
        # Preprocessing is quick enough to use only 4 workers without wasting time much time (maybe 0.1sec per 1000 batches)
        kwargs.update({'num_workers': 0, 'pin_memory': True})

    dataset_train = dataset_val = None
    if args.dataset == 'eshg':
        args.num_classes = 2  # primary: 2 diseases, secondary: 2 ethnicity classes
        dataset_train = ESHGDataset(in_channels=args.in_channels, secondary='ethnicity')
        dataset_val = ESHGDataset(in_channels=args.in_channels, target_file_path=dataset_train.get_validation_set(),
                                  augment=False, secondary='ethnicity')

        dataset_sec = dataset_train
        # Test to see what happens when training secondary classifier with all data class-pairs
        whole_set_without_validation = pd.concat([dataset_sec.target_file, dataset_val.target_file])
        whole_set_without_validation = whole_set_without_validation.drop_duplicates(keep=False)
        dataset_sec.target_file = whole_set_without_validation

        print(f"Training dataset's lookup table: {dataset_train.get_lookup_table()}")
        dist_p = dataset_train.get_distribution()
        print(f"Training dataset distribution: {dist_p}")
        print(f"Validation dataset's lookup table: {dataset_val.get_lookup_table()}")
        print(f"Validation dataset distribution: {dataset_val.get_distribution()}")
        dist_p, dist_s = dist_p
        _, dist_s = dataset_sec.get_distribution()
    else:
        print(f"Not a valid dataset ({args.dataset} given).")
        raise NotImplementedError

    # Set the batch size of the validation set loader to as high as possible (max = args.batch_size)
    args.val_bs = len(dataset_val) if len(dataset_val) < args.batch_size else args.batch_size
    args.batch_size = len(dataset_train) if len(dataset_train) < args.batch_size else args.batch_size

    train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs, shuffle=True, batch_size=args.batch_size,
                                               worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(dataset_val, **kwargs, shuffle=False, drop_last=True,
                                             worker_init_fn=seed_worker,
                                             batch_size=args.val_bs)
    sec_loader = torch.utils.data.DataLoader(dataset_sec, **kwargs, shuffle=True,
                                                  batch_size=len(dataset_sec)
                                                  if len(dataset_sec) < args.batch_size
                                                  else args.batch_size,
                                                  worker_init_fn=seed_worker)


    # Attempt to deal with possible data imbalance: inverse frequency divided by lowest frequency class (max: 1.0)
    args.ce_weights_p = torch.tensor(
        [(sum(dist_p) / freq) / (sum(dist_p) / min(dist_p)) for freq in dist_p]).float().to(
        device)
    args.ce_weights_s = torch.tensor(
        [(sum(dist_s) / freq) / (sum(dist_s) / min(dist_s)) for freq in dist_s]).float().to(
        device)
    print(f"Weighted cross entropy weights, primary: {args.ce_weights_p}, secondary: {args.ce_weights_s}")

    if args.model_type == 'DeepGestalt':
        model = DeepGestalt(in_channels=args.in_channels, num_classes=args.num_classes, device=device).to(device)
        optimizers = [
            # primary optimizer
            optim.Adam([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': (args.lr * 10)}
            ], lr=args.lr)
        ]
    elif args.model_type == 'BlindDeepGestalt':
        model = BlindDeepGestalt(in_channels=args.in_channels, num_classes_p=args.num_classes,
                                 num_classes_s=args.num_classes, device=device).to(device)
        optimizers = [
            # primary optimizer
            optim.Adam([
                {'params': model.base.parameters()},
                {'params': model.classifier_p.parameters(), 'lr': (args.lr * 10)}
            ], lr=args.lr),
            # secondary optimizer
            optim.Adam([{'params': model.classifier_s.parameters(), 'lr': (args.lr * 10)}], lr=args.lr)
        ]
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")
        raise NotImplementedError

    # Set log intervals
    args.log_interval = args.log_interval // args.batch_size
    args.val_interval = args.val_interval // args.batch_size


    scheduler = None

    # Call explicit model weight initialization
    model.init_layer_weights()

    ## Continue training/testing:
    # model.load_state_dict(torch.load(f"{saved_model_dir}/<saved weights>.pt", map_location=device))

    train(args, model, device, train_loader, sec_loader, optimizers, val_loader=val_loader, scheduler=scheduler)
    validate(model, device, val_loader, args, out=True)


if __name__ == '__main__':
    main()
