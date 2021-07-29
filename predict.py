## predict.py
# Run the chosen model on every image in the desired dataset
# and save the encodings to the file: "encodings.csv"

import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from lib.models.deep_gestalt import DeepGestalt
from lib.models.face_recog_net import FaceRecogNet

saved_model_dir = "saved_models"
encoding_file = "encodings.csv"
zero = torch.tensor(0.).cuda()


# Simple preprocessing used for the input images
def preprocess(img):
    resize = transforms.Resize((100, 100))  # Default size is (100,100)
    img = resize(img)

    # desired number of channels is 1, so we convert to gray
    img = transforms.Grayscale(1)(img)
    return transforms.ToTensor()(img)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict DeepGestalt')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save_encodings', action='store_true', default=True,
                        help='Whether to save to encodings to the file \'encodings.csv\'')
    parser.add_argument('--model-type', default='DeepGestalt', dest='model_type',
                        help='Model type to use. (Options: \'FaceRecogNet\', \'DeepGestalt\')')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--act_type', default='ReLU', dest='act_type',
                        help='activation function to use in UNet. (Options: ReLU, PReLU, LeakyReLU, Swish)')
    parser.add_argument('--in_channels', default=1, dest='in_channels')
    parser.add_argument('--num_classes', default=139, dest='num_classes', type=int)

    parser.add_argument('--data_dir', default='../data/GestaltMatcherDB/images_cropped', dest='data_dir',
                        help='Path to the data directory containing the images to run the model on.')

    return parser.parse_args()


def predict(model, device, data, args):
    model.eval()

    f = None
    if args.save_encodings:
        f = open("encodings.csv", "w+")
        if args.model_type == "FaceRecogNet":
            f.write(f"img_name,arg_max,representations\n")
        else:  # model_type == "DeepGestalt"
            f.write(f"img_name;class_conf;representations\n")

    tick = datetime.datetime.now()
    with torch.no_grad():
        for idx, img_path in enumerate(data):
            print(f"{img_path=}")
            img = Image.open(f"{args.data_dir}/{img_path}")
            img = preprocess(img).to(device, dtype=torch.float32)

            pred, pred_rep = model(img.unsqueeze(0))

            if args.save_encodings:
                # f.write(f"{img_path},{torch.argmax(pred)},{pred_rep.squeeze().tolist()}\n")
                f.write(f"{img_path};{pred.squeeze().tolist()};{pred_rep.squeeze().tolist()}\n")

    if args.save_encodings:
        f.flush()
        f.close()

    print(f"Predictions took {datetime.datetime.now() - tick}s")
    model.train()
    return


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    kwargs = {}
    if use_cuda:
        kwargs.update({'num_workers': 0, 'pin_memory': True})

    data = os.listdir(args.data_dir)

    if args.act_type == "ReLU":
        act_type = nn.ReLU
    elif args.act_type == "PReLU":
        act_type = nn.PReLU
    elif args.act_type == "LeakyReLU":
        act_type = nn.LeakyReLU
    else:
        print(f"Invalid ACT_type given! (Got {args.act_type})")
        act_type = nn.ReLU

    if args.model_type == 'FaceRecogNet':
        model = FaceRecogNet(in_channels=args.in_channels, num_classes=args.num_classes, act_type=act_type).to(device)
        model_name = "FaceRecogNet"

        # load model:
        model.load_state_dict(
            torch.load(f"saved_models/s1_casia_adam_FaceRecogNet_e50_ReLU_BN_bs100.pt",
                       map_location=device))
    elif args.model_type == 'DeepGestalt':
        model = DeepGestalt(in_channels=args.in_channels, num_classes=args.num_classes, act_type=act_type).to(device)
        model_name = "DeepGestalt"

        # load model:
        model.load_state_dict(
            torch.load(f"saved_models/s2_gmdb_aug_adam_DeepGestalt_e310_ReLU_BN_bs280.pt",
                       map_location=device))
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")

    predict(model, device, data, args)


if __name__ == '__main__':
    main()
