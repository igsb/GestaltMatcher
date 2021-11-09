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


def normalize(img, type='float'):
    normalized = (img - img.min()) / (img.max() - img.min())
    if type == 'int':
        return (normalized * 255).int()

    # Else: float
    return normalized


# Simple preprocessing used for the input images
def preprocess(img):
    resize = transforms.Resize((100, 100))  # Default size is (100,100)
    img = resize(img)

    # desired number of channels is 1, so we convert to gray
    img = transforms.Grayscale(1)(img)
    #img = transforms.RandomHorizontalFlip(p=1.0)(img)
    return transforms.ToTensor()(img)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict DeepGestalt')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model-type', default='DeepGestalt', dest='model_type',
                        help='Model type to use. Default: DeepGestalt. (Options: \'FaceRecogNet\', \'DeepGestalt\')')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--act_type', default='ReLU', dest='act_type',
                        help='activation function to use in model. (Options: ReLU, PReLU, Swish)')
    parser.add_argument('--in_channels', default=1, dest='in_channels')
    parser.add_argument('--num_classes', default=139, dest='num_classes', type=int)

    parser.add_argument('--data_dir', default='../data/GestaltMatcherDB/images_cropped', dest='data_dir',
                        help='Path to the data directory containing the images to run the model on.')

    return parser.parse_args()


def predict(model, device, data, args):
    model.eval()

    f = None
    if args.model_type == "FaceRecogNet":
        f = open("healthy_encodings.csv", "w+")
        f.write(f"img_name;arg_max;representations\n")
    elif args.model_type == "DeepGestalt":
        f = open("encodings.csv", "w+")
        f.write(f"img_name;class_conf;representations\n")
    else:
        raise NotImplementedError

    tick = datetime.datetime.now()
    with torch.no_grad():
        for idx, img_path in enumerate(data):
            print(f"{img_path=}")
            img = Image.open(f"{args.data_dir}/{img_path}")
            img = preprocess(img).to(device, dtype=torch.float32)

            pred, pred_rep = model(img.unsqueeze(0))

            if args.model_type == "FaceRecogNet":
                f.write(f"{img_path};{torch.argmax(pred)};{pred_rep.squeeze().tolist()}\n")
            else:
                f.write(f"{img_path};{pred.squeeze().tolist()};{pred_rep.squeeze().tolist()}\n")

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
    # data = [f"{root.split('/')[-1]}/{img_name}" for dir in data for root,_,img_names in os.walk(f"{args.data_dir}/{dir}") for img_name in img_names]

    if args.act_type == "ReLU":
        act_type = nn.ReLU
    elif args.act_type == "PReLU":
        act_type = nn.PReLU
    elif args.act_type == "LeakyReLU":
        act_type = nn.LeakyReLU
    else:
        raise NotImplementedError

    if args.model_type == 'FaceRecogNet':
        model = FaceRecogNet(in_channels=args.in_channels,
                             num_classes=10575,
                             # num_classes=args.num_classes,
                             act_type=act_type).to(device)

        # load model:
        model.load_state_dict(
            torch.load(f"saved_models/s1_casia_adam_FaceRecogNet_e50_ReLU_BN_bs100.pt",
                       map_location=device))
    elif args.model_type == 'DeepGestalt':
        model = DeepGestalt(in_channels=args.in_channels,
                            num_classes=args.num_classes,
                            device=device,
                            pretrained=False,  # No need to load them as we're loading full weights after..
                            act_type=act_type).to(device)

        # load model:
        model.load_state_dict(
            #torch.load(f"saved_models/s1_casia_adam_FaceRecogNet_e50_ReLU_BN_bs100.pt",
            torch.load(f"saved_models/s2_gmdb_aug_adam_DeepGestalt_e310_ReLU_BN_bs280.pt",
            #torch.load(f"saved_models/encoderdecoder_test.pt",
            # torch.load(f"saved_models/s3_gmdb_aug_adam_DeepGestalt_e150_ReLU_bs280.pt",
                       map_location=device))
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")
        raise NotImplementedError

    predict(model, device, data, args)


if __name__ == '__main__':
    main()
