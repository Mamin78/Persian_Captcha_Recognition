import argparse
import warnings
import Model
import DataFunctions as df
import DataPreprocessor as dp
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Train as T
import multiprocessing as mp
import CaptchaDataset as CD
warnings.filterwarnings("ignore")


def imshow(epoch_losses, iteration_losses, saving_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(epoch_losses)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(iteration_losses)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    plt.savefig(saving_path)
    plt.show()


parser = argparse.ArgumentParser()
# train
parser.add_argument('--train', action="store_true",
                    help='Should Model be trained or it has to use pre-trained weights?')
# directories
parser.add_argument('--path_to_data', default='/content/drive/MyDrive/Datasets/12words_cleaned.zip',
                    help='base path to dataset!')
parser.add_argument('--exp_dir', default='experiments', help='path to experiments directory')
parser.add_argument('--exp', required=True, type=str, help='experiments number e.g. 01')
parser.add_argument('--checkpoint_path', default='/content/exp/01/checkpoints/best_checkpoint.pth', type=str,
                    help='Where the model weights are')
# parameters
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--rnn_hidden_size', type=int, default=256, help='rnn hidden size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--clip_norm', type=int, default=5, help='clip norm')

args = parser.parse_args()
cpu_count = mp.cpu_count()

if not os.path.exists(os.path.join(args.exp_dir, args.exp)):
    os.makedirs(os.path.join(args.exp_dir, args.exp))

idx2char, char2idx = dp.get_maps_from_path(args.path_to_data)
num_chars = len(char2idx)

images_list = dp.get_list_of_images(args.path_to_data)
train_images_list, test_images_list = train_test_split(images_list, random_state=0)

transform_ops = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
train_set = CD.CAPTCHADataset(args.path_to_data, train_images_list, transform_ops)
test_set = CD.CAPTCHADataset(args.path_to_data, test_images_list, transform_ops)

train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=cpu_count, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=cpu_count, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('You are using ', device)

resnet = resnet18(pretrained=True)

crnn = Model.CRNN(num_chars, resnet, rnn_hidden_size=args.rnn_hidden_size)
crnn.apply(Model.weights_init)
crnn = crnn.to(device)

criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(crnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

if args.train:
    epoch_losses, iteration_losses = T.train(crnn, train_loader, args.n_epochs, device, args.clip_norm, optimizer,
                                             lr_scheduler, criterion,
                                             args.checkpoint_path, char2idx)
    plot_path = os.path.join(args.exp_dir, args.exp, 'plot.png')
    imshow(epoch_losses, iteration_losses, plot_path)
else:
    crnn.load_state_dict(torch.load(args.checkpoint_path))

train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1, shuffle=False)
results_train = T.get_prediction(crnn, train_loader, device, idx2char)

test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=1, shuffle=False)
results_test = T.get_prediction(crnn, test_loader, device, idx2char)

results_train['prediction_corrected'] = results_train['prediction'].apply(df.correct_prediction)
results_test['prediction_corrected'] = results_test['prediction'].apply(df.correct_prediction)

train_accuracy = accuracy_score(results_train['actual'], results_train['prediction_corrected'])
print('The final train accuracy is ', train_accuracy)
test_accuracy = accuracy_score(results_test['actual'], results_test['prediction_corrected'])
print('The final test accuracy is ', test_accuracy)
