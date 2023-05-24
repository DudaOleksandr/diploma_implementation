import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from commons.model import get_model
from commons.parser_settings import create_default_parser
from evaluation.datasets import TrainDataset, TestDataset
from results.plots.plot_results import *
from train import simclr

parser = create_default_parser(description="This is the command line interface for the linear evaluation model")

parser.add_argument('-model_path', type=str, help="Path to the trained self-supervised model")

# This is list of classes for used dataset
TARGET_NAMES = ['Hbv', 'He', 'IPCL', 'Le']
# TARGET_NAMES = ['pharyngitis', 'no_pharyngitis']

# Number of epoch, model was trained on (just for correct file naming, does not matter for result)
EPOCH = 5


def print_metrics(y_reprs, y_pred, epoch):
    report = classification_report(
        y_reprs,
        y_pred,
        digits=4,
        target_names=TARGET_NAMES, output_dict=True)

    print_report = classification_report(
        y_reprs,
        y_pred,
        digits=4,
        target_names=TARGET_NAMES)

    print(print_report)
    plot_loss(args, epoch)
    plot_roc_auc(y_reprs, y_pred, args, epoch)
    classification_report_csv(report, args, epoch)

    cm = confusion_matrix(y_reprs, ypred)
    plot_confusion_matrix(cm, target_names=TARGET_NAMES, args=args, epoch=EPOCH)


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    model = get_model(args)

    # summary(model, (3, 224, 224))
    # print(model)

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            TrainDataset(args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        ),
        'test': torch.utils.data.DataLoader(
            TestDataset(args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )}

    SimCLR = simclr.SimCLR(model, None, dataloaders, None)
    SimCLR.load_model(args)

    reprs = {}

    for mode in ['train', 'test']:
        reprs[mode] = SimCLR.get_representations(args, mode=mode)

    scaler = StandardScaler().fit(reprs['train']['X'])

    train_x = scaler.transform(reprs['train']['X'])
    test_x = scaler.transform(reprs['test']['X'])

    clf = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        n_jobs=16,
    ).fit(
        train_x, reprs['train']['Y']
    )

    ypred = clf.predict(test_x)

    print_metrics(reprs['test']['Y'], ypred, EPOCH)
