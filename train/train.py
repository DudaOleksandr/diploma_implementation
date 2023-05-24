import torch
import simclr
from commons import ntxent_loss, model
from commons.parser_settings import create_default_parser
from train_dataset import TrainDataset

parser = create_default_parser(
    description="This is the command line interface for training model")

parser.add_argument('-ep', '--epochs', default=10, type=int, help="The number of epochs for training")

if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    model = model.get_model(args)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            TrainDataset(args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
    }

    loss_fn = ntxent_loss.get_ntxent
    SimCLR = simclr.SimCLR(model, optimizer, dataloaders, loss_fn)
    print("====================================================\n")
    print(f"Started training: number of epochs: {args.epochs}, dataset: {args.datapath}\n")
    SimCLR.train(args, args.epochs, 10)
    print("====================================================\n")
    print("Finished training")
