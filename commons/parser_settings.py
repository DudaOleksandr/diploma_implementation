import argparse


def create_default_parser(description):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-datapath', type=str,
                        help="Path to the data root folder which contains train and test folders")

    parser.add_argument('-respath', type=str, help="Path to the results would be stored. ")

    parser.add_argument('-bs', '--batch_size', default=250, type=int, help="The batch size for evaluation")

    parser.add_argument('-nw', '--num_workers', default=1, type=int, help="The number of workers for loading data")

    parser.add_argument('-c', '--cuda', action='store_true')

    parser.add_argument('--multiple_gpus', action='store_true')

    parser.add_argument('--remove_top_layers', default=1, type=int)

    return parser
