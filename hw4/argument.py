def add_arguments(parser):
    parser.add_argument('-bz', type=int, default=64, help='batch size for training')
    parser.add_argument('-lr', type=float, default=0.0002, help='learning rate for training')
    parser.add_argument('-epochs', type=int, default=600, help='total learing epochs')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--beta1', type=float, default=0.5, help='Momentum for Adam Update')
    parser.add_argument('--save_every', type=int, default=30, help='Save Model/Samples every x iterations over batches')
    parser.add_argument('--resume_model', type=str, default=None, help='Pre-Trained Model Path, to resume from')
    parser.add_argument('--preload', type=bool, default=True, help='Pre-Loaded Data')
 
    return parser
