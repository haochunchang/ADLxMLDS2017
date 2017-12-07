def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('-bz', type=int, default=32, help='batch size for training')
    parser.add_argument('-lr', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('-eps', type=int, default=10000, help='total learing episodes')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--freq', type=float, default=10, help='update frequency')
  
    return parser
