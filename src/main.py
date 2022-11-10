import os

from utils.data import load_object, save_object
from utils.operator import Operator
from utils.parser import get_parser

OPERATOR_PATH = os.environ.get('OPERATOR_PATH', default='operator.pkl')


def main():
    try:
        operator = load_object(OPERATOR_PATH)
    except FileNotFoundError:
        operator = Operator()
    except EOFError:
        print('error: pfam model file is damaged, please delete it or initialize a new model')
        operator = Operator()

    parser = get_parser(operator)

    args = parser.parse_args()
    args.func(**vars(args))

    if args.cmd == 'del':
        if os.path.exists(OPERATOR_PATH):
            os.remove(OPERATOR_PATH)
        return

    save_object(operator, OPERATOR_PATH)


if __name__ == '__main__':
    main()
