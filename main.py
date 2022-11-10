import os

from utils.data import load_object, save_object
from utils.operator import Operator
from utils.parser import get_parser

OPERATOR_PATH = 'operator.pkl'


def main():
    try:
        operator = load_object(OPERATOR_PATH)
    except FileNotFoundError:
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
