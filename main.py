import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Attribute Recognition Framework',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--setting', type=str, default='config.yaml',
        help='Path to the setting file')

    parser.add_argument('-T', '--training', type=bool, action='store_true',
        help='Training mode')
    parser.add_argument('-V', '--verbose', type=bool, action='store_true',
        help='Print the training process')

    args = parser.parse_args()
    return args


def main():
    pass


if __name__ == '__main__':
    main()