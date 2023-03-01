import argparse


def parse_argument_main():
    arguments = {}
    arguments['--exp'] = dict(type=str, default='', help='Name of the experiment')

    return arguments


def get_main_parser():
    parser = argparse.ArgumentParser(description='Counterfactual VAE options')
    for name, options in parse_argument_main().items():
        parser.add_argument(name, **options)
    print(parser.parse_args())


get_main_parser()