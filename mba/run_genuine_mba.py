#!/usr/bin/env python3

from mba.config import build_parser
from mba.experiment import run


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()