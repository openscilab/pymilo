# -*- coding: utf-8 -*-
"""PyMilo main."""
import re
import argparse
from art import tprint
from .pymilo_param import (
    PYMILO_VERSION,
    URL_REGEX,
    CLI_MORE_INFO,
    CLI_UNKNOWN_MODEL,
    CLI_ML_STREAMING_NOT_INSTALLED,
)
from .pymilo_func import print_supported_ml_models, pymilo_help
from .pymilo_obj import Import
from .utils.util import get_sklearn_class

ml_streaming_support = True
try:
    from .streaming import PymiloServer, Compression, CommunicationProtocol
except BaseException:
    ml_streaming_support = False


def main():
    """
    CLI main function.

    :return: None
    """
    parser = argparse.ArgumentParser(description='Run the Pymilo server with a specified compression method.')
    parser.add_argument(
        '--compression',
        type=str,
        choices=['NULL', 'GZIP', 'ZLIB', 'LZMA', 'BZ2'],
        default='NULL',
        help='Specify the compression method (NULL, GZIP, ZLIB, LZMA, or BZ2). Default is NULL.'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Specify PyMiloServer port number',
        metavar="",
    )
    parser.add_argument(
        '--protocol',
        type=str,
        choices=['REST', 'WEBSOCKET'],
        default='REST',
        help='Specify the communication protocol (REST or WEBSOCKET). Default is REST.'
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='the `load` command specifies the path to the JSON file of the previously exported ML model by PyMilo.',
        metavar="",
    )
    parser.add_argument(
        '--init',
        type=str,
        default=None,
        help='the `init` command specifies the ML model to initialize the PyMilo Server with.',
        metavar="",
    )
    parser.add_argument(
        '--bare',
        default=False,
        action='store_true',
        help='The `bare` command starts the PyMilo Server without an internal ML model.',
    )
    parser.add_argument('--version', action='store_true', default=False, help='PyMilo version')
    parser.add_argument('-v', action='store_true', default=False, help='PyMilo version')
    args = parser.parse_args()
    if args.version or args.v:
        print(PYMILO_VERSION)
        return
    if not ml_streaming_support:
        print(CLI_ML_STREAMING_NOT_INSTALLED)
        print(CLI_MORE_INFO)
        tprint("PyMilo")
        tprint("V:" + PYMILO_VERSION)
        pymilo_help()
        parser.print_help()
        return
    run_ps = False
    _model = None
    _port = args.port
    _compressor = Compression[args.compression]
    _communication_protocol = CommunicationProtocol[args.protocol]
    if args.load:
        path = args.load
        run_ps = True
        _model = Import(url=path) if re.match(URL_REGEX, path) else Import(file_adr=path)
        _model = _model.to_model()
    elif args.init:
        model_name = args.init
        model_class = get_sklearn_class(model_name)
        if model_class is None:
            print(f"{CLI_UNKNOWN_MODEL}\n{print_supported_ml_models()}")
            return
        run_ps = True
        _model = model_class()
    elif args.bare:
        run_ps = True
    if not run_ps:
        tprint("PyMilo")
        tprint("V:" + PYMILO_VERSION)
        pymilo_help()
        parser.print_help()
    else:
        PymiloServer(
            model=_model,
            port=_port,
            compressor=_compressor,
            communication_protocol=_communication_protocol,
        ).communicator.run()


if __name__ == '__main__':
    main()
