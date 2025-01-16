import argparse
from sklearn.linear_model import LinearRegression
from pymilo.streaming import PymiloServer, Compression, CommunicationProtocol
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def main():
    parser = argparse.ArgumentParser(description='Run the Pymilo server with a specified compression method.')
    parser.add_argument(
        '--compression',
        type=str,
        choices=['NULL', 'GZIP', 'ZLIB', 'LZMA', 'BZ2'],
        default='NULL',
        help='Specify the compression method (NULL, GZIP, ZLIB, LZMA, or BZ2). Default is NULL.'
        )
    parser.add_argument(
        '--protocol',
        type=str,
        choices=['REST', 'WEBSOCKET'],
        default='REST',
        help='Specify the communication protocol (REST or WEBSOCKET). Default is REST.'
        )
    parser.add_argument(
        '--init',
        action="store_true",
        default=False,
        help='the `init` command specifies whether or not initializing the PyMilo Server with a ML model.',
    )
    args = parser.parse_args()
    communicator = None
    if args.init:
        x_train, y_train, _, _ = prepare_simple_regression_datasets()
        linear_regression = LinearRegression()
        linear_regression.fit(x_train, y_train)
        communicator = PymiloServer(
            model=linear_regression,
            port=9000,
            compressor=Compression[args.compression],
            communication_protocol= CommunicationProtocol[args.protocol],
            ).communicator
    else:
        communicator = PymiloServer(
            port=8000,
            compressor=Compression[args.compression],
            communication_protocol=CommunicationProtocol[args.protocol],
            ).communicator

    communicator.run()

if __name__ == '__main__':
    main()