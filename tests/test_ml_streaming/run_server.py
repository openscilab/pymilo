import argparse
from pymilo.streaming import Compression
from pymilo.streaming import PymiloServer


def main():
    parser = argparse.ArgumentParser(description='Run the Pymilo server with a specified compression method.')
    parser.add_argument('--compression', type=str, choices=['NULL', 'GZIP', 'ZLIB', 'LZMA', 'BZ2'], default='NULL',
                        help='Specify the compression method (NULL, GZIP, ZLIB, LZMA, or BZ2). Default is NULL.')
    args = parser.parse_args()
    communicator = PymiloServer(compressor=Compression[args.compression]).communicator
    communicator.run()

if __name__ == '__main__':
    main()
