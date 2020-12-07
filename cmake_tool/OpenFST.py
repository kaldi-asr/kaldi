import argparse
import os
import shutil
import subprocess
import tarfile
import distutils.spawn
import logging

from urllib.request import urlretrieve
from os import path

BASIC_ERROR = -256


def configure(source_path, install_prefix: str):
    if not distutils.spawn.find_executable('gcc'):
        print('[ERROR] Can not find executable "gcc" (need gcc/g++/make)')
        os.sys.exit(BASIC_ERROR-1)
    elif not distutils.spawn.find_executable('g++'):
        print('[ERROR] Can not find executable "g++" (need gcc/g++/make)')
        os.sys.exit(BASIC_ERROR-1)

    openfst_add_CXXFLAGS = "-g -O2"
    OPENFST_CONFIGURE = "--enable-shared --enable-far --enable-ngram-fsts --enable-lookahead-fsts --with-pic"
    command = './configure %s CXX="${CXX}" --prefix=%s\
            CXXFLAGS="${CXXFLAGS} %s" LDFLAGS="${LDFLAGS}" LIBS="-ldl"' % (OPENFST_CONFIGURE, install_prefix, openfst_add_CXXFLAGS)
    result = subprocess.run(command, shell=True, cwd=source_path)
    if result.returncode != 0:
        print('[ERROR] Configure return code is %d, abort!' %
              (result.returncode))
        os.sys.exit(BASIC_ERROR-1)
    result = subprocess.run('make clean', shell=True, cwd=source_path)
    assert result.returncode == 0


def make(source_path, nj=4):
    if not distutils.spawn.find_executable('make'):
        print('[ERROR] Can not find executable "make" (need gcc/g++/make)')
        os.sys.exit(BASIC_ERROR-2)

    command = 'make'
    if nj > 1:
        command += ' -j%d' % (nj)

    result = subprocess.run(command, shell=True, cwd=source_path)
    if result.returncode != 0:
        print(
            '[NOTE] If too many parallel job is executed, GCC might fail due to memory shortage.')
        print('[ERROR] Make return code is %d, abort!' % (result.returncode))
        os.sys.exit(BASIC_ERROR-2)


def install(source_path, install_prefix: str, nj=4):
    command = 'make install'
    if nj > 1:
        command += ' -j%d' % (nj)

    result = subprocess.run(command, shell=True, cwd=source_path)
    if result.returncode != 0:
        print('[ERROR] Insall return code is %d, abort!' % (result.returncode))
        os.sys.exit(BASIC_ERROR-3)


if __name__ == "__main__":
    BUILD_DIR = 'build'
    OPENFST_VERSION = '1.6.7'
    OPENFST_TARBALL_PATH = path.join(
        BUILD_DIR, "openfst-%s.tar.gz" % (OPENFST_VERSION))
    OPENFST_BUILD_FOLDER = path.join(
        BUILD_DIR, "openfst-%s" % (OPENFST_VERSION))

    parser = argparse.ArgumentParser(description='This is a python script to download/configure/make/install \
        OpenFST library and related binaries. This is needed by Kaldi during compilation.')
    parser.add_argument(
        '--target', type=str, choices=['download', 'configure', 'make', 'install', 'all'], default='all',
        help='Action to perform to compile/install OpenFST. Default target is "all".')
    parser.add_argument('--local_path', default='', type=str,
                        help='If this is set, local file is used as source tarball, instead of downloading from internet.')
    parser.add_argument('--nj', default=4, type=int,
                        help='Designate how many parallel job to run. Default is 4.')
    parser.add_argument('install_prefix', type=str, nargs=1)

    args = parser.parse_args()
    install_prefix = args.install_prefix[0]

    # Make build directory if not exist
    if not path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)

    # Either download OpenFST tarball, or use local file
    if (args.target in ['all', 'download']) or not path.exists(OPENFST_TARBALL_PATH):
        if args.local_path:
            if not path.exists(args.local_path):
                print('[ERROR] local tarball %s not exist, exit!' %
                      args.local_path)
                parser.print_help()
                os.sys.exit(BASIC_ERROR)

            if path.basename(args.local_path) != path.basename(OPENFST_TARBALL_PATH):
                print('[ERROR] OpenFST tarball name %s does not match system requriement!') \
                    % (path.basename(args.local_path),  path.basename(OPENFST_TARBALL_PATH))
                os.sys.exit(BASIC_ERROR)

            shutil.copy(args.local_path, OPENFST_TARBALL_PATH)
        else:
            url = "http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-%s.tar.gz" % (
                OPENFST_VERSION)
            urlretrieve(url, filename=OPENFST_TARBALL_PATH)

    # Configure/Build/Install OpenFST
    if args.target in ['configure', 'all']:
        assert path.exists(OPENFST_TARBALL_PATH)

        if not path.exists(OPENFST_BUILD_FOLDER):
            print('[INFO] Extracting tarball')
            with tarfile.open(OPENFST_TARBALL_PATH) as tar:
                tar.extractall(BUILD_DIR)

        configure(OPENFST_BUILD_FOLDER, install_prefix)

    if args.target in ['make', 'all']:
        assert path.exists(OPENFST_BUILD_FOLDER)

        make(OPENFST_BUILD_FOLDER, nj=args.nj)

    if args.target in ['install', 'all']:
        assert path.exists(OPENFST_BUILD_FOLDER)

        install(OPENFST_BUILD_FOLDER, install_prefix, nj=args.nj)
