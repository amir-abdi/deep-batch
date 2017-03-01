import urllib.request
import os
import tarfile


def download_dataset(url, name):
    if not os.path.exists('data'):
        os.makedirs('data')
    filename = 'data/' + name
    if not os.path.isfile(filename):
        print('Downloading {} dataset...'.format(name))
        testfile = urllib.request.URLopener()
        testfile.retrieve(url, filename)
        print('Download complete.')
    else:
        print('Dataset {} exists'.format(name))

    tar = tarfile.open(filename)
    tar.extractall('data/')
    tar.close()


def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='ascii')
    fo.close()
    return dict