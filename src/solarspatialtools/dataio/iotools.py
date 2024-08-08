import os
import urllib
import urllib.request


def download(url, filename):
    """
    Retrieve a file from the web and save it to a location

    Parameters
    ----------
    url : String
        The url of the file to download

    filename :
        The location of where to save it

    Returns
    -------
    void
    """

    urllib.request.urlretrieve(url, filename)


def wget_fromurl(wgeturl, targetdir, verbose=True):
    """
    Run a wget script to download files to a target dir

    Parameters
    ----------
    wgeturl : string
        url of the wget file

    targetdir : string
        Target directory for saving the files

    verbose : bool
        print status updates?

    Returns
    -------
    void
    """

    # to use locally replace: "with open(wgeturl)"
    with urllib.request.urlopen(wgeturl) as fl:
        for rowb in fl:  # rows come in as binary
            row = rowb.decode('utf-8')  # comes back as a bytestream
            # just get the URL to the file
            u = row.split()[-1].strip()
            # just get the filename portion
            fn = os.path.split(u)[-1]
            # Download the file
            if not os.path.isfile(os.path.join(targetdir, fn)):
                if verbose:
                    print(fn)
                download(u, os.path.join(targetdir, fn))
            else:
                if verbose:
                    print(fn + " exists, skipping...")


def ensurepath(path):
    """
    Test if a path exists, if not, create it
    Parameters
    ----------
    path : str
        The path to test

    Returns
    -------
        Void
    """
    if not os.path.exists(path):
        os.makedirs(path)


def getfiles(path, ext=None, sort=True):
    """
    Get a list of all files on a path, optionally filtering by ext and sorting
    Parameters
    ----------
    path : str
        the path to search
    ext : str
        the extension to match as string (e.g. ".txt").
        If None, return all files
    sort : bool
        should the files be sorted?

    Returns
    -------
    files : list
        A list of the matching files found in the directory
    """
    baselist = os.listdir(path)
    files = []

    # Split extensions if desired
    if ext is not None:
        for fn in baselist:
            if os.path.splitext(fn)[-1] == ext:
                files.append(fn)
    else:
        files = baselist

    # Sort if desired
    if sort:
        files.sort()

    return files
