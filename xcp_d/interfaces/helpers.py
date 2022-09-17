"""Helper functions for interfaces."""
import glob
import os
import shutil


def find_files(seek_dir, pattern):
    """Find all files within the directory specified that match the glob-style pattern.

    Parameters
    ----------
    seek_dir : str
        Directory to be searched.
    pattern : str
        Unix shell pattern for finding files.

    Returns
    -------
    paths : list of str
        List of relative paths of copied files (may be empty).
    """
    paths = []
    glob_pattern = os.path.join(seek_dir, pattern)
    for found_file in glob.glob(glob_pattern):
        paths.append(found_file)

    return paths


def find_and_copy_files(seek_dir, pattern, output_dir):
    """Find all files within the directory specified that match the glob-style pattern.

    Copies each file to the output directory.

    Parameters
    ----------
    seek_dir : str
        Directory to be searched.
    pattern : str
        Unix shell pattern for finding files.
    output_dir : str
        Directory to which to copy files.

    Returns
    -------
    rel_paths : list of str
        List of relative paths of copied files (may be empty).
    """
    rel_paths = []

    glob_pattern = os.path.join(seek_dir, pattern)
    for found_file in glob.glob(glob_pattern):
        # TODO: change name to BIDS name?
        filename = os.path.basename(found_file)
        rel_path = os.path.relpath(os.path.join(output_dir, filename),
                                   os.getcwd())
        shutil.copy(found_file, rel_path)
        rel_paths.append(rel_path)

    return rel_paths


def find_and_copy_file(seek_dir, pattern, output_dir):
    """Find a single file within seek_dir, using the pattern.

    If found, copies the file to the output_dir.

    Parameters
    ----------
    seek_dir : str
        Directory to be searched.
    pattern : str
        Unix shell pattern for finding files.
    output_dir : str
        Directory to which to copy the file.

    Returns
    -------
    rel_path : str
        Relative path to copied file, or None.
    """
    found_path = find_one_file(seek_dir, pattern)

    if found_path:
        # TODO: change name to BIDS name?
        # Copy the file to output_dir.
        filename = os.path.basename(found_path)
        rel_path = os.path.relpath(os.path.join(output_dir, filename),
                                   os.getcwd())
        shutil.copyfile(found_path, rel_path)
        return rel_path

    else:
        return None


def find_one_file(seek_dir, pattern):
    """Find a single file within seek_dir, using the pattern.

    Parameters
    ----------
    seek_dir : str
        Directory to be searched.
    pattern : str
        Unix shell pattern for finding files.

    Returns
    -------
    one_file : str
        Path to the found file.
    """
    one_file = None

    # Try to find a file with the pattern given in the directory given.
    glob_pattern = os.path.join(seek_dir, pattern)
    filelist = glob.glob(glob_pattern)

    # Make sure we got exactly one file.
    # numfiles = len(filelist)
    # if numfiles is 1:
    # one_file = filelist[0]
    # else:
    # TODO: Log info in errorfile.
    # print('info: Found %s files with pattern: %s' % (numfiles, glob_pattern))
    one_file = filelist[0]
    return one_file
