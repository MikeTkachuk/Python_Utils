def unzip(x):
    """
    Make an operation inverse to an in-built zip funciton
    :param x: An iterable containing array-like items to unpack
    :return: A tuple of the resulting lists
    """
    output = []
    x = list(x)
    for i in range(len(x[0])):
        output.append([item[i] for item in x])
    return tuple(output)
