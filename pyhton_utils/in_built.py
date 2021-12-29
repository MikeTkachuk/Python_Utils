def unzip(x):
    output = []
    x = list(x)
    for i in range(len(x[0])):
        output.append([item[i] for item in x])
    return tuple(output)