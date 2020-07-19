import pickle as pk
import numpy as np

# save as binary file
def data_writer_b(file_path, data):
    with open(file_path, 'wb') as file:
        pk.dump(data, file)

# save as txt
def data_writer_txt(file_path, data):
    with open(file_path, 'w') as file:
        file.write(str(data))
        file.write('\n')

# read binary file
def data_reader_b(file_path):
    f = open(file_path, 'rb')
    data = pk.load(f)
    f.close()
    return data

# normalize the adjacent matrix
def toScale(mat):
    matrix = np.copy(mat)

    max_value = np.max(matrix)
    min_value = np.min(matrix)
    dist = max_value - min_value

    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            matrix[i, j] -= min_value
            if dist != 0:
                matrix[i, j] /= dist

    return matrix

def data_reader_scale(file_path):
    # read data
    data = data_reader_b(file_path)
    # normalization
    for i in range(0, data.shape[0]):
        data[i] = toScale(data[i])
    return data