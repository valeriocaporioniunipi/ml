from utils import abs_path, get_data
filename = 'ML-CUP24-TR.csv'
filepath = abs_path(filename, 'data')
features, targets = get_data(filepath, 3, 1)
