
def get_top_k_indices(arr, k=1):
    """
        numpy function
        Returns top k elements in the array
    :param arr: numpy array
    :param k: {constant} top "k"
    :return: {List} top k (by value) element indices of the array
    """

    return arr.argsort()[-k:][::-1]

