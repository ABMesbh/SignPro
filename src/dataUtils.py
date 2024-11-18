import numpy as np

def normal(value,min,max):
    return (value-min)/(max-min)

def normalizeData(data):
    """Returns the normalized data from the input hand list of 63 element [elem0.x, elem0.y, elem0.z, ...., elemN.x, elemN.y, elemN.z]."""
    x_elements = [data[i] for i in range(len(data)) if i % 3 == 0]
    y_elements = [data[i] for i in range(len(data)) if i % 3 == 1]
    z_elements = [data[i] for i in range(len(data)) if i % 3 == 2]
    nx_elements = [0]*len(x_elements)
    ny_elements = [0]*len(y_elements)
    nz_elements = [0]*len(z_elements)
    for i in range(len(x_elements)):
        nx_elements[i] = normal(x_elements[i],min(x_elements),max(x_elements))
        ny_elements[i] = normal(y_elements[i],min(y_elements),max(y_elements))
        nz_elements[i] = normal(z_elements[i],min(z_elements),max(z_elements))
    return np.array([[nx_elements[i],ny_elements[i],nz_elements[i]] for i in range(len(nx_elements))]).flatten()

if __name__ == "__main__":
    data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(normalizeData(data))