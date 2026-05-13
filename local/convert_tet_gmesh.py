import numpy as np
from simnibs import mesh_io


def read_tetrahedron(file_name):

    tetrahedrons = []
    vertices = []

    # generate tetrahedron is not exist files
    with open(file_name, 'r') as f:
        line = f.readline()
        line = line.strip().split(' ')
        n_vert = int(line[1])
        n_t = int(line[2])
        for i in range(n_vert):
            line = f.readline()
            line = line.strip().split(' ')
            assert len(line) == 3
            vertices.append([float(v) for v in line])
        for i in range(n_t):
            line = f.readline()
            line = line.strip().split(' ')
            assert len(line) == 4
            tetrahedrons.append([int(v) for v in line])


    assert len(tetrahedrons) == n_t
    assert len(vertices) == n_vert
    # import ipdb
    # ipdb.set_trace()
    vertices = np.asarray(vertices)
    return vertices, np.asarray(tetrahedrons)



if __name__=="__main__":
    v, tets = read_tetrahedron('cube_0.014286_tet.tet')
    m = mesh_io.Msh()
    m.elm = mesh_io.Elements(tetrahedra=tets + 1)
    # Multiply by 100 to make the node coordinates be between 0 and 100 and not 0 and 1
    m.nodes = mesh_io.Nodes(100*v)
    m.write('cube_test.msh')