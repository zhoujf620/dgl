import dgl
import backend as F
import unittest


@unittest.skipIf(F._default_context_str == 'cpu', reason="Need gpu for this test")
def test_to_device():
    g = dgl.DGLGraph()
    g.add_nodes(5, {'h' : F.ones((5, 2))})
    g.add_edges([0, 1], [1, 2], {'m' : F.ones((2, 2))})
    if F.is_cuda_available():
        g = g.to(F.cuda())
        assert g is not None

@unittest.skipIf(F._default_context_str == 'cpu', reason="Need gpu for this test")
def test_to_device_non_blocking():
    # to test if DGLGraph.to is compatible with PyTorch Lighting
    # related issue: https://github.com/dmlc/dgl/issues/1547
    g = dgl.DGLGraph()
    g.add_nodes(5, {'h' : F.ones((5, 2))})
    g.add_edges([0, 1], [1, 2], {'m' : F.ones((2, 2))})
    if F.is_cuda_available():
        g = g.to(F.cuda(), non_blocking=True)
        assert g is not None

if __name__ == '__main__':
    test_to_device()
    test_to_device_non_blocking()
