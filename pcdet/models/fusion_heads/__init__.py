from .clocs_second_head import ClocsSECONDHead
from .clocs_dense_head import ClocsDenseHead
from .clocs_voxelrcnn_head import ClocsVoxelRCNNHead
from .clocs_sparse_contra_head import ClocsSparseContraHead

__all__ = {
    'ClocsVoxelRCNNHead': ClocsVoxelRCNNHead,
    'ClocsSparseContraHead' : ClocsSparseContraHead,
    'ClocsDenseHead': ClocsDenseHead,
    'ClocsSECONDHead': ClocsSECONDHead,
}
