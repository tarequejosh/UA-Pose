"""Runtime shims that let MMPose 1.x run on the only stack the RTX 5060 supports.

Blackwell (sm_120) needs PyTorch >= 2.7 with CUDA 12.8, but OpenMMLab's last releases
predate it, so two incompatibilities have to be bridged:

1. ``torch.load`` defaults to ``weights_only=True`` from 2.6 on, which rejects MMEngine
   checkpoints (they pickle numpy scalars and ``HistoryBuffer``). Only load checkpoints
   you trust; ``scripts/verify_checkpoints.py`` checks the official ones by SHA-256.
2. There are no ``mmcv`` wheels with compiled ops for this stack, so ``mmcv-lite`` is
   used. ``mmpose.models`` still imports ``mmcv.ops`` at module level for EDPose, which
   this project does not use, so a stub raises only if those ops are actually built.
3. ``xtcocotools`` has no wheel for this Python, see ``_alias_xtcocotools``.

Import this module before anything imports ``mmpose``.
"""
import os
import sys
import types


def apply() -> None:
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    _stub_mmcv_ops()
    _alias_xtcocotools()
    _stub_mmdet()


def _stub_mmdet() -> None:
    """Provide the two names RTMO's head imports from mmdet at module level.

    mmdet is only needed for RTMO and RTMDet, neither used here, and its releases refuse
    mmcv 2.2. Anything else imported from it still raises ModuleNotFoundError, which
    MMPose's optional-dependency checks already handle.
    """
    try:
        import mmdet  # noqa: F401
        return
    except (ImportError, AssertionError):
        pass

    def _unavailable(*args, **kwargs):
        raise RuntimeError('mmdet is not installed; this code path needs it.')

    package = types.ModuleType('mmdet')
    utils = types.ModuleType('mmdet.utils')
    utils.ConfigType = dict
    utils.reduce_mean = _unavailable
    package.utils = utils
    sys.modules.update({'mmdet': package, 'mmdet.utils': utils})


def _alias_xtcocotools() -> None:
    """Serve ``xtcocotools`` from ``pycocotools`` when the fork is not installed.

    xtcocotools has no Windows wheel for Python 3.12. The only API difference MMPose uses
    is the constructor's ``sigmas``/``use_area``. pycocotools always normalises OKS by
    ground-truth area, which is ``use_area=True``, the MMPose default. Agreement is
    checked empirically against published AP in the paper's sanity table.
    """
    try:
        import xtcocotools  # noqa: F401
        return
    except ImportError:
        pass

    import numpy as np
    from pycocotools import coco as pc_coco
    from pycocotools import cocoeval as pc_eval
    from pycocotools import mask as pc_mask

    class COCOeval(pc_eval.COCOeval):
        def __init__(self, cocoGt=None, cocoDt=None, iouType='keypoints',
                     sigmas=None, use_area=True):
            if not use_area:
                raise NotImplementedError('use_area=False needs real xtcocotools')
            super().__init__(cocoGt, cocoDt, iouType)
            if sigmas is not None:
                self.params.kpt_oks_sigmas = np.array(sigmas)

    package = types.ModuleType('xtcocotools')
    package.__path__ = []
    coco_mod = types.ModuleType('xtcocotools.coco')
    coco_mod.COCO = pc_coco.COCO
    eval_mod = types.ModuleType('xtcocotools.cocoeval')
    eval_mod.COCOeval = COCOeval
    package.coco, package.cocoeval, package.mask = coco_mod, eval_mod, pc_mask
    sys.modules.update({
        'xtcocotools': package,
        'xtcocotools.coco': coco_mod,
        'xtcocotools.cocoeval': eval_mod,
        'xtcocotools.mask': pc_mask,
    })


def _stub_mmcv_ops() -> None:
    try:
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                'This op needs full mmcv, which has no wheel for this PyTorch. '
                'It is only used by models outside this project (e.g. EDPose).')

    stub = types.ModuleType('mmcv.ops')
    for name in ('MultiScaleDeformableAttention', 'DeformConv2d',
                 'ModulatedDeformConv2d', 'RoIAlign', 'nms', 'batched_nms'):
        setattr(stub, name, _Unavailable)
    sys.modules['mmcv.ops'] = stub
    # Drop half-imported submodules left behind by the failed import above.
    for key in [k for k in sys.modules if k.startswith('mmcv.ops.')]:
        del sys.modules[key]


apply()
