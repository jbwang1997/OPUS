import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class FinalFormatting:

    def __init__(self, keys=None, meta_keys=None, test_mode=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.test_mode = test_mode
    
    def default_format(self, results):
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        
        if 'points' in results:
            results['points'] = DC(to_tensor(results['points']))
    
    def collect(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data
    
    def __call__(self, results):
        self.default_format(results)
        results = self.collect(results)
        # To mimic the operation in MultiScaleFlipAug3D
        if self.test_mode:
            for key, value in results.items():
                results[key] = [value]
        return results


        

    