# PointSynthesis

Currently under development

## Changelog

- 2019.09.02 `Jianwei Jiang`: Add `git`, unit test and `__init__.py`. Complete unittest for `transform_clip_feature` and `transform_scaling`.
- 2019.09.03 `Jianwei Jiang`: Add rotation and sampling transform test
- 2019.09.04 `Jianwei Jiang`: Add `XConvLayer` for x-convolution in PointCNN
- 2019.09.06 `Jianwei Jiang`: Refine `XConvLayer`, add several computation utility
- 2019.09.09 `Jianwei Jiang`: Add `FeatureReshape` layer, initialize the function of generating complete network from 
single configuration file: `modelutil.net_from_conf` (not yet completed)