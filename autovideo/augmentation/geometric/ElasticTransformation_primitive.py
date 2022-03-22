'''
Copyright 2021 D3M Team
Copyright (c) 2021 DATA Lab at Texas A&M University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from d3m import container
from d3m.metadata import hyperparams
import imgaug.augmenters as iaa
import typing
from autovideo.utils import construct_primitive_metadata
from autovideo.base.augmentation_base import AugmentationPrimitiveBase

__all__ = ('ElasticTransformationPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    alpha = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.0, 40.0),
        description="Strength of the distortion field. Higher values mean that pixels are moved further with respect to the distortion field’s direction. Set this to around 10 times the value of sigma for visible effects.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    sigma = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(4.0, 8.0),
        description="Standard deviation of the gaussian kernel used to smooth the distortion fields",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    order = hyperparams.Hyperparameter[typing.Union[int,list]](
        default=1,
        description="interpolation order to use",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    cval = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0,255),
        description=" The constant value to use when filling in newly created pixels.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class ElasticTransformationPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Transform images by moving pixels locally around using displacement fields..
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_ElasticTransformation")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        alpha = self.hyperparams["alpha"]
        sigma = self.hyperparams["sigma"]
        seed = self.hyperparams["seed"]
        order = self.hyperparams["order"]
        cval = self.hyperparams["cval"]
        return iaa.ElasticTransformation(alpha=alpha, sigma=sigma, seed=seed, order=order, cval=cval)

