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

__all__ = ('ScaleYPrimitive',)

Inputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):

    scale = hyperparams.Hyperparameter[typing.Union[float,tuple,list]](
        default=(0.5, 1.5),
        description=" Analogous to scale in Affine, except that this scale value only affects the x-axis. No dictionary input is allowed.",
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

    mode = hyperparams.Hyperparameter[typing.Union[str,list]](
        default='constant',
        description="Method to use when filling in newly created pixels",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    seed = hyperparams.Constant[int](
        default=0,
        description='Minimum workers to extract frames simultaneously',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )



class ScaleYPrimitive(AugmentationPrimitiveBase[Inputs, Hyperparams]):
    """
    A primitive which Apply affine scaling on the y-axis to input data.
    """

    metadata = construct_primitive_metadata("augmentation", "geometric_ScaleY")

    def _get_function(self):
        """
        set up function and parameter of functions
        """
        scale = self.hyperparams["scale"]
        seed = self.hyperparams["seed"]
        order = self.hyperparams["order"]
        cval = self.hyperparams["cval"]
        mode = self.hyperparams["mode"]
        return iaa.ScaleX(scale=scale, seed=seed, order=order, cval=cval, mode=mode)

