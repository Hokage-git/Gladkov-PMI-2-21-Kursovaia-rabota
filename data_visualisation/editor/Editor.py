from typing import List, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath('D:/Docs/Kursach/stylegan3-editing/utils/'))

from config.config import interfacegan_aligned_edit_paths, interfacegan_unaligned_edit_paths
from models.stylegan3.model import GeneratorType
from models.stylegan3.networks_stylegan3 import Generator
from common import tensor2im, generate_random_transform


class FaceEditor:

    def __init__(self, stylegan_generator: Generator, generator_type=GeneratorType.ALIGNED):
        self.generator = stylegan_generator
        if generator_type == GeneratorType.ALIGNED:
            paths = interfacegan_aligned_edit_paths
        else:
            paths = interfacegan_unaligned_edit_paths

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.interfacegan_directions = {
            'age': torch.from_numpy(np.load(paths['age'])).cuda(),
            'smile': torch.from_numpy(np.load(paths['smile'])).cuda(),
            'asian_eyes': torch.from_numpy(np.load(paths['asian_eyes'])).cuda(),
            'Male': torch.from_numpy(np.load(paths['Male'])).cuda(),
            'pose': torch.from_numpy(np.load(paths['Male'])).cuda(),
            'glasses': torch.from_numpy(np.load(paths['Male'])).cuda(),
        }

    def edit(self, latents: torch.tensor, direction: str, factor: int = 1, factor_range: Optional[Tuple[int, int]] = None,
             user_transforms: Optional[np.ndarray] = None, apply_user_transformations: Optional[bool] = False):
        edit_latents = []
        edit_images = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + f * direction
                edit_image, user_transforms = self._latents_to_image(edit_latent,
                                                                     apply_user_transformations,
                                                                     user_transforms)
                edit_latents.append(edit_latent)
                edit_images.append(edit_image)
        else:
            edit_latents = latents + factor * direction
            edit_images, _ = self._latents_to_image(edit_latents, apply_user_transformations)
        return edit_images, edit_latents
    
    def edit_more_then_one(self, latents: torch.Tensor, directions: List[str], factor_ranges: dict[str, Tuple[float, float]],
                            apply_user_transformations: bool = True, num_images: int = 5, direction_strength: List[float] = [1.0]) -> Tuple[List[Image.Image], List[torch.Tensor]]:
        edited_latents = []
        edited_images = []

        if len(direction_strength) != len(directions):
            raise ValueError("Длинна списков не совпадает")

        for direction, factor_range, strength in zip(directions, factor_ranges.values(), direction_strength):
            direction = self.interfacegan_directions[direction]

            for factor in np.linspace(factor_range[0], factor_range[1], num_images):
                edit_latent = latents + (factor * strength) * direction
                if len(edited_latents) == 0:
                    edited_latent = edit_latent
                    edited_image, _ = self._latents_to_image(edited_latent, apply_user_transformations)
                    edited_latents.append(edited_latent)
                    edited_images.append(edited_image)
                else:
                    edited_latent = edited_latents[-1] + (factor * strength) * direction
                    edited_image, _ = self._latents_to_image(edited_latent, apply_user_transformations)
                    edited_latents[-1] = edited_latent
                    edited_images[-1] = edited_image

        return edited_images, edited_latents

    def _latents_to_image(self, all_latents: torch.tensor, apply_user_transformations: bool = False,
                          user_transforms: Optional[torch.tensor] = None):
        with torch.no_grad():
            if apply_user_transformations:
                if user_transforms is None:
                    # if no transform provided, generate a random transformation
                    user_transforms = generate_random_transform(translate=0.3, rotate=25)
                # apply the user-specified transformation
                if type(user_transforms) == np.ndarray:
                    user_transforms = torch.from_numpy(user_transforms)
                self.generator.synthesis.input.transform = user_transforms.cuda().float()
            # generate the images
            images = self.generator.synthesis(all_latents, noise_mode='const')
            images = [tensor2im(image) for image in images]
        return images, user_transforms
