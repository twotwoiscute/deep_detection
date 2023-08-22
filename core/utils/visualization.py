#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np


COLORS = ['GoldenRod', 'MediumTurquoise', 'GreenYellow', 'SteelBlue', 'DarkSeaGreen', 'SeaShell', 'LightGrey',
          'IndianRed', 'DarkKhaki', 'LawnGreen', 'WhiteSmoke', 'Peru', 'LightCoral', 'FireBrick', 'OldLace',
          'LightBlue', 'SlateGray', 'OliveDrab', 'NavajoWhite', 'PaleVioletRed', 'SpringGreen', 'AliceBlue', 'Violet',
          'DeepSkyBlue', 'Red', 'MediumVioletRed', 'PaleTurquoise', 'Tomato', 'Azure', 'Yellow', 'Cornsilk',
          'Aquamarine', 'CadetBlue', 'CornflowerBlue', 'DodgerBlue', 'Olive', 'Orchid', 'LemonChiffon', 'Sienna',
          'OrangeRed', 'Orange', 'DarkSalmon', 'Magenta', 'Wheat', 'Lime', 'GhostWhite', 'SlateBlue', 'Aqua',
          'MediumAquaMarine', 'LightSlateGrey', 'MediumSeaGreen', 'SandyBrown', 'YellowGreen', 'Plum', 'FloralWhite',
          'LightPink', 'Thistle', 'DarkViolet', 'Pink', 'Crimson', 'Chocolate', 'DarkGrey', 'Ivory', 'PaleGreen',
          'DarkGoldenRod', 'LavenderBlush', 'SlateGrey', 'DeepPink', 'Gold', 'Cyan', 'LightSteelBlue', 'MediumPurple',
          'ForestGreen', 'DarkOrange', 'Tan', 'Salmon', 'PaleGoldenRod', 'LightGreen', 'LightSlateGray', 'HoneyDew',
          'Fuchsia', 'LightSeaGreen', 'DarkOrchid', 'Green', 'Chartreuse', 'LimeGreen', 'AntiqueWhite', 'Beige',
          'Gainsboro', 'Bisque', 'SaddleBrown', 'Silver', 'Lavender', 'Teal', 'LightCyan', 'PapayaWhip', 'Purple',
          'Coral', 'BurlyWood', 'LightGray', 'Snow', 'MistyRose', 'PowderBlue', 'DarkCyan', 'White', 'Turquoise',
          'MediumSlateBlue', 'PeachPuff', 'Moccasin', 'LightSalmon', 'SkyBlue', 'Khaki', 'MediumSpringGreen',
          'BlueViolet', 'MintCream', 'Linen', 'SeaGreen', 'HotPink', 'LightYellow', 'BlanchedAlmond', 'RoyalBlue',
          'RosyBrown', 'MediumOrchid', 'DarkTurquoise', 'LightGoldenRodYellow', 'LightSkyBlue']

#Overlay mask with transparency on top of the image.
def overlay(image, mask, color, alpha_transparency=0.5):
    for channel in range(3):
        image[:, :, channel] = np.where(mask == 1,
                              image[:, :, channel] *
                              (1 - alpha_transparency) + alpha_transparency * color[channel] * 255,
                              image[:, :, channel])
    return image