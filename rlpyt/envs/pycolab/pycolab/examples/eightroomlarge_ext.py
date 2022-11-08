# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of the environments from 'World Discovery Models'[https://arxiv.org/pdf/1902.07685.pdf]. 
Learn to explore!

This environment uses a simple scrolling mechanism: cropping! As far as the pycolab engine is concerned, 
the game world doesn't scroll at all: it just renders observations that are the size
of the entire map. Only later do "cropper" objects crop out a part of the
observation to give the impression of a moving world/partial observability.

Keys: up, down, left, right - move. q - quit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import curses

import sys
import numpy as np

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the object sprites
    # 'a', 'b', 'c', 'd' and 'e', 'f', 'g', 'h'. I guess if you really don't want them in your maze
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': fixed object A.
    #     'P': player starting location.    'b': fixed object B.
    #     ' ': boring old maze floor.       'c': fixed object C.
    #                                       'd': fixed object D.
    #                                       'e': fixed object E.
    #                                       'f': fixed object F.
    #                                       'g': fixed object G.
    #                                       'h': fixed object H.
    #
    # Room layout:
    # 8 1 2
    # 7 0 3
    # 6 5 4
    ['##########################################',
    '############ ##           ## #############',
    '###########  ##           ##  ############',
    '##########   ##           ##   ###########',
    '#########    ##     a     ##    ##########',
    '########     ##           ##     #########',
    '#######      ###         ###      ########',
    '######       ###         ###       #######',
    '#####         ##         ##         ######',
    '####          ###       ###          #####',
    '###     h      ##       ##      b     ####',
    '##             ###     ###             ###',
    '#               ###   ###               ##',
    '########        ###   ###        #########',
    '##########       ### ###       ###########',
    '#     ######     ##   ##     ######     ##',
    '#        #####             #####        ##',
    '#          #####         #####          ##',
    '#           ####         ####           ##',
    '#             #           #             ##',
    '#    g              P             c     ##',
    '#             #           #             ##',
    '#           ####         ####           ##',
    '#          #####         #####          ##',
    '#        #####             #####        ##',
    '#     ######     ##   ##     ######     ##',
    '##########       ### ###       ###########',
    '########        ###   ###        #########',
    '#               ###   ###               ##',
    '##             ###     ###             ###',
    '###     f      ##       ##     d      ####',
    '####          ###       ###          #####',
    '#####         ##         ##         ######',
    '######       ###         ###       #######',
    '#######      ###         ###      ########',
    '########     ##           ##     #########',
    '#########    ##     e     ##    ##########',
    '##########   ##           ##   ###########',
    '###########  ##           ##  ############',
    '############ ##           ## #############',
    '##########################################',
    '##########################################',]
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'a': (999, 0, 780),    # Patroller A
             'b': (145, 987, 341),   # Patroller B
             'c': (252, 186, 3),
             'd': (3, 240, 252),
             'e': (240, 3, 252),
             'f': (252, 28, 3),
             'g': (136, 3, 252),
             'h': (20, 145, 60)}

COLOUR_BG = {'@': (0, 0, 0)}  # So the coins look like @ and not solid blocks.

ENEMIES = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'} # Globally accessible set of sprites

# Empty coordinates corresponding to each numbered room (width 1 passageways not blocked)
ROOMS = {0: [[16, 17], [16, 18], [16, 19], [16, 20], [16, 21], [16, 22], [16, 23], [17, 16], [17, 17], [17, 18], [17, 19], [17, 20], [17, 21], [17, 22], [17, 23], [17, 24], [18, 16], [18, 17], [18, 18], [18, 19], [18, 20], [18, 21], [18, 22], [18, 23], [18, 24], [19, 16], [19, 17], [19, 18], [19, 19], [19, 20], [19, 21], [19, 22], [19, 23], [19, 24], [20, 16], [20, 17], [20, 18], [20, 19], [20, 20], [20, 21], [20, 22], [20, 23], [20, 24], [21, 16], [21, 17], [21, 18], [21, 19], [21, 20], [21, 21], [21, 22], [21, 23], [21, 24], [22, 16], [22, 17], [22, 18], [22, 19], [22, 20], [22, 21], [22, 22], [22, 23], [22, 24], [23, 16], [23, 17], [23, 18], [23, 19], [23, 20], [23, 21], [23, 22], [23, 23], [23, 24], [24, 17], [24, 18], [24, 19], [24, 20], [24, 21], [24, 22], [24, 23]], 
          1: [[1, 15], [1, 16], [1, 17], [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [1, 23], [1, 24], [1, 25], [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23], [2, 24], [2, 25], [3, 15], [3, 16], [3, 17], [3, 18], [3, 19], [3, 20], [3, 21], [3, 22], [3, 23], [3, 24], [3, 25], [4, 15], [4, 16], [4, 17], [4, 18], [4, 19], [4, 20], [4, 21], [4, 22], [4, 23], [4, 24], [4, 25], [5, 15], [5, 16], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [5, 22], [5, 23], [5, 24], [5, 25], [6, 16], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 22], [6, 23], [6, 24], [7, 16], [7, 17], [7, 18], [7, 19], [7, 20], [7, 21], [7, 22], [7, 23], [7, 24], [8, 16], [8, 17], [8, 18], [8, 19], [8, 20], [8, 21], [8, 22], [8, 23], [8, 24]], 
          2: [[1, 28], [2, 28], [2, 29], [3, 28], [3, 29], [3, 30], [4, 28], [4, 29], [4, 30], [4, 31], [5, 28], [5, 29], [5, 30], [5, 31], [5, 32], [6, 28], [6, 29], [6, 30], [6, 31], [6, 32], [6, 33], [7, 28], [7, 29], [7, 30], [7, 31], [7, 32], [7, 33], [7, 34], [8, 27], [8, 28], [8, 29], [8, 30], [8, 31], [8, 32], [8, 33], [8, 34], [8, 35], [9, 27], [9, 28], [9, 29], [9, 30], [9, 31], [9, 32], [9, 33], [9, 34], [9, 35], [9, 36], [10, 26], [10, 27], [10, 28], [10, 29], [10, 30], [10, 31], [10, 32], [10, 33], [10, 34], [10, 35], [10, 36], [10, 37], [11, 27], [11, 28], [11, 29], [11, 30], [11, 31], [11, 32], [11, 33], [11, 34], [11, 35], [11, 36], [11, 37], [11, 38], [12, 28], [12, 29], [12, 30], [12, 31], [12, 32], [12, 33], [12, 34], [12, 35], [12, 36], [12, 37], [12, 38], [12, 39], [13, 29], [13, 30], [13, 31], [13, 32], [14, 30]], 
          3: [[15, 35], [15, 36], [15, 37], [15, 38], [15, 39], [16, 32], [16, 33], [16, 34], [16, 35], [16, 36], [16, 37], [16, 38], [16, 39], [17, 32], [17, 33], [17, 34], [17, 35], [17, 36], [17, 37], [17, 38], [17, 39], [18, 32], [18, 33], [18, 34], [18, 35], [18, 36], [18, 37], [18, 38], [18, 39], [19, 32], [19, 33], [19, 34], [19, 35], [19, 36], [19, 37], [19, 38], [19, 39], [20, 32], [20, 33], [20, 34], [20, 35], [20, 36], [20, 37], [20, 38], [20, 39], [21, 32], [21, 33], [21, 34], [21, 35], [21, 36], [21, 37], [21, 38], [21, 39], [22, 32], [22, 33], [22, 34], [22, 35], [22, 36], [22, 37], [22, 38], [22, 39], [23, 32], [23, 33], [23, 34], [23, 35], [23, 36], [23, 37], [23, 38], [23, 39], [24, 32], [24, 33], [24, 34], [24, 35], [24, 36], [24, 37], [24, 38], [24, 39], [25, 35], [25, 36], [25, 37], [25, 38], [25, 39]], 
          4: [[26, 30], [27, 29], [27, 30], [27, 31], [27, 32], [28, 28], [28, 29], [28, 30], [28, 31], [28, 32], [28, 33], [28, 34], [28, 35], [28, 36], [28, 37], [28, 38], [28, 39], [29, 27], [29, 28], [29, 29], [29, 30], [29, 31], [29, 32], [29, 33], [29, 34], [29, 35], [29, 36], [29, 37], [29, 38], [30, 26], [30, 27], [30, 28], [30, 29], [30, 30], [30, 31], [30, 32], [30, 33], [30, 34], [30, 35], [30, 36], [30, 37], [31, 27], [31, 28], [31, 29], [31, 30], [31, 31], [31, 32], [31, 33], [31, 34], [31, 35], [31, 36], [32, 27], [32, 28], [32, 29], [32, 30], [32, 31], [32, 32], [32, 33], [32, 34], [32, 35], [33, 28], [33, 29], [33, 30], [33, 31], [33, 32], [33, 33], [33, 34], [34, 28], [34, 29], [34, 30], [34, 31], [34, 32], [34, 33], [35, 28], [35, 29], [35, 30], [35, 31], [35, 32], [36, 28], [36, 29], [36, 30], [36, 31], [37, 28], [37, 29], [37, 30], [38, 28], [38, 29], [39, 28]], 
          5: [[32, 16], [32, 17], [32, 18], [32, 19], [32, 20], [32, 21], [32, 22], [32, 23], [32, 24], [33, 16], [33, 17], [33, 18], [33, 19], [33, 20], [33, 21], [33, 22], [33, 23], [33, 24], [34, 16], [34, 17], [34, 18], [34, 19], [34, 20], [34, 21], [34, 22], [34, 23], [34, 24], [35, 15], [35, 16], [35, 17], [35, 18], [35, 19], [35, 20], [35, 21], [35, 22], [35, 23], [35, 24], [35, 25], [36, 15], [36, 16], [36, 17], [36, 18], [36, 19], [36, 20], [36, 21], [36, 22], [36, 23], [36, 24], [36, 25], [37, 15], [37, 16], [37, 17], [37, 18], [37, 19], [37, 20], [37, 21], [37, 22], [37, 23], [37, 24], [37, 25], [38, 15], [38, 16], [38, 17], [38, 18], [38, 19], [38, 20], [38, 21], [38, 22], [38, 23], [38, 24], [38, 25], [39, 15], [39, 16], [39, 17], [39, 18], [39, 19], [39, 20], [39, 21], [39, 22], [39, 23], [39, 24], [39, 25]], 
          6: [[26, 10], [27, 8], [27, 9], [27, 10], [27, 11], [28, 1], [28, 2], [28, 3], [28, 4], [28, 5], [28, 6], [28, 7], [28, 8], [28, 9], [28, 10], [28, 11], [28, 12], [29, 2], [29, 3], [29, 4], [29, 5], [29, 6], [29, 7], [29, 8], [29, 9], [29, 10], [29, 11], [29, 12], [29, 13], [30, 3], [30, 4], [30, 5], [30, 6], [30, 7], [30, 8], [30, 9], [30, 10], [30, 11], [30, 12], [30, 13], [30, 14], [31, 4], [31, 5], [31, 6], [31, 7], [31, 8], [31, 9], [31, 10], [31, 11], [31, 12], [31, 13], [32, 5], [32, 6], [32, 7], [32, 8], [32, 9], [32, 10], [32, 11], [32, 12], [32, 13], [33, 6], [33, 7], [33, 8], [33, 9], [33, 10], [33, 11], [33, 12], [34, 7], [34, 8], [34, 9], [34, 10], [34, 11], [34, 12], [35, 8], [35, 9], [35, 10], [35, 11], [35, 12], [36, 9], [36, 10], [36, 11], [36, 12], [37, 10], [37, 11], [37, 12], [38, 11], [38, 12], [39, 12]], 
          7: [[15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [16, 1], [16, 2], [16, 3], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [17, 1], [17, 2], [17, 3], [17, 4], [17, 5], [17, 6], [17, 7], [17, 8], [18, 1], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6], [18, 7], [18, 8], [19, 1], [19, 2], [19, 3], [19, 4], [19, 5], [19, 6], [19, 7], [19, 8], [20, 1], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [20, 7], [20, 8], [21, 1], [21, 2], [21, 3], [21, 4], [21, 5], [21, 6], [21, 7], [21, 8], [22, 1], [22, 2], [22, 3], [22, 4], [22, 5], [22, 6], [22, 7], [22, 8], [23, 1], [23, 2], [23, 3], [23, 4], [23, 5], [23, 6], [23, 7], [23, 8], [24, 1], [24, 2], [24, 3], [24, 4], [24, 5], [24, 6], [24, 7], [24, 8], [25, 1], [25, 2], [25, 3], [25, 4], [25, 5]], 
          8: [[1, 12], [2, 11], [2, 12], [3, 10], [3, 11], [3, 12], [4, 9], [4, 10], [4, 11], [4, 12], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [13, 8], [13, 9], [13, 10], [13, 11], [14, 10]]}

def make_game(level, reward_config):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  maze_ascii = MAZES_ART[level]

  # change location of fixed object in all the rooms
  for row in range(1,40):
    if 'a' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('a', ' ', 1)
      new_coord = random.sample(ROOMS[1], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'a' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'b' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('b', ' ', 1)
      new_coord = random.sample(ROOMS[2], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'b' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'c' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('c', ' ', 1)
      new_coord = random.sample(ROOMS[3], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'c' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'd' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('d', ' ', 1)
      new_coord = random.sample(ROOMS[4], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'd' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'e' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('e', ' ', 1)
      new_coord = random.sample(ROOMS[5], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'e' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'f' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('f', ' ', 1)
      new_coord = random.sample(ROOMS[6], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'f' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'g' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('g', ' ', 1)
      new_coord = random.sample(ROOMS[7], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'g' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'h' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('h', ' ', 1)
      new_coord = random.sample(ROOMS[8], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'h' + maze_ascii[new_coord[0]][new_coord[1]+1:]

  a, b, c, d, e, f, g, h = reward_config.values()
  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': ascii_art.Partial(PlayerSprite, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h),
          'a': FixedObject,
          'b': FixedObject,
          'c': FixedObject,
          'd': FixedObject,
          'e': FixedObject,
          'f': FixedObject,
          'g': FixedObject,
          'h': FixedObject},
      update_schedule=['P', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
      z_order='abcdefghP')

def make_croppers(level):
  """Builds and returns `ObservationCropper`s for the selected level.

  We make one cropper for each level: centred on the player. Room
  to add more if needed.

  Args:
    level: level to make `ObservationCropper`s for.

  Returns:
    a list of all the `ObservationCropper`s needed.
  """
  return [
      # The player view.
      cropping.ScrollingCropper(rows=5, cols=5, to_track=['P']),
  ]

class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer."""

  def __init__(self, corner, position, character, a, b, c, d, e, f, g, h):
    """Constructor: just tells `MazeWalker` we can't walk through walls or objects."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#abcdefgh')
    self.a_reward = a
    self.b_reward = b
    self.c_reward = c
    self.d_reward = d
    self.e_reward = e
    self.f_reward = f
    self.g_reward = g
    self.h_reward = h

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # stay put? (Not strictly necessary.)
      self._stay(board, the_plot)
    if actions == 5:    # just quit?
      the_plot.terminate_episode()

    pr, pc = self.position

    ar, ac = things['a'].position
    br, bc = things['b'].position
    cr, cc = things['c'].position
    dr, dc = things['d'].position
    er, ec = things['e'].position
    fr, fc = things['f'].position
    gr, gc = things['g'].position
    hr, hc = things['h'].position
    if ((ar-1) <= pr <= (ar+1)) and ((ac-1) <= pc <= (ac+1)):
      the_plot.add_reward(self.a_reward)
    elif ((br-1) <= pr <= (br+1)) and ((bc-1) <= pc <= (bc+1)):
      the_plot.add_reward(self.b_reward)
    elif ((cr-1) <= pr <= (cr+1)) and ((cc-1) <= pc <= (cc+1)):
      the_plot.add_reward(self.c_reward)
    elif ((dr-1) <= pr <= (dr+1)) and ((dc-1) <= pc <= (dc+1)):
      the_plot.add_reward(self.d_reward)
    elif ((er-1) <= pr <= (er+1)) and ((ec-1) <= pc <= (ec+1)):
      the_plot.add_reward(self.e_reward)
    elif ((fr-1) <= pr <= (fr+1)) and ((fc-1) <= pc <= (fc+1)):
      the_plot.add_reward(self.f_reward)
    elif ((gr-1) <= pr <= (gr+1)) and ((gc-1) <= pc <= (gc+1)):
      the_plot.add_reward(self.g_reward)
    elif ((hr-1) <= pr <= (hr+1)) and ((hc-1) <= pc <= (hc+1)):
      the_plot.add_reward(self.h_reward)

class FixedObject(plab_things.Sprite):
  """Static object. Doesn't move."""

  def __init__(self, corner, position, character):
    super(FixedObject, self).__init__(
        corner, position, character)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.

def main(argv=()):
  level = int(argv[1]) if len(argv) > 1 else 0

  # Build the game.
  game = make_game(level)
  # Build the croppers we'll use to scroll around in it, etc.
  croppers = make_croppers(level)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG,
      croppers=croppers)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)
