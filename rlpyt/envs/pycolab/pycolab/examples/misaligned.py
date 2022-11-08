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
    # 'a', 'b', 'c', 'd' and 'e'. I guess if you really don't want them in your maze
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': fixed object A.
    #     'P': player starting location.    'b': white noise object B.
    #     ' ': boring old maze floor.
    #
    # Room layout:
    #   6  
    #   5
    #   4
    #   3  
    #   2
    #   1
    #   0 7 8 9 10 11 12 

    # Maze #0: (paper: 5 rooms environment)
    ['######################################################',
    '######################################################',
    '######################################################',
    '######################################################',
    '######################################################',
    '######################################################',
    '######     ###########################################',
    '######      ##########################################',
    '######  c   ##########################################',
    '######      ##########################################',
    '#######     ##########################################',
    '##########  ##########################################',
    '#######     ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######     ###########################################',
    '######  ##############################################',
    '######     ###########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '#######     ##########################################',
    '##########  ##########################################',
    '#######     ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######     ###########################################',
    '######  ##############################################',
    '######     ###########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '#######     ##########################################',
    '##########  ##########################################',
    '#######     ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######      ##########################################',
    '######     ###########################################',
    '######  ##############################################',
    '######     ###         ###         ###         #######',
    '######      #           #           #           ######',
    '######      #     #     #     #     #     #     ######',
    '######  P   #     #  a  #     #  b  #     #     ######',
    '######            #           #           #     ######',
    '######           ###         ###         ###    ######',
    '######################################################',
    '######################################################',
    '######################################################',
    '######################################################',
    '######################################################',
    '######################################################']
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'a': (999, 0, 780),    # Patroller A
             'b': (145, 987, 341),  # Patroller B
             'c': (55, 55, 55)}     # Patroller C

COLOUR_BG = {'@': (0, 0, 0)}  # So the coins look like @ and not solid blocks.

ENEMIES = {'a', 'b', 'c'} # Globally accessible set of sprites

# Empty coordinates corresponding to each numbered room (width 1 passageways not blocked)
ROOMS = {
    0: [[42, 6], [42, 7], [42, 8], [42, 9], [42, 10], [43, 6], [43, 7], [43, 8], [43, 9], [43, 10], [43, 11], [44, 6], [44, 7], [44, 8], [44, 9], [44, 10], [44, 11], [45, 6], [45, 7], [45, 8], [45, 9], [45, 10], [45, 11], [46, 6], [46, 7], [46, 8], [46, 9], [46, 10], [46, 11], [47, 6], [47, 7], [47, 8], [47, 9], [47, 10], [47, 11]], 
    1: [[36, 7], [36, 8], [36, 9], [37, 6], [37, 7], [37, 8], [37, 9], [37, 10], [37, 11], [38, 6], [38, 7], [38, 8], [38, 9], [38, 10], [38, 11], [39, 6], [39, 7], [39, 8], [39, 9], [39, 10], [39, 11], [40, 8], [40, 9], [40, 10]], 
    2: [[30, 8], [30, 9], [30, 10], [31, 6], [31, 7], [31, 8], [31, 9], [31, 10], [31, 11], [32, 6], [32, 7], [32, 8], [32, 9], [32, 10], [32, 11], [33, 6], [33, 7], [33, 8], [33, 9], [33, 10], [33, 11], [34, 7], [34, 8], [34, 9]], 
    3: [[24, 7], [24, 8], [24, 9], [25, 6], [25, 7], [25, 8], [25, 9], [25, 10], [25, 11], [26, 6], [26, 7], [26, 8], [26, 9], [26, 10], [26, 11], [27, 6], [27, 7], [27, 8], [27, 9], [27, 10], [27, 11], [28, 8], [28, 9], [28, 10]], 
    4: [[18, 8], [18, 9], [18, 10], [19, 6], [19, 7], [19, 8], [19, 9], [19, 10], [19, 11], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11], [21, 6], [21, 7], [21, 8], [21, 9], [21, 10], [21, 11], [22, 7], [22, 8], [22, 9]], 
    5: [[12, 7], [12, 8], [12, 9], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [16, 8], [16, 9], [16, 10]], 
    6: [[6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [10, 7], [10, 8], [10, 9]], 
    7: [[42, 14], [42, 15], [42, 16], [43, 13], [43, 14], [43, 15], [43, 16], [44, 13], [44, 14], [44, 15], [44, 16], [44, 17], [45, 13], [45, 14], [45, 15], [45, 16], [45, 17], [46, 14], [46, 15], [46, 16], [46, 17], [47, 14], [47, 15], [47, 16]], 
    8: [[42, 20], [42, 21], [42, 22], [43, 20], [43, 21], [43, 22], [43, 23], [44, 19], [44, 20], [44, 21], [44, 22], [44, 23], [45, 19], [45, 20], [45, 21], [45, 22], [45, 23], [46, 19], [46, 20], [46, 21], [46, 22], [47, 20], [47, 21], [47, 22]], 
    9: [[42, 26], [42, 27], [42, 28], [43, 25], [43, 26], [43, 27], [43, 28], [44, 25], [44, 26], [44, 27], [44, 28], [44, 29], [45, 25], [45, 26], [45, 27], [45, 28], [45, 29], [46, 26], [46, 27], [46, 28], [46, 29], [47, 26], [47, 27], [47, 28]], 
    10: [[42, 32], [42, 33], [42, 34], [43, 32], [43, 33], [43, 34], [43, 35], [44, 31], [44, 32], [44, 33], [44, 34], [44, 35], [45, 31], [45, 32], [45, 33], [45, 34], [45, 35], [46, 31], [46, 32], [46, 33], [46, 34], [47, 32], [47, 33], [47, 34]], 
    11: [[42, 38], [42, 39], [42, 40], [43, 37], [43, 38], [43, 39], [43, 40], [44, 37], [44, 38], [44, 39], [44, 40], [44, 41], [45, 37], [45, 38], [45, 39], [45, 40], [45, 41], [46, 38], [46, 39], [46, 40], [46, 41], [47, 38], [47, 39], [47, 40]],
    12: [[42, 44], [42, 45], [42, 46], [43, 44], [43, 45], [43, 46], [43, 47], [44, 43], [44, 44], [44, 45], [44, 46], [44, 47], [45, 43], [45, 44], [45, 45], [45, 46], [45, 47], [46, 43], [46, 44], [46, 45], [46, 46], [46, 47], [47, 44], [47, 45], [47, 46], [47, 47]]
    }

def make_game(level, reward_config):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  maze_ascii = MAZES_ART[level]

  # change location of fixed object in the top room
  for row in range(41, 48):
    if 'a' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('a', ' ', 1)
  new_coord_a = random.sample(ROOMS[7]+ROOMS[8], 1)[0]
  maze_ascii[new_coord_a[0]] = maze_ascii[new_coord_a[0]][:new_coord_a[1]] + 'a' + maze_ascii[new_coord_a[0]][new_coord_a[1]+1:]
  for row in range(41, 48):
    if 'b' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('b', ' ', 1)
  new_coord_b = random.sample(ROOMS[9]+ROOMS[10], 1)[0]
  maze_ascii[new_coord_b[0]] = maze_ascii[new_coord_b[0]][:new_coord_b[1]] + 'b' + maze_ascii[new_coord_b[0]][new_coord_b[1]+1:]

  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'a': ascii_art.Partial(FixedObject, reward=reward_config['a']),
          'b': ascii_art.Partial(FixedObject, reward=reward_config['b']),
          'c': ascii_art.Partial(FixedObject, reward=reward_config['c'])},
      update_schedule=['P', 'a', 'b', 'c'],
      z_order='abcP')

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

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls or objects."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#abc')

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

class FixedObject(plab_things.Sprite):
  """Static object. Doesn't move."""

  def __init__(self, corner, position, character, reward=0.0):
    super(FixedObject, self).__init__(
        corner, position, character)
    self.reward = reward

  def update(self, actions, board, layers, backdrop, things, the_plot):
    mr, mc = self.position
    pr, pc = things['P'].position
    if abs(pr-mr) <= 2 and abs(pc-mc) <= 2:
      the_plot.add_reward(self.reward)
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
