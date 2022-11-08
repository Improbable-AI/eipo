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
    '##########################################',
    '##########################################',
    '##########################################',
    '################         #################',
    '############ ###         ### #############',
    '##########   ###         ###   ###########',
    '##########   ###         ###   ###########',
    '########      ###   e   ###      #########',
    '########      ###       ###      #########',
    '######         ###     ###         #######',
    '######         ###     ###         #######',
    '#####       b   ###   ###    d      ######',
    '########        ###   ###        #########',
    '##########       ### ###       ###########',
    '############     ##   ##     #############',
    '####    ######             ######    #####',
    '####      ######         ######      #####',
    '####        ####         ####        #####',
    '####          #           #          #####',
    '####      h         P         a      #####',
    '####          #           #          #####',
    '####        ####         ####        #####',
    '####      ######         ######      #####',
    '####    ######             ######    #####',
    '############     ##   ##     #############',
    '##########       ### ###       ###########',
    '########        ###   ###        #########',
    '#####      c    ###   ###   g       ######',
    '######         ###     ###         #######',
    '######         ###  f  ###         #######',
    '########      ###       ###      #########',
    '########      ###       ###      #########',
    '##########   ###         ###   ###########',
    '##########   ###         ###   ###########',
    '############ ###         ### #############',
    '################         #################',
    '##########################################',
    '##########################################',
    '##########################################',
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
ROOMS = {0: [[16, 17], [16, 18], [16, 19], [16, 20], [16, 21], [16, 22], [16, 23], [16, 24], [17, 16], [17, 17], [17, 18], [17, 19], [17, 20], [17, 21], [17, 22], [17, 23], [17, 24], [17, 25], [18, 16], [18, 17], [18, 18], [18, 19], [18, 20], [18, 21], [18, 22], [18, 23], [18, 24], [18, 25], [19, 16], [19, 17], [19, 18], [19, 19], [19, 20], [19, 21], [19, 22], [19, 23], [19, 24], [19, 25], [20, 16], [20, 17], [20, 18], [20, 19], [20, 20], [20, 21], [20, 22], [20, 23], [20, 24], [20, 25], [21, 16], [21, 17], [21, 18], [21, 19], [21, 20], [21, 21], [21, 22], [21, 23], [21, 24], [21, 25], [22, 16], [22, 17], [22, 18], [22, 19], [22, 20], [22, 21], [22, 22], [22, 23], [22, 24], [22, 25], [23, 16], [23, 17], [23, 18], [23, 19], [23, 20], [23, 21], [23, 22], [23, 23], [23, 24], [23, 25], [24, 16], [24, 17], [24, 18], [24, 19], [24, 20], [24, 21], [24, 22], [24, 23], [24, 24], [24, 25], [25, 17], [25, 18], [25, 19], [25, 20], [25, 21], [25, 22], [25, 23], [25, 24]], 
1: [[5, 18], [5, 19], [5, 20], [5, 21], [5, 22], [5, 23], [6, 18], [6, 19], [6, 20], [6, 21], [6, 22], [6, 23], [7, 18], [7, 19], [7, 20], [7, 21], [7, 22], [7, 23], [8, 18], [8, 19], [8, 20], [8, 21], [8, 22], [8, 23], [9, 18], [9, 19], [9, 20], [9, 21], [9, 22], [9, 23], [10, 18], [10, 19], [10, 20], [10, 21], [10, 22], [11, 18], [11, 19], [11, 20], [11, 21], [11, 22]], 
2: [[7, 28], [8, 28], [8, 29], [9, 27], [9, 28], [9, 29], [9, 30], [10, 26], [10, 27], [10, 28], [10, 29], [10, 30], [10, 31], [11, 26], [11, 27], [11, 28], [11, 29], [11, 30], [11, 31], [11, 32], [12, 25], [12, 26], [12, 27], [12, 28], [12, 29], [12, 30], [12, 31], [12, 32], [12, 33], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [13, 30], [13, 31], [14, 26], [14, 27], [14, 28], [14, 29], [14, 30], [15, 27], [15, 28]], 
3: [[17, 31], [17, 32], [17, 33], [17, 34], [17, 35], [18, 29], [18, 30], [18, 31], [18, 32], [18, 33], [18, 34], [18, 35], [19, 29], [19, 30], [19, 31], [19, 32], [19, 33], [19, 34], [19, 35], [20, 29], [20, 30], [20, 31], [20, 32], [20, 33], [20, 34], [20, 35], [21, 29], [21, 30], [21, 31], [21, 32], [21, 33], [21, 34], [21, 35], [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35], [23, 31], [23, 32], [23, 33], [23, 34], [23, 35]], 
4: [[25, 27], [25, 28], [26, 26], [26, 27], [26, 28], [26, 29], [26, 30], [27, 25], [27, 26], [27, 27], [27, 28], [27, 29], [27, 30], [27, 31], [28, 25], [28, 26], [28, 27], [28, 28], [28, 29], [28, 30], [28, 31], [28, 32], [28, 33], [29, 26], [29, 27], [29, 28], [29, 29], [29, 30], [29, 31], [29, 32], [30, 26], [30, 27], [30, 28], [30, 29], [30, 30], [30, 31], [31, 27], [31, 28], [31, 29], [31, 30], [32, 28], [32, 29], [33, 28]], 
5: [[29, 18], [29, 19], [29, 20], [29, 21], [29, 22], [30, 18], [30, 19], [30, 20], [30, 21], [30, 22], [31, 17], [31, 18], [31, 19], [31, 20], [31, 21], [31, 22], [31, 23], [32, 17], [32, 18], [32, 19], [32, 20], [32, 21], [32, 22], [32, 23], [33, 17], [33, 18], [33, 19], [33, 20], [33, 21], [33, 22], [33, 23], [34, 17], [34, 18], [34, 19], [34, 20], [34, 21], [34, 22], [34, 23], [35, 17], [35, 18], [35, 19], [35, 20], [35, 21], [35, 22], [35, 23]], 
6: [[25, 12], [25, 13], [26, 10], [26, 11], [26, 12], [26, 13], [26, 14], [27, 9], [27, 10], [27, 11], [27, 12], [27, 13], [27, 14], [27, 15], [28, 7], [28, 8], [28, 9], [28, 10], [28, 11], [28, 12], [28, 13], [28, 14], [28, 15], [29, 8], [29, 9], [29, 10], [29, 11], [29, 12], [29, 13], [29, 14], [30, 9], [30, 10], [30, 11], [30, 12], [30, 13], [30, 14], [31, 10], [31, 11], [31, 12], [31, 13], [32, 11], [32, 12], [33, 12]], 
7: [[17, 5], [17, 6], [17, 7], [17, 8], [17, 9], [18, 5], [18, 6], [18, 7], [18, 8], [18, 9], [18, 10], [18, 11], [19, 5], [19, 6], [19, 7], [19, 8], [19, 9], [19, 10], [19, 11], [20, 5], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11], [21, 5], [21, 6], [21, 7], [21, 8], [21, 9], [21, 10], [21, 11], [22, 5], [22, 6], [22, 7], [22, 8], [22, 9], [22, 10], [22, 11], [23, 5], [23, 6], [23, 7], [23, 8], [23, 9]], 
8: [[7, 12], [8, 11], [8, 12], [9, 10], [9, 11], [9, 12], [9, 13], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [15, 12], [15, 13]]}

def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  maze_ascii = MAZES_ART[level]

  # change location of fixed object in all the rooms
  for row in range(4, 37):
    if 'a' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('a', ' ', 1)
      new_coord = random.sample(ROOMS[3], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'a' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'b' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('b', ' ', 1)
      new_coord = random.sample(ROOMS[8], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'b' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'c' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('c', ' ', 1)
      new_coord = random.sample(ROOMS[6], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'c' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'd' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('d', ' ', 1)
      new_coord = random.sample(ROOMS[2], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'd' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'e' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('e', ' ', 1)
      new_coord = random.sample(ROOMS[1], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'e' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'f' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('f', ' ', 1)
      new_coord = random.sample(ROOMS[5], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'f' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'g' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('g', ' ', 1)
      new_coord = random.sample(ROOMS[4], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'g' + maze_ascii[new_coord[0]][new_coord[1]+1:]
    if 'h' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('h', ' ', 1)
      new_coord = random.sample(ROOMS[7], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'h' + maze_ascii[new_coord[0]][new_coord[1]+1:]

  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
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

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls or objects."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#abcdefgh')

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
