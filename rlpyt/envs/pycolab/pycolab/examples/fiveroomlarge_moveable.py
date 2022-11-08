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
    #   2
    # 1 0 3
    #   4

    # Maze #0: (paper: 5 rooms environment)
    ['##########################################',
    '##########################################',
    '##########################################',
    '######                             #######',
    '#######                           ########',
    '########                         #########',
    '### #####                       ##### ####',
    '###  #####                     #####  ####',
    '###   #####                   #####   ####',
    '###    #####                 #####    ####',
    '###     #####               #####     ####',
    '###      #####             #####      ####',
    '###       #####     e     #####       ####',
    '###        #####         #####        ####',
    '###         #####       #####         ####',
    '###          #####     #####          ####',
    '###           ###### ######           ####',
    '###            ##### #####            ####',
    '###             ##     ##             ####',
    '###             ##     ##             ####',
    '###                 P                 ####',
    '###             ##     ##             ####',
    '###             ##     ##             ####',
    '###            ##### #####            ####',
    '###           ###### ######           ####',
    '###          #####     #####          ####',
    '###         #####       #####         ####',
    '###        #####         #####        ####',
    '###       #####           #####       ####',
    '###      #####      b      #####      ####',
    '###     #####               #####     ####',
    '###    #####                 #####    ####',
    '###   #####                   #####   ####',
    '###  #####                     #####  ####',
    '### #####                       ##### ####',
    '########                         #########',
    '#######                           ########',
    '######                             #######',
    '##########################################',
    '##########################################',
    '##########################################',
    '##########################################',]
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'e': (99, 140, 140),   # Patroller A
             'b': (145, 987, 341)}  # Patroller B

COLOUR_BG = {'@': (0, 0, 0)}  # Target spot

ENEMIES = {'e', 'b'} # Globally accessible set of sprites

# Empty coordinates corresponding to each numbered room (width 1 passageways not blocked)
ROOMS = {0: [[18, 18], [18, 19], [18, 20], [18, 21], [18, 22], [19, 18], [19, 19], [19, 20], [19, 21], [19, 22], [20, 18], [20, 19], [20, 20], [20, 21], [20, 22], [21, 18], [21, 19], [21, 20], [21, 21], [21, 22], [22, 18], [22, 19], [22, 20], [22, 21], [22, 22]], 
1: [[9, 5], [10, 5], [10, 6], [11, 5], [11, 6], [11, 7], [12, 5], [12, 6], [12, 7], [12, 8], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [17, 5], [17, 6], [17, 7], [17, 8], [17, 9], [17, 10], [17, 11], [17, 12], [17, 13], [18, 5], [18, 6], [18, 7], [18, 8], [18, 9], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [19, 5], [19, 6], [19, 7], [19, 8], [19, 9], [19, 10], [19, 11], [19, 12], [19, 13], [19, 14], [20, 5], [20, 6], [20, 7], [20, 8], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [21, 5], [21, 6], [21, 7], [21, 8], [21, 9], [21, 10], [21, 11], [21, 12], [21, 13], [21, 14], [22, 5], [22, 6], [22, 7], [22, 8], [22, 9], [22, 10], [22, 11], [22, 12], [22, 13], [22, 14], [23, 5], [23, 6], [23, 7], [23, 8], [23, 9], [23, 10], [23, 11], [23, 12], [23, 13], [24, 5], [24, 6], [24, 7], [24, 8], [24, 9], [24, 10], [24, 11], [24, 12], [25, 5], [25, 6], [25, 7], [25, 8], [25, 9], [25, 10], [25, 11], [26, 5], [26, 6], [26, 7], [26, 8], [26, 9], [26, 10], [27, 5], [27, 6], [27, 7], [27, 8], [27, 9], [28, 5], [28, 6], [28, 7], [28, 8], [29, 5], [29, 6], [29, 7], [30, 5], [30, 6], [31, 5]], 
2: [[5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 16], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [5, 22], [5, 23], [5, 24], [5, 25], [5, 26], [5, 27], [5, 28], [5, 29], [5, 30], [5, 31], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [6, 16], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 22], [6, 23], [6, 24], [6, 25], [6, 26], [6, 27], [6, 28], [6, 29], [6, 30], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 20], [7, 21], [7, 22], [7, 23], [7, 24], [7, 25], [7, 26], [7, 27], [7, 28], [7, 29], [8, 12], [8, 13], [8, 14], [8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [8, 20], [8, 21], [8, 22], [8, 23], [8, 24], [8, 25], [8, 26], [8, 27], [8, 28], [9, 13], [9, 14], [9, 15], [9, 16], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 22], [9, 23], [9, 24], [9, 25], [9, 26], [9, 27], [10, 14], [10, 15], [10, 16], [10, 17], [10, 18], [10, 19], [10, 20], [10, 21], [10, 22], [10, 23], [10, 24], [10, 25], [10, 26], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19], [11, 20], [11, 21], [11, 22], [11, 23], [11, 24], [11, 25], [12, 16], [12, 17], [12, 18], [12, 19], [12, 20], [12, 21], [12, 22], [12, 23], [12, 24], [13, 17], [13, 18], [13, 19], [13, 20], [13, 21], [13, 22], [13, 23], [14, 18], [14, 19], [14, 21], [14, 22]], 
3: [[9, 35], [10, 34], [10, 35], [11, 33], [11, 34], [11, 35], [12, 32], [12, 33], [12, 34], [12, 35], [13, 31], [13, 32], [13, 33], [13, 34], [13, 35], [14, 30], [14, 31], [14, 32], [14, 33], [14, 34], [14, 35], [15, 29], [15, 30], [15, 31], [15, 32], [15, 33], [15, 34], [15, 35], [16, 28], [16, 29], [16, 30], [16, 31], [16, 32], [16, 33], [16, 34], [16, 35], [17, 27], [17, 28], [17, 29], [17, 30], [17, 31], [17, 32], [17, 33], [17, 34], [17, 35], [18, 26], [18, 27], [18, 28], [18, 29], [18, 30], [18, 31], [18, 32], [18, 33], [18, 34], [18, 35], [19, 26], [19, 27], [19, 28], [19, 29], [19, 30], [19, 31], [19, 32], [19, 33], [19, 34], [19, 35], [20, 27], [20, 28], [20, 29], [20, 30], [20, 31], [20, 32], [20, 33], [20, 34], [20, 35], [21, 26], [21, 27], [21, 28], [21, 29], [21, 30], [21, 31], [21, 32], [21, 33], [21, 34], [21, 35], [22, 26], [22, 27], [22, 28], [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35], [23, 27], [23, 28], [23, 29], [23, 30], [23, 31], [23, 32], [23, 33], [23, 34], [23, 35], [24, 28], [24, 29], [24, 30], [24, 31], [24, 32], [24, 33], [24, 34], [24, 35], [25, 29], [25, 30], [25, 31], [25, 32], [25, 33], [25, 34], [25, 35], [26, 30], [26, 31], [26, 32], [26, 33], [26, 34], [26, 35], [27, 31], [27, 32], [27, 33], [27, 34], [27, 35], [28, 32], [28, 33], [28, 34], [28, 35], [29, 33], [29, 34], [29, 35], [30, 34], [30, 35], [31, 35]], 
4: [[26, 18], [26, 19], [26, 21], [26, 22], [27, 17], [27, 18], [27, 19], [27, 20], [27, 21], [27, 22], [27, 23], [28, 16], [28, 17], [28, 18], [28, 19], [28, 20], [28, 21], [28, 22], [28, 23], [28, 24], [29, 15], [29, 16], [29, 17], [29, 18], [29, 19], [29, 20], [29, 21], [29, 22], [29, 23], [29, 24], [29, 25], [30, 14], [30, 15], [30, 16], [30, 17], [30, 18], [30, 19], [30, 20], [30, 21], [30, 22], [30, 23], [30, 24], [30, 25], [30, 26], [31, 13], [31, 14], [31, 15], [31, 16], [31, 17], [31, 18], [31, 19], [31, 20], [31, 21], [31, 22], [31, 23], [31, 24], [31, 25], [31, 26], [31, 27], [32, 12], [32, 13], [32, 14], [32, 15], [32, 16], [32, 17], [32, 18], [32, 19], [32, 20], [32, 21], [32, 22], [32, 23], [32, 24], [32, 25], [32, 26], [32, 27], [32, 28], [33, 11], [33, 12], [33, 13], [33, 14], [33, 15], [33, 16], [33, 17], [33, 18], [33, 19], [33, 20], [33, 21], [33, 22], [33, 23], [33, 24], [33, 25], [33, 26], [33, 27], [33, 28], [33, 29], [34, 10], [34, 11], [34, 12], [34, 13], [34, 14], [34, 15], [34, 16], [34, 17], [34, 18], [34, 19], [34, 20], [34, 21], [34, 22], [34, 23], [34, 24], [34, 25], [34, 26], [34, 27], [34, 28], [34, 29], [34, 30], [35, 9], [35, 10], [35, 11], [35, 12], [35, 13], [35, 14], [35, 15], [35, 16], [35, 17], [35, 18], [35, 19], [35, 20], [35, 21], [35, 22], [35, 23], [35, 24], [35, 25], [35, 26], [35, 27], [35, 28], [35, 29], [35, 30], [35, 31]]}


def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  maze_ascii = MAZES_ART[level]

  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'e': MoveableObject,
          'b': WhiteNoiseObject},
      update_schedule=['P', 'e', 'b'],
      z_order='ebP')

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
        corner, position, character, impassable='#')
    self.last_position = None # store last position for moveable object
    self.last_action = None # store last action for moveable object

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, layers  # Unused

    self.last_position = self.position
    self.last_action = actions
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

class WhiteNoiseObject(prefab_sprites.MazeWalker):
  """Randomly sample direction from left/right/up/down"""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(WhiteNoiseObject, self).__init__(corner, position, character, impassable='#P')
    # Initialize empty space in surrounding radius.
    self._empty_coords = ROOMS[4]

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.
    self._teleport(self._empty_coords[np.random.choice(len(self._empty_coords))])

class MoveableObject(prefab_sprites.MazeWalker):
  """Moveable object. Can be pushed by agent."""

  def __init__(self, corner, position, character):
    super(MoveableObject, self).__init__(corner, position, character, impassable='#')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    mr, mc = self.position
    pr, pc = things['P'].last_position
    p_action = things['P'].last_action

    # move up
    if (mc == pc) and (mr - pr == -1) and (p_action == 0):
      moved = self._north(board, the_plot)
      if moved is not None:
        things['P']._south(board, the_plot)

    # move down
    elif (mc == pc) and (mr - pr == 1) and (p_action == 1):
      exiting_room = (self.position == (14, 20))
      if exiting_room == True:
        things['P']._north(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._south(board, the_plot)
        if moved is not None: # obstructed
          things['P']._north(board, the_plot)

    # move right
    elif (mc - pc == 1) and (mr == pr) and (p_action == 3):
      exiting_room = (self.position == (15, 19))
      if exiting_room == True:
        things['P']._west(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._east(board, the_plot)
        if moved is not None: # obstructed
          things['P']._west(board, the_plot)

    # move left
    elif (mc - pc == -1) and (mr == pr) and (p_action == 2):
      exiting_room = (self.position == (15, 21))
      if exiting_room == True:
        things['P']._east(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._west(board, the_plot)
        if moved is not None: # obstructed
          things['P']._east(board, the_plot)

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
