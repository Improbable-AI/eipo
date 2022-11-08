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

import curses
import random

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
    #     '#': impassable walls.            'a': bouncing object A.
    #     'P': player starting location.    'b': fixed object B.
    #     ' ': boring old maze floor.       'c': fixed object C.
    #                                       'd': fixed object D.
    #                                       'e': fixed object E.
    #
    # Room layout:
    # 5 4
    # 0 3
    # 1 2

    # Maze #0: (paper: maze environment)
    ['#####################',
     '#     e      d      #',
     '#                   #',
     '#################   #',
     '#################   #',
     '#          ###   c  #',
     '#         ###       #',
     '#    P  ###        ##',
     '#    #########      #',
     '#    #              #',
     '#   ## a          b #',
     '#                   #',
     '#####################'],
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '@': (999, 862, 110),  # Shimmering golden coins
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'a': (999, 0, 780),    # Patroller A
             'b': (145, 987, 341),  # Patroller B
             'c': (987, 623, 145),  # Patroller C
             'd': (987, 623, 145),  # Patroller D
             'e': (987, 623, 145)}  # Patroller E

COLOUR_BG = {'@': (0, 0, 0)}  # So the coins look like @ and not solid blocks.

ENEMIES = {'a', 'b', 'c', 'd', 'e'} # Globally accessible set of sprites

# Empty coordinates corresponding to each numbered room (width 1 passageways not blocked)
ROOMS = {
  0 : [
       [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10],
       [6, 1], [6, 2], [6, 3], [6, 4],         [6, 6], [6, 7], [6, 8], [6, 9],
               [7, 2], [7, 3],                 [7, 6], [7, 7],
      ],
  1 : [
               [9, 2], [9, 3], [9, 4],         [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11],
               [10, 1], [10, 2], [10, 3],                 [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11],
               [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11],
      ],
  2 : [
               [9, 12], [9, 13], [9, 14], [9, 15], [9, 16], [9, 17], [9, 18], [9, 19],
               [10, 12], [10, 13], [10, 14], [10, 15], [10, 16], [10, 17], [10, 18], [10, 19],
               [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], [11, 18], [11, 19],
      ],
  3 : [
       [5, 14], [5, 15], [5, 16], [5, 17], [5, 18], [5, 19],
       [6, 13], [6, 14], [6, 15], [6, 16], [6, 17], [6, 18], [6, 19],
       [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19],
      ],
  4 : [
       [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18], [1, 19],
       [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17], [2, 18], [2, 19],
      ],
  5 : [
       [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9],
       [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
      ],
}

def make_game(level):
  """Builds and returns a mazee game for the selected level."""
  maze_ascii = MAZES_ART[level]

  # change location of fixed objects
  for row in range(9, 12):
    if 'b' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('b', ' ', 1)
  new_coord = random.sample(ROOMS[2], 1)[0]
  maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'b' + maze_ascii[new_coord[0]][new_coord[1]+1:]

  for row in range(5, 8):
    if 'c' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('c', ' ', 1)
  new_coord = random.sample(ROOMS[3], 1)[0]
  maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'c' + maze_ascii[new_coord[0]][new_coord[1]+1:]

  for row in range(1, 3):
    if 'd' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('d', ' ', 1)
  new_coord = random.sample(ROOMS[4], 1)[0]
  maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'd' + maze_ascii[new_coord[0]][new_coord[1]+1:]

  for row in range(1, 3):
    if 'e' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('e', ' ', 1)
  new_coord = random.sample(ROOMS[5], 1)[0]
  maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'e' + maze_ascii[new_coord[0]][new_coord[1]+1:]

  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'a': WhiteNoiseObject1,
          'b': FixedObject,
          'c': FixedObject,
          'd': FixedObject,
          'e': FixedObject},
      # drapes={
      #     '@': CashDrape},
      update_schedule=['P', 'a', 'b', 'c', 'd', 'e'],
      z_order='abcdeP')


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
        corner, position, character, impassable='#abcde')

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

class WhiteNoiseObject1(prefab_sprites.MazeWalker):
  """Randomly sample direction from left/right/up/down"""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(WhiteNoiseObject1, self).__init__(corner, position, character, impassable='#')
    # Initialize empty space in surrounding radius.
    self._empty_coords = ROOMS[1]

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.
    self._teleport(self._empty_coords[np.random.choice(len(self._empty_coords))])

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
