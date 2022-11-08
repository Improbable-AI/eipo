from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab.examples import (maze, 
                              fiveroom, 
                              fiveroom_whitenoise,
                              fiveroom_flipped,
                              fiveroom_all,
                              fiveroom_long,
                              fiveroom_longunpadded,
                              fiveroom_longwide,
                              fiveroom_longext,
                              fiveroom_noobj,
                              fiveroom_oneobj,
                              fiveroom_onewhite,
                              fiveroom_randomfixed, 
                              fiveroom_bouncing,
                              fiveroom_brownian,
                              fiveroom_moveable,
                              fiveroom_moveable_stoch,
                              fiveroom_extint,
                              fiveroom_moveable_brownian,
                              fiveroomlarge,
                              fiveroomlarge_extint,
                              fiveroomlarge_weather,
                              fiveroomlarge_enemy,
                              fiveroomlarge_whitenoise,
                              fiveroomlargetext,
                              fiveroomlargetext_whitenoise,
                              fiveroomlarge_randomfixed,
                              fiveroomlargetext_randomfixed,
                              fiveroomlarge_moveable,
                              fiveroomlarge_moveable_ext,
                              fiveroomlargetext_moveable,
                              fiveroomlarge_moveable_stoch,
                              fiveroomlarge_moveable_stoch_ext,
                              fiveroomlargetext_moveable_stoch,
                              fiveroomlarge_moveable_brownian,
                              fiveroomlargetext_moveable_brownian,
                              fiveroomlarge_brownian,
                              fiveroomlargetext_brownian,
                              fiveroomlarge_all,
                              fiveroomlarge_allext,
                              fiveroomlarge_all_stoch,
                              fiveroomlarge_all_stochext,
                              fiveroomlargetext_all,
                              eightroom,
                              eightroom_rotated,
                              eightroom_ext,
                              eightroom_weather,
                              eightroomlarge_ext,
                              eightroomlarge_weather,
                              eightroomhard_ext,
                              eightroomhard_weather,
                              piano_long,
                              aligned,
                              misaligned
                              )
from pycolab import cropping
from . import pycolab_env

#############################################################################################################################
######################################################## FIVE ROOM ##########################################################
#############################################################################################################################
class FiveRoom(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoom, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomWhitenoise(pycolab_env.PyColabEnv):
    """A 5 room map with whitenoise background, and a fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomWhitenoise, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=224,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_whitenoise.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomFlipped(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1, with the rooms flipped.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomFlipped, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_flipped.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomAll(pycolab_env.PyColabEnv):
    """Map with all four objects (white noise, brownian, Fixed, moveable)
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomAll, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=221,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_all.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomLong(pycolab_env.PyColabEnv):
    """A 5 room variant with a long corridor leading to another room.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomLong, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=236,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,31))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_long.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomLongunpadded(pycolab_env.PyColabEnv):
    """5 room map with a long corridor, with the sides not padded with walls.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomLongunpadded, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=236,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,31))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_longunpadded.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomLongwide(pycolab_env.PyColabEnv):
    """5 room maze with a wider long corridor.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomLongwide, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=208,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,31))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_longwide.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomLongExt(pycolab_env.PyColabEnv):
    """5 room maze with a long corridor and an extrinsic reward in the far room.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0}):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomLongExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=208,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,31))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_longext.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomNoobj(pycolab_env.PyColabEnv):
    """5 room map with no objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomNoobj, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=224,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,24))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_noobj.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomOneobj(pycolab_env.PyColabEnv):
    """5 room map with a single fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomOneobj, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=224,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_oneobj.make_game(self.level)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomOnewhite(pycolab_env.PyColabEnv):
    """5 room map with a white noise teleporter.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomOnewhite, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=224,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_onewhite.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []


class FiveRoomRandomfixed(pycolab_env.PyColabEnv):
    """5 room map with a randomly relocating fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomRandomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_randomfixed.make_game(self.level)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomBouncing(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 3.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomBouncing, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=222,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_bouncing.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomBrownian(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 4.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomBrownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class Maze(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 5.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(Maze, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=150.,
            color_palette=0,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(13,21))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return maze.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomMoveable(pycolab_env.PyColabEnv):
    """A 5 room environment with a controllable object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomMoveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_moveable.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomMoveableBrownian(pycolab_env.PyColabEnv):
    """A 5 room environment with a controllable object and a brownian object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomMoveableBrownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_moveable_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomMoveableStoch(pycolab_env.PyColabEnv):
    """A 5 room environment with a stochastic controllable object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomMoveableStoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_moveable_stoch.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomExtInt(pycolab_env.PyColabEnv):
    """5 room map with an intrinsic attractor and an extrinsic attractor.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':0.0, 'b':1.0}):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects 
        self.obs_type = obs_type
        super(FiveRoomExtInt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(19,19))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroom_extint.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class PianoLong(pycolab_env.PyColabEnv):
    """Piano map with no objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(PianoLong, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=1516,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(18,113))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return piano_long.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

#############################################################################################################################
###################################################### 5 ROOM LARGE #########################################################
#############################################################################################################################
class FiveRoomXL(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXL, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLExtInt(pycolab_env.PyColabEnv):
    """5 room map with an intrinsic attractor and an extrinsic attractor.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':0.0, 'b':1.0}):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects 
        self.obs_type = obs_type
        super(FiveRoomXLExtInt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_extint.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []


class FiveRoomXLEnemy1(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a bouncing enemy.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0,'b':-2.0}):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLEnemy1, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_enemy.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLEnemy2(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a bouncing enemy.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0,'b':-0.5}):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLEnemy2, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_enemy.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLWeather(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a changing central background color.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.0,
                 reward_config={'a':1.0,'b':0.0,'c':0.0,'d':0.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLWeather, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=913,
            color_palette=3,
            reward_switch=['a', 'b', 'c', 'd'],
            reward_config=reward_config,
            switch_perturbations=[(-80., -80., 70.),(-65., 40., -65.),(-40., -50., 0.),(40., -65., -65.)],
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_weather.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLText(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment, with textures.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLText, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=891,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLWhitenoise(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a whitenoise background room.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLWhitenoise, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=916,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_whitenoise.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLTextWhitenoise(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with textures and a whitenoise background room.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLTextWhitenoise, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=892,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext_whitenoise.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLRandomfixed(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a fixed object that moves
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLRandomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_randomfixed.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLTextRandomfixed(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with textures and a randomly initialized fixed.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLTextRandomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=891,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext_randomfixed.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLMoveable(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a controllable object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLMoveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_moveable.make_game(self.level)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLMoveableExt(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a controllable object and extrinsic target.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'e':1.0}):
        self.level = level
        self.objects = ['e']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLMoveableExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_moveable_ext.make_game(self.level, reward_config)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLTextMoveable(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a controllable object
    and textures.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLTextMoveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=891,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext_moveable.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLMoveableStoch(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a stochastic controllable.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLMoveableStoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_moveable_stoch.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLMoveableStochExt(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a stochastic controllable
    and an extrinsic reward target.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'e':1.0}):
        self.level = level
        self.objects = ['e']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLMoveableStochExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_moveable_stoch_ext.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLTextMoveableStoch(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a stochastic controllable and
    textured background.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLTextMoveableStoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=891,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext_moveable_stoch.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLMoveableBrownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a controllable object
    and a brownian object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLMoveableBrownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_moveable_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLTextMoveableBrownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with textured walls and
    a controllable object/brownian object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLTextMoveableBrownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=891,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext_moveable_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLBrownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a brownian object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLBrownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=915,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLTextBrownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with wall textures and
    a brownian object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLTextBrownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=891,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLAll(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a brownian, a fixed,
    a controllable, and a teleporter object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLAll, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=913,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_all.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLAllExt(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a brownian,
    a fixed, a controllable, and a teleporter object. An extrinsic
    reward target is in the top room.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLAllExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=913,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_allext.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLAllStoch(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a brownian,
    a teleporter, a fixed object, and a stochastic controllable.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLAllStoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=913,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_all_stoch.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLAllStochExt(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a brownian,
    a fixed, a stochastic controllable, and a teleporter object. An extrinsic
    reward target is in the top room. 
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLAllStochExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=913,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlarge_all_stochext.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class FiveRoomXLTextAll(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment with a brownian,
    a teleporter, a controllable, and a fixed object. Extra wall
    textures included.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(FiveRoomXLTextAll, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            visitable_states=889,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return fiveroomlargetext_all.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

#############################################################################################################################
####################################################### EIGHT ROOM ##########################################################
#############################################################################################################################
class EightRoom(pycolab_env.PyColabEnv):
    """An eight room environment with 8 fixed object, each
    with the same color.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoom, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=633,
            color_palette=2,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroom.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomExt(pycolab_env.PyColabEnv):
    """An eight room environment with 8 fixed object, each with the
    same color. One object gives an extrinsic reward.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'d':1.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=633,
            color_palette=2,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroom_ext.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomRgb(pycolab_env.PyColabEnv):
    """An eight room environment with 8 fixed objects. Each object is
    a different color, using a color palette where each color averages
    to 60 across channels.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomRgb, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=633,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroom.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomDiff(pycolab_env.PyColabEnv):
    """An eight room environment with 8 fixed objects, each
    with a different color.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomDiff, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=633,
            color_palette=0,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroom.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomDiffRotated(pycolab_env.PyColabEnv):
    """An eight room environment with 8 fixed objects, each
    with a different color, but rotated to new rooms.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomDiffRotated, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=633,
            color_palette=0,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroom_rotated.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomDiffWhite(pycolab_env.PyColabEnv):
    """An eight room environment with 8 fixed objects, each
    with a different color, but rotated to new rooms.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config=dict()):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomDiffWhite, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=633,
            color_palette=4,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroom_rotated.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []


class EightRoomWeather(pycolab_env.PyColabEnv):
    """An eight room environment with 8 fixed objects, and a
    central room that changes color and switches the source of
    extrinsic reward.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0,'b':0.0,'c':0.0,'d':0.0,'e':0.0,'f':0.0,'g':0.0,'h':0.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomWeather, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=633,
            color_palette=3,
            reward_switch=['a','b','c','d','e','f','g','h'],
            reward_config=reward_config,
            switch_perturbations=[(-45.,-25.,-20.),(-20.,-20.,-40.),(-70.,20.,-40.),(-60.,-60.,30.),(-90.,0.,0.),(0.,-90.,0.),(-20.,-20.,-40.),(-20.,-80.,10.)],
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroom_weather.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomXLExt(pycolab_env.PyColabEnv):
    """A large eight room environment with 8 fixed objects, and a
    central room that changes color and switches the source of
    extrinsic reward.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':0.0,'b':0.0,'c':0.0,'d':1.0,'e':0.0,'f':0.0,'g':0.0,'h':0.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomXLExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=952,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            switch_perturbations=[],
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroomlarge_ext.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomXLWeather(pycolab_env.PyColabEnv):
    """A large eight room environment with 8 fixed objects, and a
    central room that changes color and switches the source of
    extrinsic reward.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0,'b':0.0,'c':0.0,'d':0.0,'e':0.0,'f':0.0,'g':0.0,'h':0.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomXLWeather, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=952,
            color_palette=3,
            reward_switch=['a','b','c','d','e','f','g','h'],
            reward_config=reward_config,
            switch_perturbations=[(-45.,-25.,-20.),(-20.,-20.,-40.),(-70.,20.,-40.),(-60.,-60.,30.),(-90.,0.,0.),(0.,-90.,0.),(-20.,-20.,-40.),(-20.,-80.,10.)],
            dimensions=(42,42))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroomlarge_weather.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomHardExt(pycolab_env.PyColabEnv):
    """A larger eight room environment with 8 fixed objects, with one
    object that has extrinsic reward.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':0.0,'b':0.0,'c':0.0,'d':1.0,'e':0.0,'f':0.0,'g':0.0,'h':0.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomHardExt, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=5504,
            color_palette=3,
            reward_switch=[],
            reward_config=reward_config,
            switch_perturbations=[],
            dimensions=(85,85))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroomhard_ext.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class EightRoomHardWeather(pycolab_env.PyColabEnv):
    """A larger eight room environment with 8 fixed objects, and a
    central room that changes color and switches the source of
    extrinsic reward.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':1.0,'b':0.0,'c':0.0,'d':0.0,'e':0.0,'f':0.0,'g':0.0,'h':0.0}):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects
        self.obs_type = obs_type
        super(EightRoomHardWeather, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=5504,
            color_palette=3,
            reward_switch=['a','b','c','d','e','f','g','h'],
            reward_config=reward_config,
            switch_perturbations=[(-45.,-25.,-20.),(-20.,-20.,-40.),(-70.,20.,-40.),(-60.,-60.,30.),(-90.,0.,0.),(0.,-90.,0.),(-20.,-20.,-40.),(-20.,-80.,10.)],
            dimensions=(85,85))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return eightroomhard_weather.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class Aligned(pycolab_env.PyColabEnv):
    """Aligned double corridor map
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':0.0, 'b':0.0, 'c':1.0}):
        self.level = level
        self.objects = ['a', 'b', 'c']
        self.state_layer_chars = ['#'] + self.objects 
        self.obs_type = obs_type
        super(Aligned, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=395,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(54,54))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return aligned.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class Misaligned(pycolab_env.PyColabEnv):
    """Aligned double corridor map
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 reward_config={'a':0.0, 'b':0.0, 'c':1.0}):
        self.level = level
        self.objects = ['a', 'b', 'c']
        self.state_layer_chars = ['#'] + self.objects 
        self.obs_type = obs_type
        super(Misaligned, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1),
            act_null_value=4,
            visitable_states=395,
            color_palette=1,
            reward_switch=[],
            reward_config=reward_config,
            dimensions=(54,54))

    def make_game(self, reward_config):
        self._croppers = self.make_croppers()
        return misaligned.make_game(self.level, reward_config)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []










