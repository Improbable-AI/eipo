from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

def register(id, entry_point, max_episode_steps, kwargs):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        del env_specs[id]
    gym.register(id=id, 
                 entry_point=entry_point, 
                 max_episode_steps=max_episode_steps, 
                 kwargs=kwargs)

register(
    id='5Room-v0',
    entry_point='mazeworld.envs:FiveRoom',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5Room-v1',
    entry_point='mazeworld.envs:FiveRoomFlipped',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomWhitenoise-v0',
    entry_point='mazeworld.envs:FiveRoomWhitenoise',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomAll-v0',
    entry_point='mazeworld.envs:FiveRoomAll',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomMoveable-v0',
    entry_point='mazeworld.envs:FiveRoomMoveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomMoveableStoch-v0',
    entry_point='mazeworld.envs:FiveRoomMoveableStoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomMoveableBrownian-v0',
    entry_point='mazeworld.envs:FiveRoomMoveableBrownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomExtInt-v0',
    entry_point='mazeworld.envs:FiveRoomExtInt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLarge-v0',
    entry_point='mazeworld.envs:FiveRoomXL',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeExtInt-v0',
    entry_point='mazeworld.envs:FiveRoomXLExtInt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeEnemy-v0',
    entry_point='mazeworld.envs:FiveRoomXLEnemy1',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeEnemy-v1',
    entry_point='mazeworld.envs:FiveRoomXLEnemy2',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeWeather-v0',
    entry_point='mazeworld.envs:FiveRoomXLWeather',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeText-v0',
    entry_point='mazeworld.envs:FiveRoomXLText',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeWhitenoise-v0',
    entry_point='mazeworld.envs:FiveRoomXLWhitenoise',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeTextWhitenoise-v0',
    entry_point='mazeworld.envs:FiveRoomXLTextWhitenoise',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeRandomFixed-v0',
    entry_point='mazeworld.envs:FiveRoomXLRandomfixed',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeTextRandomFixed-v0',
    entry_point='mazeworld.envs:FiveRoomXLTextRandomfixed',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeMoveable-v0',
    entry_point='mazeworld.envs:FiveRoomXLMoveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeMoveable-v1',
    entry_point='mazeworld.envs:FiveRoomXLMoveableExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeTextMoveable-v0',
    entry_point='mazeworld.envs:FiveRoomXLTextMoveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeMoveableStoch-v0',
    entry_point='mazeworld.envs:FiveRoomXLMoveableStoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeMoveableStoch-v1',
    entry_point='mazeworld.envs:FiveRoomXLMoveableStochExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeTextMoveableStoch-v0',
    entry_point='mazeworld.envs:FiveRoomXLTextMoveableStoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeMoveableBrownian-v0',
    entry_point='mazeworld.envs:FiveRoomXLMoveableBrownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeTextMoveableBrownian-v0',
    entry_point='mazeworld.envs:FiveRoomXLTextMoveableBrownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeBrownian-v0',
    entry_point='mazeworld.envs:FiveRoomXLBrownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeTextBrownian-v0',
    entry_point='mazeworld.envs:FiveRoomXLTextBrownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeAll-v0',
    entry_point='mazeworld.envs:FiveRoomXLAllStoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeAllExt-v0',
    entry_point='mazeworld.envs:FiveRoomXLAllStochExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeAll-v1',
    entry_point='mazeworld.envs:FiveRoomXLAll',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeAllExt-v1',
    entry_point='mazeworld.envs:FiveRoomXLAllExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLargeTextAll-v0',
    entry_point='mazeworld.envs:FiveRoomXLTextAll',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLong-v0',
    entry_point='mazeworld.envs:FiveRoomLong',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLong-v1',
    entry_point='mazeworld.envs:FiveRoomLongwide',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLong-v2',
    entry_point='mazeworld.envs:FiveRoomLongunpadded',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomLong-v3',
    entry_point='mazeworld.envs:FiveRoomLongExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomNoObj-v0',
    entry_point='mazeworld.envs:FiveRoomNoobj',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomOneObj-v0',
    entry_point='mazeworld.envs:FiveRoomOneobj',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomOneObj-v1',
    entry_point='mazeworld.envs:FiveRoomOnewhite',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomRandomFixed-v0',
    entry_point='mazeworld.envs:FiveRoomRandomfixed',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomBouncing-v0',
    entry_point='mazeworld.envs:FiveRoomBouncing',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='5RoomBrownian-v0',
    entry_point='mazeworld.envs:FiveRoomBrownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Maze-v0',
    entry_point='mazeworld.envs:Maze',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8Room-v0',
    entry_point='mazeworld.envs:EightRoom',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8Room-v1',
    entry_point='mazeworld.envs:EightRoomRgb',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8Room-v2',
    entry_point='mazeworld.envs:EightRoomDiff',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8Room-v3',
    entry_point='mazeworld.envs:EightRoomExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8Room-v4',
    entry_point='mazeworld.envs:EightRoomDiffRotated',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8Room-v5',
    entry_point='mazeworld.envs:EightRoomDiffWhite',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8RoomWeather-v0',
    entry_point='mazeworld.envs:EightRoomWeather',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8RoomLargeExt-v0',
    entry_point='mazeworld.envs:EightRoomXLExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8RoomLargeWeather-v0',
    entry_point='mazeworld.envs:EightRoomXLWeather',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8RoomHardExt-v0',
    entry_point='mazeworld.envs:EightRoomHardExt',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='8RoomHardWeather-v0',
    entry_point='mazeworld.envs:EightRoomHardWeather',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='PianoLong-v0',
    entry_point='mazeworld.envs:PianoLong',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Aligned-v0',
    entry_point='mazeworld.envs:Aligned',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Misaligned-v0',
    entry_point='mazeworld.envs:Misaligned',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})
