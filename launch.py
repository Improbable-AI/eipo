from rlpyt.utils.launching.arguments import get_args
from rlpyt.utils.launching.launcher import start_experiment, launch_tmux

if __name__ == "__main__":

    args = get_args()
    if args.launch_tmux == 'yes':
        launch_tmux(args) # launches tmux
    else:
        start_experiment(args) # launches the actual experiment inside of a tmux session









