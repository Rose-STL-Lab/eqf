import sys
script_name = sys.argv[0]
if script_name == "-m" and len(sys.argv) > 1:
    script_name = sys.argv[1]

if not script_name.endswith('droid.py'):
    from .droid import DroidVideoDataset

if not script_name.endswith('minecraft.py'):
    from .minecraft import MinecraftVideoDataset

if not script_name.endswith('re10k.py'):
    from .re10k import Re10KVideoDataset
