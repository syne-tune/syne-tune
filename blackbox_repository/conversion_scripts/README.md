## Conversion scripts

To add a new dataset of tabular evaluation, you need to 
1) write a function able to regenerate it (see fcnet_import.py for an example)
2) add your generate in `recipes.py` in `generate_blackbox_recipe`.