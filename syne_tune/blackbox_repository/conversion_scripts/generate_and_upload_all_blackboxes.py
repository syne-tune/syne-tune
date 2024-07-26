"""
Generates all blackbox recipes locally and upload them to HuggingFace Hub.
Intended to use for internal developers, you should set HF_TOKEN environment variable to get write access to the hub.
"""
from syne_tune.blackbox_repository.conversion_scripts.recipes import (
    generate_blackbox_recipes,
)

if __name__ == "__main__":
    for blackbox, recipe in generate_blackbox_recipes.items():
        try:
            recipe.generate(upload_on_hub=True)
        except Exception as e:
            print(f"Failed generating and uploading {blackbox}")
            print(e)
