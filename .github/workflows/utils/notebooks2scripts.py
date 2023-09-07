import os
import json

from traitlets.config import Config
from nbconvert.preprocessors import TagRemovePreprocessor
from nbconvert.exporters import PythonExporter

c = Config()
c.TagRemovePreprocessor.remove_cell_tags = ("install-dependencies")
python_exporter = PythonExporter(config=c)


notebook_directory = 'examples/notebooks'
files = [f for f in os.listdir(notebook_directory) if os.path.isfile(os.path.join(notebook_directory, f))]

converted_files = []

for file_name in files:

    # convert to python script and write to disk
    python_code, _ = python_exporter.from_filename(f"{notebook_directory}/{file_name}")
    converted_file_path = f"{notebook_directory}/{file_name.split('.')[0]}.py"
    with open(converted_file_path, 'w') as f:
        f.write(python_code)

    converted_files.append(converted_file_path.split('/')[-1])

matrix_json = json.dumps(converted_files)
print(matrix_json)
with open('matrix.json', 'w') as f:
    f.write(matrix_json)

name = "matrix"
value = matrix_json
with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
    print(f'{name}={value}', file=fh)
