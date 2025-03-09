# Python Project with UV and Git

Open a Git Bash terminal.

## Initialise the project
A folder with a pyproject.toml is recognised as a project
```bash
uv init my_project --python 3.10
cd my_project
```

Take a look at the project
```bash
ls -a
cat pyproject.toml
```

## Create a virtual environment
Install the Python version in .python-version
```bash
uv venv
```

If a different Python version is required
```bash
uv venv --python 3.13
```

Activate the virtual environment
```bash
source .venv/scripts/activate
```

Check that main.py can be executed
```bash
uv run main.py
```

Make source code directory and move main.py
```bash
mkdir -p src/my_project
mv main.py src/my_project/main.py
```

Add the build system to pyproject.toml
```bash
echo '' >> pyproject.toml
echo '[build-system]' >> pyproject.toml
echo 'requires = ["hatchling"]' >> pyproject.toml
echo 'build-backend = "hatchling.build"' >> pyproject.toml
echo '' >> pyproject.toml
echo '[tool.hatch.build.targets.wheel]' >> pyproject.toml
echo 'packages = ["src/my_project"]' >> pyproject.toml
```

## Add project dependencies 

UV will automatically install dependencies and add them to pyproject.toml
```bash
uv add plotly scikit-learn "pandas[computation,excel,output-formatting,performance,plot]"
```

Remove dependencies like so
```bash
uv remove matplotlib
```

Add optional developer dependencies 
```bash
uv add --dev pytest
```bash

Check that pyproject.toml has been updated
```bash
cat pyproject.toml
```

Check if the dependencies were installed in venv
```bash
uv pip list
```

## Create a Git and GitHub repository

UV already created a .git directory; to re-initialise
```bash
git init
```

Check that you're on the master branch with no commits yet and all files untracked
```bash
git status
```

Stage all files
```bash
git add .
```

Make initial commit
```bash
uv sync
git commit -m "initial commit"
```

Go to https://github.com/stabilefrisur?tab=repositories to add a new GitHub repository my_project

Establish the remote connection
```bash
git remote add origin https://github.com/stabilefrisur/my_project.git
```

Push to GitHub
```bash
git branch -M master
git push -u origin master
```

## Create a Git branch and merge into master

Create a new branch
```bash
git checkout -b new_feature
```

Stage all code changes
```bash
git add .
```

Commit code changes 
```bash
git commit -m "Add new feature"
```

Push the new branch to GitHub
```bash
git push -u origin new_feature
```

Start a pull request:
- Go to your repository on GitHub.
- You should see a prompt to create a pull request for the new_feature branch.
- Follow the instructions to create the pull request.

Merge the pull request into master:
- Once the pull request is reviewed and approved, merge it into the master branch using the GitHub interface.

Delete the new_feature branch locally and remotely:
```bash
git checkout master
git pull origin master
git branch -d new_feature
git push origin --delete new_feature
```

## Register and run main.py 

Register an entry point to the package
```bash
echo '[project.scripts]' >> pyproject.toml
echo 'my_project = "my_project.main:main"' >> pyproject.toml
```

Install the package in editable mode
```bash
uv pip install -e .
```

The local package will now be listed in pip
```bash
uv pip list
```

Run the main program which is now registered under my_project
```bash
uv run my_project
```

## Configure and run pytest

Make tests directory for pytest
```bash
mkdir tests
```

Add the tests path to pyproject.toml
```bash
echo '' >> pyproject.toml
echo '[tool.pytest.ini_options]' >> pyproject.toml
echo 'pythonpath = "src"' >> pyproject.toml
echo 'testpaths = "tests"' >> pyproject.toml
```

Run pytest
```bash
pytest
```

## Build and publish the package

Update the project environment
```bash
uv sync
```

Build Python packages into source distributions and wheels
```bash
uv build
```

Commit changes and push to GitHub before publishing
```bash
git add .
git commit -m "MESSAGE"
git push -u origin master
```

Upload distributions to an index; input PyPi credentials
```bash
uv publish
```

Verify that the package is published on PyPi: 
https://pypi.org/project/my_project/

## Test the package installation

Create a test environment 
```bash
python -m venv test_env
source test_env/scripts/activate
```

Install the package and check that it's listed
```bash
pip install my_project
pip list
pip show my_project
```

Import the package and run its entry point in a Python shell
```bash
python
>>> import my_project
>>> my_project.main()
```

Alternatively, run the main program as a module
```bash
python -m my_project.main
```

If the main module is registered in pyproject.toml, just run
```bash
my_project
```