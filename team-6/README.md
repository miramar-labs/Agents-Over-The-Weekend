# Team 6 Project

## Description

## Dev Notes

### Linux/WSL:

- First set up a python venv and work in that:

    # build a version of python with sqlite3 built-in (in case we use tit as our DB)
    sudo apt-get update
    sudo apt-get install libsqlite3-dev

    pyenv install 3.13.0

    # now vreate a virtual env with this sql-enabled python
    pyenv virtualenv 3.13.0 aotw

    # set it as the venv for team-6 folder
    cd Agents-Over-The-Weekend/team-6

    pyenv local aotw
    
    # ensure sqlite3 not listed in requirements.txt
    
    # install any packages we need
    pip install -r requirements.txt

