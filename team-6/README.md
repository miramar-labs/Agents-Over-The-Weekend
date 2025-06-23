# Team 6 AI Agent Project

## Description

## Dev Notes

### Linux/WSL:

- [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
  - get API key `GOOGLE_API_KEY`

- [LangChain docs](https://python.langchain.com/docs/introduction/)

- [LangSmith](https://smith.langchain.com/)
  - get API key `LANGCHAIN_API_KEY`

- First set up a python venv and work in that:

        # (*if* we need sqlite3) build a version of python with sqlite3 built-in 
        sudo apt-get update
        sudo apt-get install libsqlite3-dev

        # build some version of python
        pyenv install 3.13.0

        # now create a virtual env with this sql-enabled python
        pyenv virtualenv 3.13.0 aotw

        # set it as the venv for team-6 folder
        cd Agents-Over-The-Weekend/team-6

        pyenv local aotw
        
        # ensure sqlite3 not listed in requirements.txt
        
        # install any packages we need
        pip install -r requirements.txt

- Create a `.env` file in the team-6 folder to store your ENV vars/API keys:

        LANGCHAIN_API_KEY=
        LANGCHAIN_TRACING_V2=true
        GOOGLE_API_KEY=
