# Introduction
The **Intelligent Document Processing System** is an application designed and developed by a group of 3 - Sujatha, Jun Ming and Gwen - during our Master of Technology in Artificial Intelligence Systems Programme in NUS-ISS. We worked closely with our industry partner which provides the domain expertise to ensure the application would improve engineering manuals understanding for the engineering sector.

| <img src="image/Overview.png" alt="Overall System Process Flow" style="width: 50%;">|
|:-----------------------------------:|
| Fig. Architecture Overview |

The overall architecture for the application is as depicted in the figure above. The key components and techniques used in the developments include:
- **Image Processing:** Object Detection
- **Text Processing:** Named-Entity Recognition (NER) & Vector Embeddings
- **Retrieval Augmented Generation (RAG)**

Head over to the following link to view a [snapshot of the application](https://sujatha-sureshkmr.github.io/tech-analytics/nus-capstone/mechchatbotmanual.html).

# Setting up of the application


## 1. Setting up virtual environment

- Set up a new virtual environment with the python version `python = 3.11.9`
- Install the required libraries from requirements.txt
    
    `pip install -r requirements.txt`
    

## 2. Download of source code from github

- Download the code from github, following the following instruction.
    1. Ensure Git Large File Storage (LFS) has been installed. Follow the instructions in the link.
        - For *Windows* - [https://git-lfs.com/](https://git-lfs.com/)
        - For *Linux* - [https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)
    2. Clone the repo
        
        `git lfs clone **repo address**`
        

## 3. Start the application

- Navigate to the tachatbot folder in which the [app.py](http://app.py) is located. Run the following commands.
    - For *Windows*
        
        ```bash
        set FLASK_APP=app
        set FLASK_DEBUG=1
        flask run
        ```
        
    - For *Linux*
        
        ```bash
        export FLASK_APP=app
        export FLASK_DEBUG=1
        flask run
        ```
        
- The application will be running on localhost `http://127.0.0.1:5000/`

## 4. Testing of chatbot

- The vector database for the EV nissan leaf manual was pre-loaded.
- Navigate to the `Chatbot-LLM` page, choose the product `EVB_Nissan_2013` and load the LLM.

|<img src="image/UI.png" alt="Overall System Process Flow" style="width: 50%;">|
|:-----------------------------------:|
| Fig. Chatbot Snapshot |
