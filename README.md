# Chat with your Homework:

## Learn more from Your Data (PDF + word) with Langchain and Streamlit

### Hardware requiments
- **Minimum RAM**: 8 GB or higher
- **Decent CPU**: 10-12 core

### Software Requirements
- **Anaconda**: Recommended for managing Python environments

## Steps
### 1. Install Anaconda
If you haven't already, download and install Anaconda from [Anaconda's website](https://www.anaconda.com/products/distribution).


### 2. Create a New Anaconda 
Open a terminal or Anaconda Prompt and create a new environment named `myenv` with Python 3.8:
```bash
conda create --name myenv python=3.8
```

### 3. Activate the Anaconda Environment
Activate the newly created environment `myenv`:
```bash
conda activate myenv
```

### 4. Navigate to your cloned directory
Navigate to the directory where you have cloned your project:
```bash
cd `C://{your cloned directory}
```

### 5. Install Packages
Install the necessary Python packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```

###  5. Run app
Run the Streamlit application using the following command:
```bash
streamlit run interface.py
```
This opens app [http://localhost:8501/](http://localhost:8501/)

Read more on Streamlit application [Streamlit documentation](https://docs.streamlit.io/)