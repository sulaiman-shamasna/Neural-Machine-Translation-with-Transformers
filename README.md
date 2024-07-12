# Neural-Machine-Translation-with-Transformers
---

## Usage

To work with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sulaiman-shamasna/Neural-Machine-Translation-with-Transformers.git
    ```
    
2. **Set up Python environment:**
    - In this project, I used **Python 3.10**.
    - Create and activate a virtual environment:
        ```bash
        python -m venv env
        ```
    - And activate it, 
      - For Windows (using Git Bash):
        ```bash
        source env/Scripts/activate
        ```
      - For Linux and macOS:
        ```bash
        source env/bin/activate
        ```

3. **Install dependencies:**

    To prepare the packages and dependencies, please install the following in order.

    - ```pip install protobuf~=3.20.3```
    - ```pip install tensorflow_datasets```
    - ```pip install tensorflow-text tensorflow```
    - ```pip install numpy==1.26.4```   
    - ```pip install tensorflow-datasets --upgrade```

2. **Modules:**
    - ```prepare_data.py```
    - ```models.py```
    - ```train.py```
    
    To run an experiment, use the command: ```python train.py```, by doing so, the dataset will be downloaded and prepared automatically as implemented in ```prepare_data.py```.

## References
- [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer).