# Neural-Machine-Translation-with-Transformers
---
## Project Structure
```
project/
├── config.py
├── data/
│   ├── __init__.py
│   ├── prepare_data.py
├── models/
│   ├── __init__.py
│   ├── transformer.py
├── train.py
├── callbacks.py
├── run.py
├── utils/
│   ├── __init__.py
│   ├── learning_rate.py
└── requirements.txt

```
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

2. **Core modules:**
    - ```data/prepare_data.py```
    - ```models/transformer.py```
    - ```utils/learning_rate.py```
    - ```utils/metrics.py```
    - ```train.py```
    - ```callbacks.py```
    - ```run.py```
    
    To run an experiment, use the command: ```python run.py```, by doing so, the dataset will be downloaded and prepared automatically as implemented in ```data/prepare_data.py```.

3. **Training monitoring**

    - To be able to monitor the evaluation metrics of your experiement, ```tensorboard``` is used. It is integrated with the pipeline as a callback, you can run it by:

        - Opening a new *shell* and navigate to the project's directory.
        - Activate the virtual environment, and run the command:
            - ```tensorboard --logdir=logs/```
    - Having done this, a ```port:IP``` will appear in the shell, either press on it + ctrl or open a browser and paste it, it should look like this:
        - ``` http://localhost:6006/```


## References
- [Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer).