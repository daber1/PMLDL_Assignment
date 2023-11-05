# PMLDL Assignment 1 - Text De-Toxification

<font size="3"> *Student's name: Vladislav Lopatovskii* <br>
    *Email: v.lopatovskii@innopolis.university* <br>
    *Group: BS21-AAI* <br>
    </font>
    

## Getting Started

Follow the steps below to clone the repository and set up the project:

1. Open a terminal or command prompt.

2. Change the current working directory to the location where you want the cloned directory.

3. Run the following command to clone the repository:
```
    git clone https://github.com/daber1/PMLDL_Assignment
```
4. Navigate to the project directory:
```
    cd PMLDL_Assignment
```
5. Install the required Python dependencies:
```
    pip install -r requirements.txt
```
### Prerequisites

- Python 3.8+
- Git
- Cuda supported GPU (optional)



## Usage
For prediction
    ```
    python predict_model.py --encoder --decoder --input_path
    ```
For training
    ```
    python train_model.py --epochs --hidden_size -- batch_size --input_path --output_path
    ```
To make a dataset
    ```
    python make_dataset.py --data_link --output_path
    ```
