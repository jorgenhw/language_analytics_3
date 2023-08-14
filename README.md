<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Cultural Datascience 2023</h1> 
  <h2 align="center">Assignment 3</h2> 
  <h3 align="center">Language Analytics</h3> 
  <p align="center">
    Jørgen Højlund Wibe<br>
    Student number: 201807750
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project
This repository contains the steps required to generate text based on a pre-trained RNN model using a dataset of comments. This involves preprocessing the data, tokenizing the comments, create and train a model, and generate text based on the trained model.

Please refer to the ```README.md``` file for detailed instructions on how to use the script and any necessary prerequisites.

<!-- USAGE -->
## Usage

To use or reproduce the results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.77.3 (Universal) on Mac OS version 13.3.1 (a). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment, install libraries and run the project. BEWARE that the ```requirements.txt``` installs ```tensorflow-macos==2.12.0``` which will run on a Mac. If you are on a windows or linux machine consider just installing ```tensorflow==2.12.0```.

1. Get data
2. Clone repository
3. Run ```setup.sh```
4. [Optional] Changing arguments
5. [Optional] Generate text from a user-suggested prompt

### Get data
Download the [New York Times Comments](https://www.kaggle.com/datasets/aashita/nyt-comments) dataset and place all ```comments``` files in the ```data``` folder.

### Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/jorgenhw/language_analytics_assignment_3.git
cd language_analytics_assignment_3
```

### Run ```setup.sh```

To replicate the results, I have included a bash script that automatically 

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

Run the code below in your bash terminal:

```bash
bash setup.sh
```

### [Optional] Changing arguments via ```argparse```
To provide more flexibility and enable the user to change the parameters of the script from the command line, we have implemented argparse in our script. This means that by running the script with specific command line arguments, you can modify parameters such as the batch size and the number of epochs to train the model.

To see all the available arguments and their descriptions, simply run the command:

```bash
python main.py --help
```

### Changing the prompt
You can change the prompt that the should continue by using the argument ```seed_text```. In the bash terminal, type

```bash
python3 main.py --seed_text "I love pizza"
```

This would make the model continue on the sentence "I love pizza".

## Inspecting results
The trained models will be located in the ```model``` folder along with a plot showing the loss during training.

## Example of text generation
Below are two examples of how well this model generates text. 

NB. Just a reminder that this model was trained on only a 1000 of the comments.

When tasked with producing a 5-word continuation of the sentence "I like the smell of fries because the", the model outputs:
```
I like the smell of fries because the smell is the smell is
```
This is somewhat meaningful, although one could suspect that the model would keep on repeating itself. 

With a 10-word continuation, the model output was
```
I like the smell of fries because the the the the the the the the the the the
```
This is pure nonsense and probably has to do with the limited training.

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:
```
│   main.py
│   README.md
│   requirements.txt
│   setup.sh
│   task_description.md
│
├───src
│       preprocessing_data.py
│       creating_model.py
│       generate_text.py
│
├───misc
│       pre_trained_model_197.h5
│       training_plot.png
│
│
└──data
        empty folder <--- Put your data here
```

<!-- DATA -->
## Data
The repository utilizes the [New York Times Comments dataset](https://www.kaggle.com/datasets/aashita/nyt-comments) available on [Kaggle](https://www.kaggle.com/). The dataset contains just above 2 million comments from New York Times articles in the time period Jan-May 2017.

Due to computational limitations, the model was trained using only 1000 comments. Importantly, it should be noted that the code is designed to be able to run on the entire dataset.

<!-- RESULTS -->
## Remarks on findings
From looking at the output of the model along with the loss curve located in the ```misc``` folder, we can tell that the model training was not very successful. 

We might be able to improve performance by including the entirety of the dataset, increasing model complexity by e.g. adding more hidden layers or increase the size of the embedding layer, as well as increase the number of epochs (again, due to computational limitations, this was set to only 10). 