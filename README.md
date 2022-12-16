# The Welfare Effects of Debt: Crowding Out and Risk Shifting
The code and data in this repository allow users to replicate analysis results and create the figures found in CBO's working paper
 [*The Welfare Effects of Debt: Crowding Out and Risk Shifting*](https://www.cbo.gov/publication/58849) by Michael Falkenheim, Senior Adviser in CBO's Financial Analysis Division.

## How to Install the Welfare Effects of Debt Analysis
Follow these three steps to install the code and data associated with the **Welfare Effects of Debt** analysis on your computer:

1) **Install the Anaconda distribution of Python**  
Download and install the Anaconda distribution of Python from Anaconda's [Installation page](https://docs.anaconda.com/anaconda/install/index.html).
</br></br>The **Welfare Effects of Debt** analysis was conducted using Python 3.8 on computers running Windows 10, although the analysis should run on other operating systems as well.
</br></br>The external packages used in the analysis were managed using Anaconda's built-in package manager, `conda`. To replicate the results in this repository, you will need to use `conda` to create a virtual environment that loads the same versions of Python and external packages used when the analysis was conducted. All the external packages (and their versions) are documented in the `environment.yml` file in the project’s root directory. That file is used to create a virtual environment that matches the one used when the analysis was conducted. This is done in step 3, below.

2) **Download the repository ("repo") from GitHub**  
There are several options for how to get the code and data from GitHub to your computer:  

    * If you have `git` installed on your computer, you can clone a copy of the repo to your computer. This is done through git with the command:  
    `git clone https://github.com/us-cbo/debtwelfare.git`

    * If you also have a GitHub account, you should first `fork` a copy of the repo to your own GitHub account and then clone it to your computer with the command:  
    `git clone https://github.com/<your-GitHub-account-name>/debtwelfare.git`.

    * If you don’t have git installed on your computer, you can [download a zip file](https://github.com/us-cbo/debtwelfare/archive/refs/heads/main.zip) containing the entire repo and then unzip that file in a directory on your computer.

3) **Create the virtual environment**  
Once you have installed the Anaconda distribution of Python and you have downloaded a copy of the repo to your computer, follow these steps to create a virtual environment that will make sure you have all the appropriate dependencies to run the analysis:

    * Open the `Anaconda Prompt` application, which comes as part of the Anaconda installation  

    * Navigate to the root directory where you cloned or downloaded the repository on your computer using the change directory (`cd`) command:  
    `cd path/to/your/copy/of/CBO-debtwelfare`  
    (The last subdirectory name, `CBO-debtwelfare`, is just a suggested name; you may name the subdirectory anything you wish.)

    * Create a virtual environment that matches the one used to conduct the **Welfare Effects of Debt** analysis with the command:  
    `conda env create -f environment.yml`  
    (That command will create a virtual environment on your computer named `CBO-debtwelfare` and may take several minutes to complete.)

    * Activate the newly created virtual environment with the command:  
    `conda activate CBO-debtwelfare`  
    (To replicate the results in the `/data/outputs/` directory, the code needs to be run from within that virtual environment.)

    * When finished working with the **Welfare Effects of Debt** repository, deactivate the virtual environment from the Anaconda Prompts by typing:  
    `conda deactivate`

## How to Run the Welfare Effects of Debt Analysis
Once the above steps have been followed, and with the `CBO-welfare-effects-debt` virtual environment activated you can run the model with the following commands typed into the `Anaconda Prompt` from the root of the project directory:  
`python src/main.py`

That command will run the analysis using a default **Cobb-Douglas** production function specification.

To run the analysis using a **linear** production function specification, use the command:  
`python src/main_linear.py`

For each of those runs, a single output file will be written to the `/data/output/` directory under the project (`welfare_effect_summary.csv` and `welfare_effect_summary_linear.csv`, respectively).

Finally, to create the figures in the working paper, use the command:  
`python src/plot_figures.py`

The figures are written out to the `/figures/` directory under the root directory for the project.

> When you are finished working with the repository, deactivate the virtual environment by typing: `conda deactivate` at the Anaconda Prompt.

## Contact
Questions about the code and data in this repository may be directed to CBO's Office of Communications at communications@cbo.gov.

CBO will respond to such requests as its workload permits.
