# Independent Research Project - hs1623

## Overview
This project aims to develop a comprehensive system to predict pollution dispersion and wind velocity field using advanced machine learning techniques, including Convolutional VAE, Computational Fluid Dynamic, and Data Assimilation (DA). The system leverages historical data and observational imagery to enhance prediction accuracy and provide actionable insights.

## Data Description
- `Urban_street.npy`: Data of detailed layout to generate structured mesh in Du Cane Road
- `INHALE_1280.npy`: Data of detailed layout to generate structured mesh in South Kensington
- `final_combined_data.npy`: Processed velocity data needed for CVAE and further simulations
- `final_combined_data_p.npy`: Processed pollution data needed for CVAE and further simulations

## Getting Started

To run the code in this repository, please follow the steps below. 

1. Clone this repository: 

   ```bash
   git clone https://github.com/ese-msc-2023/irp-hs1623.git
   cd the-directory-name-where-the-repo-is
   ```

2. Unzip the necessary files: 
   - `Data Collection`: `Urban_street.npy.zip`, `INHALE_1280.npy.zip`
   - `Data Preprocessing`: `final_combined_data.npy.zip`, `final_combined_data_p.npy.zip`
  
3. Extract the downloaded files to directory

   ```bash
   ./Data
   ```

4. Create and Activate a Virtual Environment

   ```bash
   conda create --name irp python=3.10
   conda activate irp
   ```

5. Install Your Project

   With your virtual environment activated, install your project and its dependencies using the `setup.py` file: 

   ```bash
   pip install .
   ```

## Repository Structure

- **Data Collection**
  - `AI4Urban.py`: Contains computational fluid dynamic model
  - `Raw dataset`: INHALE_1280.npy and Urban_street.npy

- **Data Preprocessing**
  - `Du Cane Road`: Contains processed dataset and trained Convolutional VAE models
  - `South Kensington`: Contains processed dataset and trained Convolutional VAE models
   
- **Models** 
  - `CFD.py`: Contains the CFD models used for simulations
  - `CVAE.py`: Contains the convolutional VAE structure
  - `DA.py`: Contains data assimilation process
  - `utils.py`: Contains helper functions needed by other files
  - `DA_CFD_CVAE.py`: Contains a comprehensive system that include DA, CFD, and CVAE models
  
- **Notebooks** 
  - `IRP.ipynb`: Jupyter notebook for training and testing CVAE models.
  - `notebook.utils.py`: Contains helper functions needed by other files
  
- **Output**
  - `velocity`:  Directory containing updated CVAE models and data for future iterations
  - `pollution`: Directory containing updated CVAE models and data for future iterations

- **Packages**
  - `setup.py`: Setup script for installing the project as a package.
  - `requirements.txt`: List all dependencies that need to be installed using pip to ensure the project runs correctly.

- **License**
  - `LICENSE`: The license file that specifies the permissions, limitations, and conditions under which the software can be used.
  



This is your IRP repository. You must **submit your deliverables** (project plan, final report, and presentation slides) to this repository before the deadlines specified in the IRP Guidelines. In addition, you are expected to **submit your code** here. If, for instance, you are working on an external project and your code is confidential, or you were asked by your supervisor(s) to work in another repository, your code does not have to be submitted here. In that case, you must discuss and agree with both supervisors as soon as possible on how they can access your code. Finally, please keep the `title.toml` file up to date - whenever the title of your project changes, please ensure that this is reflected in `title.toml`.

- For instructions on submitting your **deliverables**, check [`deliverables/README.md`](deliverables/README.md).
- To learn how to update the `title.toml` file, refer to [`title/README.md`](title/README.md).

We will collect your deliverables and the project title from this repository automatically. Therefore, you must follow the instructions in the README files. To ensure your submissions of deliverables and the modifications to `title.toml` are correct, we defined several workflows in GitHub Actions. You can see their status in the Actions tab of this repository or on the badges in README files. It usually takes some time for the badge status to update, even after refreshing the page. Therefore, to see the real-time status of the workflow, check the `Actions` tab.

After you familiarise yourself with this repository, you can delete the content of this `README.md` and populate it with your project-specific information. :)
