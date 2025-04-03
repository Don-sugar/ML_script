## Naming Convention for Systems

The naming convention for systems is structured as follows:  
**Non-equivalent site (NbI or NbII or SiI or SiII)_Central substitution element - Environment substitution site (NbI, NbII, SiI, SiII denote globally non-equivalent sites)_Environmental substitution element (Numbers indicate locally distinct nearest neighbors)**.  

For example, **NbI_AlI-SiI_Co1** indicates that the NbI site serves as the central atom and is substituted by an Al atom. The hyphen “-” separates the description of the environment, where the SiI non-equivalent site in the surrounding shell is substituted by a Co atom (the number 1 signifies that the Co atom is the first nearest neighbor to the NbI central atom).

---

## Machine Learning Script for High-Temperature Alloy Materials

This script incorporates a variety of machine learning algorithms specifically designed for data mining in high-temperature alloy materials. To utilize this script effectively, it is essential to ensure that the pre-prepared data and the Python script are located within the same directory. This arrangement facilitates seamless access and processing of the data by the script.

### Prerequisites

- Python 3.x
- Required libraries (e.g., scikit-learn, pandas, numpy)
- Pre-prepared data files in the same directory as the script

### Instructions

1. **Grant Execution Permissions**  
   Before running the script, you need to grant it execution permissions. This can be done using the following command in the terminal:  
   ```bash
   chmod +x ML_script.py
   ```  
   This command modifies the file permissions to allow the script to be executed as a program.

2. **Run the Script**  
   Once the permissions are set, you can execute the script using Python. The command to run the script is:  
   ```bash
   python ML_script.py
   ```  
   This command initiates the script, which will then perform the specified machine learning tasks on the data located in the same directory. By following these steps, the script will process the data using the embedded machine learning algorithms, providing valuable insights and analysis for materials.

---
## Example Usage

1. Place your data files and `ML_script.py` in the same directory.
2. Open a terminal and navigate to the directory containing the script.
3. Run the following commands:
   ```bash
   chmod +x ML_script.py
   python ML_script.py
   ```
4. The script will process the data and output the results.
---
## Contact
For any questions or issues regarding the script, please contact yctang@shu.edu.cn.
