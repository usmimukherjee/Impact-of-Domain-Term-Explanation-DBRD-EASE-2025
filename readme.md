# Replication Package: Understanding the Impact of Domain Term Explanation on Duplicate Bug Report Detection

## Abstract
Duplicate bug reports make up 42\% of all reports in bug tracking systems (e.g., Bugzilla), causing significant maintenance overhead. Hence, detecting and resolving duplicate bug reports is essential for effective issue management. Traditional techniques often focus on detecting textually similar duplicates. However, existing literature has shown that up to 23\% of the duplicate bug reports are textually dissimilar. Moreover, about  78\% of bug reports in open-source projects are very short (e.g., less than 100 words) often containing domain-specific terms or jargon, making the detection of their duplicate bug reports difficult. In this paper, we conduct a large-scale empirical study to investigate whether and how enrichment of bug reports with the explanations of their domain terms or jargon can help improve the detection of duplicate bug reports. We use  92,854 bug reports from three open-source systems, replicate seven existing baseline techniques for duplicate bug report detection, and answer two research questions in this work. We found significant performance gains in the existing techniques when explanations of domain-specific terms or jargon were leveraged to enrich the bug reports. Our findings also suggest that enriching bug reports with such explanations can significantly improve the detection of duplicate bug reports that are textually dissimilar.

## Folder Structure

1. **data**: Contains the following files:
   - `baseline_dataset`: Original bug reports.
   - `domain_terms`: A collection of domain-specific terms or jargon.
   - `enriched`: Bug reports enriched with domain-specific term explanations.

2. **data_pairs**: Contains pairs of bug reports for classification techniques.

3. **src**: Contains the source code for various components:
   - **baselines**: Replications of baseline techniques for duplicate bug report detection (DBRD).
     -  `bm25`: BM25-based replication.
     -  `ctedb`: CTEDB replication.
     -  `cupid`: CUPID replication.
     -  `DC-CNN`: DC-CNN-based replication.
     -  `LDA`: LDA-based replication.
     -  `SBERT`: SBERT replication
     -  `siameseCNN`: Siamese CNN-based replication.
   - **domain_term_extraction**: Code for extracting domain terms from bug reports for the three subject systems.
   - **enrich**: Code for enriching bug reports with domain-specific term explanations.
   - **statistical_test**: Code for performing statistical tests to evaluate the performance of all techniques.
   - **T5Train**: Training the T5 model


## System Requirements
- Operating System: Windows 10 or Mac OS 15.3.2 or higher
- Python Version: 3.11
- Development Environment:  Visual Studio Code (VSCode)
- RAM: 16GB
- GPU: 8GB

## Installation Version
To replicate the work, follow these steps:

### Step 1: Setting Up the Virtual Environment
1. Create a virual environment using the following command:
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
### Step 2: Installing Dependencies

1. After creating the virtual environment, activate it by following the instructions -  [here](https://docs.python.org/3/library/venv.html)

2. Install the necessary dependencies
```sh
pip install -r requirements.txt
```

### Step 3: Running the Python Scripts
After setting up the environment and installing dependencies, you can run the Python scripts as follows:

1. **Extract domain terms**:
   ```sh
   python src/domain_term_extraction/extract_terms.py
   ```

2. **Enrich bug reports with domain term explanations**:
   Open the desired `.ipynb` file and run the cells as needed.

3. **Perform statistical tests**:
   Open the desired `.ipynb` file and run the cells as needed.

5. **Train the T5 model**:
   ```sh
   python src/T5Train/java_model.py

## Preprint
The preprint of this work is available at:
[http://arxiv.org/abs/2503.18832](http://arxiv.org/abs/2503.18832)

For any further questions or issues, please feel free to open an issue on the GitHub page.

## Licensing Information
This project is licensed under the MIT License, a permissive open-source license that allows others to use, modify, and distribute the project's code with very few restrictions. This license can benefit research by promoting collaboration and encouraging the sharing of ideas and knowledge. With this license, researchers can build on existing code to create new tools, experiments, or projects, and easily adapt and customize the code to suit their specific research needs without worrying about legal implications. The open-source nature of the MIT License can help foster a collaborative research community, leading to faster innovation and progress in their respective fields. Additionally, the license can help increase the visibility and adoption of the project, attracting more researchers to use and contribute to it.