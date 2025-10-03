# Github Repository for the Project in course Data Mining (fall 2025)

## Preprocessing Steps

### 1. Attribute Selection
We started with the full set of attributes available in the arXiv metadata.  
From these, we kept only a minimal subset relevant to our task.

**✅ Kept**
- **id** – unique identifier for each arXiv submission  
- **title** – paper title  
- **journal-ref** – reference to published journal (if available)  
- **categories** – main category of the paper (preprocessed to only the primary main category)  
- **abstract** – paper abstract text  

**❌ Removed**
- **authors** – list of author names  
- **comments** – additional metadata such as page count or figures  
- **doi** – digital object identifier (if available)  
- **license** – licensing information  
- **report-no** – report number from institutions  
- **versions** – submission version history  
- **update_date** – date of last update on arXiv  
- **submitter** – name/email of the submitting author  
- **other auxiliary metadata fields** not required for text classification  

### 2. Format Conversion
- Converted the dataset from **JSON** to **CSV** for easier processing with pandas.  

### 3. Grouping of categories
Sub-categories are replaced with their parent category to reduce the number of categories,
resulting in the following categories:
- **astro-ph**
- **cond-mat**
- **cs**
- **econ**
- **eess**
- **gr-qc**
- **hep-ex**
- **hep-lat**
- **hep-ph**
- **hep-th**
- **math**
- **math-ph**
- **nlin**
- **nucl-ex**
- **nucl-th**
- **physics**
- **q-bio**
- **q-fin**
- **quant-ph**
- **stat**

### 4. Category Normalization
- From the `categories` field, only the **first listed category** was retained (the primary category in arXiv).  
- Extracted the **main category prefix** (the part before the dot).  

**Example:**  
- Input: `"math.bayes stat.analytic"`  
- Output: `"math"`
