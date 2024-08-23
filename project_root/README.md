# AI Pipeline for Image Segmentation and Object Analysis

## Overview
This project is designed to create an AI pipeline for image segmentation and object analysis. The pipeline includes the following stages:

1. Image Segmentation: Extracting objects from images.
2. Object Identification: Identifying and describing the segmented objects.
3. Text Extraction: Extracting text data from images.
4. Summarization: Generating summaries based on the extracted data.
5. The application is built using Python and Streamlit for the UI, with various models and utilities handling the core functionality.
The application is built using Python and Streamlit for the UI, with various models and utilities handling the core functionality.

## Features
- Segmentation: Automatically segments objects in uploaded images.
- Identification: Identifies and describes segmented objects using pre-trained models.
- Text Extraction: Extracts text from images, useful for OCR and document analysis.
- Summarization: Summarizes the identified objects and extracted text for easier interpretation.

## Setup Instructions
### Prerequisites
Ensure you have Python 3.8 or higher installed on your system.
### Installation
1. Clone the Repository: git clone https://github.com/rahuljoshi1814/rahul-joshi1814-wasserstoff-AiInternTask.git
   cd yourrepository
2. Create a Virtual Environment: python -m venv venv
   To activate virtual Environment: venv\Scripts\activate
3. Install Required Packages: pip install -r requirements.txt

## Running the Application
1. Start the Streamlit App: streamlit run app.py
2. Upload an Image: Use the sidebar to upload an image.
3. Run the Pipeline: Click on the buttons in the sidebar to run segmentation, identification, text extraction, and summarization sequentially

## Usage Guidelines
- Segmentation: The app will segment objects from the uploaded image and save metadata.
- Identification: The app will identify each segmented object and save the descriptions.
- Text Extraction: Extract text from the uploaded image.
- Summarization: Generate summaries based on the identification and text extraction results.
- Data Mapping: Merges and maps the data for visualization.
- Visualization: Annotates the original image and displays summary tables.
- 
## Troubleshooting
If you encounter issues during any stage of the pipeline, ensure:

The directories specified in app.py exist and are correctly named.
All necessary files (e.g., metadata.csv) are generated during the segmentation step.
For further assistance, refer to the error messages displayed in the Streamlit app.
