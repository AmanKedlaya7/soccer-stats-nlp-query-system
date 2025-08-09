# Soccer Stats NLP Query System

## Overview
This project is a Python-based natural language query system for soccer player statistics. Using SentenceTransformers, it matches user questions to a set of predefined queries based on semantic similarity. It then retrieves and displays relevant stats such as goals, assists, and player positions from a sample dataset. The tool is useful for fans, analysts, or coaches interested in quick soccer stats insights.

## Installation
To run this project, you need Python installed on your machine.  
Install the required Python packages using pip:
- `sentence-transformers` for embedding and semantic similarity  
- `pandas` for data manipulation  

You can install the dependencies with:  
`pip install sentence-transformers pandas`  

## Usage
Run the script `soccer_stats_qa.py` in your terminal or command prompt.  
Type your soccer-related questions, such as “Who has the most goals?” or “How many assists does Mo Salah have?”  
The program will return the best matching answer based on the dataset.  
Type `exit` to close the program.

## Example Questions
- Who has the most goals?  
- Which midfielder has the most assists?  
- How many goals does Mo Salah have?  
- Top 5 assist makers  
- Which defender has the least goals?  

## Dataset
The project uses a sample dataset containing stats for 10 well-known soccer players. The data includes player names, positions, goals scored, and assists.
