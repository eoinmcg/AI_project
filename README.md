---
title: AI Project
emoji: 👾
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# TowardsAI Course final project

By Eoin McGrath eoin.mcg@gmail.com

## Title
Game dev tutor focusing on LittleJS framework

## Overview
Data gathered from offical docs and github repo source.
Processed and generated embeddings stored in Chroma DB.
Evaluation scripts and data provided.
Reranker used to improve generated answers.

## Optional Extras
1.  Implement streaming responses.
2.  The app is designed for a specific goal/domain that is not a tutor about AI: designed for a specific javascript based game engine
3.  You have shown evidence of collecting at least two data sources beyond those provided in our course: 
  fetch_docs.py = output stored in ./data/littlejs_docs.csv
  fetch_repo.py = output stored in ./data/littlejs_repo.csv
4  There’s code for RAG evaluation in the folder, and the README contains the evaluation results:
  eval_generate_dataset.py = generates synthetic question context  pairs 
  eval_process_dataset.py  = evaluation results based on above questions
  eval_results.txt         = sample saved results
5. Use a reranker in your RAG pipeline. It can be a fine-tuned version (your choice): uses LLMRerank postprocessor
