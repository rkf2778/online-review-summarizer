import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from scipy.special import softmax
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from rouge_score import rouge_scorer
from rouge import Rouge
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



import time
import io
import os
import pprint
from IPython.display import HTML
import traceback
import logging
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# st.set_page_config(page_title="Review Summary App", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
# # st.set_page_config(layout="wide")
# st.title("Review Summarizer App")
# st.write("This app summarises all the reviews of a product")

FILE_PATH = 'amazon_reviews_us_Mobile_Electronics_v1_00.csv' 

rouge = Rouge()

@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name) if(model_name == "google/pegasus-large") else AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def dataset_load():

    # ===================================================================================================================
    # ================================================= UTILITY FUNCTIONS ===============================================
    # ===================================================================================================================
    with st.spinner("Initialising methods ............"):


        # ==================
        # EDA of product
        # ==================
        def showEda(df):
            pr = ProfileReport(df, explorative=True)
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)

        # ==================
        # Load & Clean Data
        # ==================

        def assign_star_label(row):
            return 'positive' if row['star_rating'] > 2 else 'negative'

        @st.cache_data
        def data_load_clean_df():
            df = pd.read_csv(FILE_PATH, on_bad_lines='skip')
            df = df[['customer_id','product_title','star_rating','review_body','product_id']]
            df = df[~df.duplicated(subset='review_body')]                                   #Remove duplicates
            df = df.apply(lambda row: row[df['star_rating'].isin(['1','2','3','4','5'])])   # Remove date fields inside star_rating
            df['star_rating']=df['star_rating'].astype('int64')                             # Convert data type for star_rating
            df['star_rating_label'] = df.apply(assign_star_label, axis=1)                   # Apply the function to create the 'label' column
            df['review_body'] = df['review_body'].apply(lambda x : str(x))                  # Convert text inputs to STRING
            df['review_body'] = df['review_body'].apply(lambda x : x[:512])                 # Limit length of string
            return df.reset_index(drop=True)


        # ======================
        # Assign Polarity Score
        # ======================
        def polarity_scores_roberta(review):
            encoded_text = roberta_tokenizer(review, return_tensors='pt').to(device)
            with torch.no_grad():
                output = roberta_model(**encoded_text)
        #     scores = output[0][0].detach().numpy() # FOR CPU
            #scores = softmax(output.logits.detach().cpu().numpy()) # CONVERT from GPU to CPU
            #scores = softmax(scores[0])
            scores = torch.softmax(output.logits[0], dim=0).cpu().detach().numpy() #New code
            scores_dict = {
                'roberta_negative' : scores[0],
                'roberta_positive' : scores[1]
            }
            return scores_dict

        # ==================
        # Summarize Text
        # ==================
        def text_summarizer(review):
            batch = pegasus_tokenizer(review, 
                                      truncation=True, 
                                      padding="longest", 
                                      max_length=1024, 
                                      return_tensors="pt"
                                      ).to(device)
            with torch.no_grad():
                translated = pegasus_model.generate(**batch,
                                                    max_length=100,
                                                    min_length=50,
                                                    length_penalty=2.0,
                                                    num_beams=4,
                                                    early_stopping=True)
            tgt_text = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)
            summary_dict = {"summary":tgt_text[0]}
            return summary_dict

        # =======================================================
        # Define a function to assign labels based on star rating
        # =======================================================
        def assign_label(row):
            if row['roberta_positive'] > row['roberta_negative']:
                return 'positive'
            else:
                return 'negative'

        # =======================================================
        # Summarize batch of summaries together
        # =======================================================
        @st.cache_data
        def data_summarizer(df, marker, summary_count):
            summaries = []
            marker   = 'positive' if marker==1 else 'negative'
            df_new   = df[(df['star_rating_label']==marker) & (df['roberta_rating_label']==marker)]
            df_new = df_new[~df_new.duplicated(subset=["review_body","summary"])]
            sentence = df_new.sort_values(['roberta_positive','Rouge_1','Rouge_2','Rouge_L'],ascending=[False, False,False,False])['summary'].reset_index(drop=True) if marker==1 \
                        else df_new.sort_values(['roberta_negative','Rouge_1','Rouge_2','Rouge_L'],ascending=[False, False,False,False])['summary'].reset_index(drop=True)
            print(sentence)
            print(f"Sentence len :{len(sentence)}")
            count=0
            for i in range(0,len(sentence),10):
                if(count==summary_count):
                    break
                else:
                    chunk = sentence[i:i + 10]
                    joined_sentence = ' '.join(chunk)
                    print(f"JOINED SENTENCE :{joined_sentence}\n\n\n")
                    summaries.append(text_summarizer(joined_sentence[:512])["summary"])
                    count+=1
            print(f"SUMMARY IS:{summaries}\n")
            return summaries

        # ==========================================================
        # Convert the array to a markdown string with bullet points
        # ==========================================================
        def bullet_markdown(array):
            return "\n".join(f"- {item}" for item in array)

        # ==========================================================
        # Get rows with same rating labels
        # ==========================================================
        def getMatchCols(df,value):
            marker = "positive" if value == 1 else "negative"
            df_new = df[(df['star_rating_label']==marker) & (df['roberta_rating_label']==marker)]
            if df_new.shape[0]>0:
                return df_new.sort_values(['roberta_positive','Rouge_1','Rouge_2','Rouge_L'],ascending=[False,False,False,False])['review_body'].values
            else:
                return [f"No {marker} reviews available"]

    # =========================================================================================================================
    # ================================================= LOADING OF THE DATA ===================================================
    # =========================================================================================================================

    ## Load & Clean Data
    with st.spinner("Loading the data ............"):
        df = data_load_clean_df()
    loaded_df = df.copy()
    if(not loaded_df.empty):
        st.header("The Dataframe loaded is shown below :")
        st.dataframe(loaded_df)
    # Controlling the sidebar for loaded DF and new DF with selected product
    productDataframeCheck = False

    # =========================================================================================================================
    # ================================================= LIST OF ALL PRODUCTS ==================================================
    # =========================================================================================================================
    with st.spinner("Loading list of products ............"):
        time.sleep(2)
        prod_ids = df['product_id'].unique()

    # =========================================================================================================================
    # ================================================= CHOOSE A PRODUCT ======================================================
    # =========================================================================================================================

    # Create a dual slider to select the range of product ids to display
    st.markdown("---")
    st.subheader("Step 0 : Choose a product")

    # Group the dataframe by product_id and count the number of rows for each product_id
    grouped_df = df.groupby("product_id").size().reset_index(name="count")

    # Create a slider in streamlit with min value as 0, and max value as max_rows
    slider_value = st.select_slider("Slide to select the number of rows(shows products with that many number of rows)", options=sorted(grouped_df['count'].unique()),value=max(grouped_df['count']))

    # Filter the grouped dataframe by the slider value and get the product_id column as a list
    filtered_df = grouped_df[grouped_df["count"] == slider_value]["product_id"].tolist()

    # Create a select box in streamlit with the filtered list of product_id
#     st.write(f"There {if len(filtered_df)>1 "are" else "is"} {len(filtered_df)} {if len(filtered_df)>1 "products" else "product"} with {slider_value} rows")
    if(len(filtered_df)>1):
        st.write(f"There are {len(filtered_df)} products with {slider_value} rows")
    else:
        st.write(f"There is ONLY {len(filtered_df)} product with {slider_value} rows")
    selected_product_id = st.selectbox("Select the product_id", filtered_df)

    #Selected product dataframe
    preview_df = df.loc[df['product_id']==selected_product_id].reset_index(drop=True)
    
    label_counts = preview_df['star_rating_label'].value_counts()
    label_counts_pos = label_counts.get('positive', 0)
    label_counts_neg = label_counts.get('negative', 0)

    if(not preview_df.empty):
        prod_name = preview_df['product_title'][0]

        # Display the selected product id
        st.markdown("---")
        st.subheader("Step 1 : Product Details :")
        st.write(f'Product Name : {prod_name}')
        st.write(f'Product ID   : {selected_product_id} ')
        st.write(f'Total Rows   : {preview_df.shape[0]}')
        st.write(f"Positive :{label_counts_pos}")
        st.write(f"Negative :{label_counts_neg}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Selected device for processing is (CPU/GPU) : {device.upper()}")
        if device=="cuda":
            st.write(f"Selected GPU: {tf.config.list_physical_devices('GPU')}")

    #================================================================
    # Use the condition to control the display of the radio buttons
    #================================================================
    if(not preview_df.empty):
        productDataframeCheck = True

    if (not productDataframeCheck):
        option = st.sidebar.radio("Select an option", ["None","Show EDA"])
    else:
        option = st.sidebar.radio("Select an option", ["None","Show EDA", "Product EDA"])

    if(option=="Show EDA"):
        showEda(loaded_df)
    elif option=="Product EDA":
        showEda(preview_df)

    if st.button('Confirm Product'):
        df = df.loc[df['product_id']==selected_product_id].reset_index(drop=True)
        st.markdown("---")
        st.subheader("Step 2 : Dataframe with chosen product :")
        st.dataframe(df)

        df_rows = df.shape[0] #Show number of rows

        # =========================================================================================================================
        # ================================================ PRE-TRAINED MODEL ======================================================
        # =========================================================================================================================
        st.markdown("---")
        st.subheader("Step 3 : Initialising the models & running operation")

        # ROBERTA Model
        with st.spinner("Initializing RoBERTa Model ............"):
            roberta_tokenizer, roberta_model = load_model_and_tokenizer("siebert/sentiment-roberta-large-english")
            roberta_model.to(device)

        # PEGASUS Model
        with st.spinner("Initializing Pegasus Model ............"):
            pegasus_tokenizer, pegasus_model = load_model_and_tokenizer("google/pegasus-large")
            pegasus_model.to(device)

        st.success("Models successfully loaded")
        # =========================================================================================================================
        # ================================================ RUN MODEL ON DATA ======================================================
        # =========================================================================================================================

        # Sentimental Analysis & Text Summarization

        res = {}
        summaries = {}
        rouge_1 = {}
        rouge_2 = {}
        rouge_L = {}
        broken_ids = []

        with st.spinner("Operation in progress ............"):

            progress_bar_analysis = st.progress((0/len(df))*100, text="Please wait......... 0%")

            progress_percent = 0
            progress_text = f"Please wait......... {float(progress_percent):.2f}%"


            for i, row in df.iterrows():
                #Calculate percentage length
                progress_percent = (i/len(df))*100
                progress_text = f"Please wait......... {progress_percent:.2f}%"
                progress_bar_analysis.progress(int(progress_percent+1), text=progress_text)

                # Process Sentimental Analysis
                text = row['review_body']
                myid = row['customer_id']
                roberta_result = polarity_scores_roberta(text)
                both = {**roberta_result}
                res[myid] = both

                # Process Summaries
                summary_result = text_summarizer(text)
                summaries[myid] = {**summary_result}

                # Calculate Rouge Scores
                original_text = row['review_body']
                generated_summary = summary_result['summary']    
                scores = rouge.get_scores(generated_summary, original_text)[0]

                # Store Rouge scores in respective dictionaries
                rouge_1[myid] = {"rouge-1": scores['rouge-1']['f']}
                rouge_2[myid] = {"rouge-2": scores['rouge-2']['f']}
                rouge_L[myid] = {"rouge-L": scores['rouge-l']['f']}

            progress_bar_analysis.progress(int(100), text="Completed......... 100%")
        st.success("Operation Completed")

        with st.spinner("Merging in progress ............"):
            # Merge dataframes
            results_df = pd.DataFrame(res).T
            results_df['summary'] = (pd.DataFrame(summaries).T)['summary'].values #Add summary column
            results_df['Rouge_1'] = pd.DataFrame(rouge_1).T[:].values
            results_df['Rouge_2'] = pd.DataFrame(rouge_2).T[:].values
            results_df['Rouge_L'] = pd.DataFrame(rouge_L).T[:].values
            results_df = results_df.reset_index().rename(columns={'index': 'customer_id'})
            results_df = results_df.merge(df, how='left')
            results_df['roberta_rating_label'] = results_df.apply(assign_label, axis=1) # Apply the function to create the 'label' column
            st.markdown("---")
            st.subheader("Step 4 : Dataframe after operation")

        with st.spinner("Matching Columns in progress ............"):
            # prod_a = results_df.loc[results_df['product_id']=='B00J46XO9U']
            prod_a = results_df.copy()
            prod_a = prod_a[prod_a['star_rating_label'] == prod_a['roberta_rating_label']]
            prod_a = prod_a.sort_values(['Rouge_1','Rouge_2','Rouge_L'],ascending=[False,False,False]).reset_index(drop=True)
        st.dataframe(prod_a)
        
        # Positive Reviews Dataframe
        prod_a_positives = prod_a[(prod_a['star_rating_label']=='positive') & (prod_a['roberta_rating_label']=='positive')].sort_values(['Rouge_1','Rouge_2','Rouge_L'],ascending=[False,False,False]).reset_index(drop=True)
        st.subheader("Positive Reviews")
        st.dataframe(prod_a_positives)
        
        # Negative Reviews Dataframe
        prod_a_negatives = prod_a[(prod_a['star_rating_label']=='negative') & (prod_a['roberta_rating_label']=='negative')].sort_values(['Rouge_1','Rouge_2','Rouge_L'],ascending=[False,False,False]).reset_index(drop=True)
        st.subheader("Negative Reviews")
        st.dataframe(prod_a_negatives)

        # =========================================================================================================================
        # ============================================= EVALUATION METRICS ======================================================
        # =========================================================================================================================

        # RUN only if NUMBER OF ROWS > 4
        if(df_rows>4):
            with st.spinner("Creating confusion matrix...."):
                st.markdown("---")
                st.subheader("Step 5. - Confusion Matrix")
                # Sample confusion matrix (replace this with your actual data)
                conf_df = results_df.copy()
                actual_labels = conf_df['star_rating_label']
                predicted_labels = conf_df['roberta_rating_label']

                # Create the confusion matrix
                cm_a = confusion_matrix(actual_labels, predicted_labels)

                # Display the confusion matrix using seaborn
                st.set_option('deprecation.showPyplotGlobalUse', False)
                sns.heatmap(cm_a, annot=True, fmt='d')
                st.pyplot()

                # Extract true positives, false positives, false negatives, true negatives
                tn, fp, fn, tp = cm_a.ravel()

                # Calculate accuracy
                accuracy = accuracy_score(actual_labels, predicted_labels)

                # Calculate precision, recall, and F1 score
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)

                st.write(f"Accuracy :{accuracy*100:.2f} | Precision :{precision:.2f} | Recall:{recall:.2f} | F1-Score:{f1:.2f}")

        # =========================================================================================================================
        # ============================================= SUMMARY OF PRODUCT =======================================================
        # =========================================================================================================================
        st.markdown("---")
        st.subheader("Step 6 : Summary of product")
        choice = 10#st.number_input("Choose number of summaries", 0, 10)

        # POSITIVE SUMMARIES
        st.header("Positive Reviews Summary")
        if(df_rows<=10):
            st.markdown(bullet_markdown(getMatchCols(prod_a,1)))
        else:
            with st.spinner("Generating Positive Summaries ............"):
                sum_list_pos = data_summarizer(prod_a,1,choice)
                st.markdown(bullet_markdown(sum_list_pos))

        # NEGATIVE SUMMARIES
        st.header("Negative Reviews Summary")
        if(df_rows<=10):
            st.markdown(bullet_markdown(getMatchCols(prod_a,0)))
        else:
            with st.spinner("Generating Negative Summaries ............"):
                sum_list_neg =data_summarizer(prod_a,0,choice)
                st.markdown(bullet_markdown(sum_list_neg))