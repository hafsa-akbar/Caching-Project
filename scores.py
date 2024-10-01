import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import base64
from tqdm import tqdm
from openai import OpenAI
import time
import json
import re
import numpy as np
import pandas as pd
import csv
import shutil
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from anthropic import AnthropicVertex

#########################################################################
key_path = 'caching-436119-3f7e7f2329ed.json'

credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

PROJECT_ID = 'caching-436119'
REGION = 'us-east5'

claude_client = AnthropicVertex(project_id=PROJECT_ID, region=REGION, credentials=credentials)

load_dotenv()
openai_api_key = os.getenv('api_key')
api_key = openai_api_key
gpt_client = OpenAI(api_key=openai_api_key)

###########################################################################

prompt = f"""
Evaluate the potential for replacing Image A with Image B in a general context. Consider the following factors:
a. Theme or Subject Matter:
   How closely do the images align in terms of their primary topic or subject?
   Are there any significant thematic differences that might hinder replacement?
b. Visual Style and Composition:
   Do the images share a similar visual style (e.g., realistic, cartoonish, abstract)?
   Are the compositional elements (e.g., framing, perspective) comparable?
c. Mood and Tone:
   Do the images evoke similar emotions or convey the same tone (e.g., humorous, serious, uplifting)?
d. Target Audience:
   Would both images appeal to the same target audience or are there potential differences in their intended viewers?
e. Contextual Fit:
   Given the article headings and alt text, how well do the images align with the overall context of the article or content?
   Are there any specific visual elements that might be crucial for conveying the intended message?
Consider all of this and rate the similarity on a discrete/ordinal scale from 0 to 4, where:
0: Not replaceable
1: Somewhat replaceable
2: Moderately replaceable
3: Very replaceable
4: Completely replaceable."""

###########################################################################

def extract_number(filename, article_num=False):
    match = re.search(r'image_(\d+)_(\d+)', filename) 
    group_num = 1 if article_num else 2
    return int(match.group(group_num))

def load_image_as_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def read_matrix_from_csv(filepath):
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        matrix = [list(map(float, row)) for row in reader]
    return np.array(matrix)

def find_similar_category(category, train_dir='train', model_type='gpt'):
    categories = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    prompt = f"Which of these categories: {', '.join(categories)} is most similar to '{category}'? You CANNOT choose a category not mentioned in the given list \
    Respond ONLY with the category name without any additional words or punctuation."

    try:
        if model_type == 'gpt':
            response = gpt_client.chat.completions.create(
                model="gpt-4o",  
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            similar_category = response.choices[0].message['content'].strip()
        
        elif model_type == 'claude':
            response = claude_client.messages.create(
                model="claude-3-5-sonnet@20240620",
                max_tokens=10,
                stream = False,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            )
            similar_category = response.content[0].text
    
    except Exception as e:
        print(f"Error in finding similar category: {e}")
        return None
    
    return similar_category

def few_shot(category, static_folder=None, model_type='gpt'):
    if static_folder is None:
        similar_category = find_similar_category(category, model_type=model_type)
        print(f'category similar to {category}: {similar_category}')
        images_path = os.path.join('train', similar_category)
    else:
        images_path = static_folder

    labels_csv = os.path.join(images_path, 'labels.csv')
    similarity_csv = os.path.join(images_path, 'similarity.csv')

    image_paths = sorted(
        [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('jpg', 'jpeg'))],
        key=extract_number
    )
    
    labels = pd.read_csv(labels_csv)
    similarity_scores = read_matrix_from_csv(similarity_csv)
    few_shot_examples = create_examples(image_paths, labels, similarity_scores, model_type)
    
    return few_shot_examples

def create_examples(image_paths, labels, similarity_scores, model_type='gpt'):
    few_shot_examples = []
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            base64_img1 = load_image_as_base64(image_paths[i])
            base64_img2 = load_image_as_base64(image_paths[j])
            
            img1_number = extract_number(image_paths[i])
            img2_number = extract_number(image_paths[j])
            
            alt_text1 = labels.loc[labels['image number'] == img1_number, 'alt'].values[0]
            heading1 = labels.loc[labels['image number'] == img1_number, 'article_heading'].values[0]
            alt_text2 = labels.loc[labels['image number'] == img2_number, 'alt'].values[0]
            heading2 = labels.loc[labels['image number'] == img2_number, 'article_heading'].values[0]

            image1_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img1}"}
            }
            image2_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img2}"}
            }
            
            if model_type == 'claude':
                image1_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_img1
                    }
                }
                image2_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_img2
                    }
                }

            few_shot_examples.append({
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": f"Image A Alt Text: {alt_text1}\nImage A Heading: {heading1}"
                },
                image1_content,
                {
                    "type": "text",
                    "text": f"Image B Alt Text: {alt_text2}\nImage B Heading: {heading2}"
                },
                image2_content,
                {
                    "type": "text",
                    "text": f"{prompt} \nAnswer: {similarity_scores[i][j]}"
                }]
            })
    return few_shot_examples

def compare_images(image_paths, labels, few_shot_examples, model_type='gpt'):

    user_messages = []

    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            article_num1 = extract_number(image_paths[i], article_num=True)
            article_num2 = extract_number(image_paths[j], article_num=True)

            if article_num1 == article_num2:
                user_messages.append('')
                continue

            base64_img1 = load_image_as_base64(image_paths[i])
            base64_img2 = load_image_as_base64(image_paths[j])

            if model_type == 'gpt':
                image_content1 = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img1}"}}
                image_content2 =  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img2}"}}
            elif model_type == 'claude':
                image_content1 = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_img1  
                    }
                }
                image_content2 = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_img2 
                    }
                }
            
            img1_number = extract_number(image_paths[i]) 
            img2_number = extract_number(image_paths[j])
            alt_text1 = labels.loc[(labels['article_number'] == article_num1) & (labels['image number'] == img1_number), 'alt'].values[0]
            heading1 = labels.loc[(labels['article_number'] == article_num1) & (labels['image number'] == img1_number), 'article_heading'].values[0]

            alt_text2 = labels.loc[(labels['article_number'] == article_num2) & (labels['image number'] == img2_number), 'alt'].values[0]
            heading2 = labels.loc[(labels['article_number'] == article_num2) & (labels['image number'] == img2_number), 'article_heading'].values[0]

            user_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Image A Alt Text: {alt_text1}\nImage B Heading: {heading1}"},
                    image_content1,

                    {"type": "text", "text": f"Image A Alt Text: {alt_text2}\nImage B Heading: {heading2}"},
                    image_content2, 

                    {"type": "text",
                    "text": "{prompt}. Only respond with the similarity score. Do NOT include anything else in the answer (not even punctuation)."}
                ]
            })

    responses = []
    for message in tqdm(user_messages, desc="Processing image pairs"):
        if credentials.expired:
            credentials.refresh(Request())
    
        if not message:
            responses.append(0)
            continue
        
        try:    
            if model_type == 'gpt':
                response = gpt_client.chat.completions.create(
                    model="gpt-4o", 
                    messages=few_shot_examples + [message],
                    max_tokens=10
                )
                res = response.choices[0].message['content'].strip()

            elif model_type == 'claude':
                response = claude_client.messages.create(
                    model="claude-3-5-sonnet@20240620",
                    max_tokens=10,
                    messages=[
                        {
                            "role": "user",
                            "content": message['content'],
                        }
                    ],
                    stream=False
                )
                res = response.content[0].text

            score = re.findall(r'-?\d*\.?\d+', res)[0]
            responses.append(int(score))
            time.sleep(15)

        except Exception as e:
            print(e)
            responses.append(0)
        
    return responses

def get_filename(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

def make_matrix(image_paths, similarity_scores):
    image_names = [get_filename(path) for path in image_paths]
    
    n = len(image_paths)
    similarity_matrix = [[4 if i == j else 0 for j in range(n)] for i in range(n)]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            similarity_matrix[i][j] = int(similarity_scores[idx])
            similarity_matrix[j][i] = int(similarity_scores[idx])
            idx += 1
    
    df = pd.DataFrame(similarity_matrix, index=image_names, columns=image_names)
    
    return df

def process_categories(base_dir, static_folder=None, zero_shot=False, model_type='gpt'):
    def get_dirs(base_dir):
        return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    website_folders = get_dirs(base_dir)
    

    for website in website_folders:
        website_dir = os.path.join(base_dir, website)
        category_folders = get_dirs(website_dir)

        for category in category_folders:
            category_dir = os.path.join(website_dir, category)
            labels = pd.read_csv(os.path.join(category_dir, 'image_data.csv'))

            image_paths = sorted(
                [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith(('jpg', 'jpeg'))],
                key=lambda x: (extract_number(x, article_num=True), extract_number(x))
            )
            if len(image_paths) < 2:
                continue

            few_shot_examples = []
            if not zero_shot:
                few_shot_examples = few_shot(category, static_folder, model_type=model_type)

            similarity_scores = compare_images(image_paths, labels, few_shot_examples, model_type=model_type)
            similarity_df = make_matrix(image_paths, similarity_scores)

            few_shot_out = 'with_fewshot' if not zero_shot else f'without_fewshot'
            done_testing_dir = os.path.join(base_dir, '..', f'done_testing/{few_shot_out}', website)
            os.makedirs(done_testing_dir, exist_ok=True)

            shutil.move(category_dir, os.path.join(done_testing_dir, category))
            similarity_df.to_csv(f'pred_labels/{few_shot_out}/{model_type}/{website}_{category}.csv')
            
        shutil.rmtree(website_dir)

def compute_rmse_with_filter(pred_df, true_df, tolerance=0):
    valid_pairs_pred = []
    valid_pairs_true = []
    
    columns = pred_df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            img_i = columns[i].split('_')[1]
            img_j = columns[j].split('_')[1]
            
            if img_i != img_j:
                valid_pairs_pred.append(pred_df.iloc[i, j])
                valid_pairs_true.append(true_df.iloc[i, j])

    if valid_pairs_pred and valid_pairs_true:
        squared_errors = []
        for pred_val, true_val in zip(valid_pairs_pred, valid_pairs_true):
            difference = abs(pred_val - true_val)
            error = max(0, difference - tolerance)
            squared_errors.append(error ** 2)

        if squared_errors:
            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
            return rmse/4.0
    return None

def compute_average_rmse(pred_dir, true_dir, tolerance=0):
    rmse_scores = []

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]

    for pred_file in pred_files:
        parts = pred_file.split('_')
        website_name = parts[0]
        category = "_".join(parts[1:]).replace('.csv', '')

        true_file = os.path.join(true_dir, website_name, f'{website_name} - {category}.csv')

        if os.path.exists(true_file):
            pred_df = pd.read_csv(os.path.join(pred_dir, pred_file), index_col=0)
            true_df = pd.read_csv(true_file, index_col=0)

            if pred_df.shape == true_df.shape:
                rmse = compute_rmse_with_filter(pred_df, true_df, tolerance=tolerance)
                if rmse is not None:
                    rmse_scores.append(rmse)
                    print(f"RMSE for {website_name} - {category}: {rmse}")
                else:
                    print(f"No valid pairs found for {website_name} - {category}")
            else:
                print(f"Shape mismatch for {website_name} - {category}")
        else:
            print(f"True file not found for {website_name} - {category}")

    if rmse_scores:
        avg_rmse = sum(rmse_scores) / len(rmse_scores)
        print(f"Average RMSE: {avg_rmse}")
        return avg_rmse
    else:
        print("No RMSE scores were computed.")
        return None

if __name__ == '__main__':
    test_dir = 'test'
    process_categories(test_dir, zero_shot=True, model_type='claude') 
