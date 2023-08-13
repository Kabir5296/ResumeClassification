# install wheels in your terminal with following commands
# pip install Wheels/pytest-shutil/*.whl
# pip install Wheels/textract/*.whl
# pip install Wheels/transformers/*.whl

#imports
import codecs
import textract
import re
from string import punctuation
import os
import pandas as pd
import shutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

# take input for data input and output
path_to_pdfs = input("Enter the path to resume PDFs: ")

if not os.path.exists(path_to_pdfs):
    print("PDF Directory Doesn't Exist")
else:
    files=os.listdir(path_to_pdfs)
    
output_dir = input("Enter the path for output files to be saved: ")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("A directory has been made for output")


# Create an empty dataframe
df=pd.DataFrame(columns=['path','id','text','category','output_dir'])
df.id = files
df.path=df.id.apply(lambda x: os.path.join(path_to_pdfs,x))

# Load Stopwords from Local
stopwords_file_path = "stopwords/english"

print("Loading Preprocessing Functions.")
def load_stopwords(file_path):
    stopwords = []
    with open(file_path, "r") as file:
        for line in file:
            stopwords.append(line.strip())
    return stopwords

stop_words = load_stopwords(stopwords_file_path)

# Functions Defined
def read_pdf(path):
    textract_text = textract.process(path)
    textract_str_text = codecs.decode(textract_text)
    return preprocess_text(textract_str_text)

def remove_stop_words(sentence): 
    words = sentence.split() 
    filtered_words = [word for word in words if word not in stop_words] 
    return ' '.join(filtered_words)

def preprocess_text(resumeText):
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) # remove non-ascii characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = re.sub(r'[0-9]+', '', resumeText)  #remove numbers
    resumeText = remove_stop_words(resumeText)
    return resumeText.lower()

# Read pdfs into text
print("Reading the pdf files and applying preprocessing...")
df.text=df.path.apply(lambda x: read_pdf(x))
print("Read Complete.")


# Load Model and Tokenizer
print("Loading the models for classification.")
if not os.path.exists("Model"):
    print("The offline model file couldn't be loaded. Loading from Huggingface.")
    print("This requires internet connection")
    model="kabir5297/ResumeClassificationNew"
else:
    model="Model"
model= AutoModelForSequenceClassification.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased") # The tokenizer from the model for some unknown reason is not working. So we're using the xlnet tokenizer instead.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Load Complete.")

id2label=model.config.id2label

# AI Function Define
def classification(example):
    inputs = tokenizer(example,return_tensors='pt',truncation=True,max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim= -1)
    return id2label[int(predictions)]

# Run Inference
print("Running inference. This might take a while. Why don't you have a cup of tea in the meantime?")
df.category=df.text.apply(lambda x: classification(x))
df.output_dir=df.category.apply(lambda x: os.path.join(output_dir,x+'/'))
df.output_dir=df.apply(lambda x: os.path.join(x['output_dir'],x['id']),axis=1)
print("Inference Complete.")


# Create Directories for each unique categories
print("Creating new folders according to categories.")
cat=df.category.unique()
for categories in cat:
    dir_name=(os.path.join(output_dir,categories))
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# Copy the files to new directory
print("Copying....")
df.apply(lambda x: shutil.copyfile(x['path'], x['output_dir']),axis=1)

# Save the csv file
categorized_resumes=pd.DataFrame(columns=['filename','category'])
categorized_resumes.filename=df.id
categorized_resumes.category=df.category
categorized_resumes.to_csv("categorized_resumes.csv")
print("Done doing all the tasks. Gonna sleep now, bye!")
