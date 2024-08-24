import pandas as pd
import numpy as np
import torch
from transformers import  GPT2Tokenizer, GPT2Model,BertModel,BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the GPT-2 and BERT models and their tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_id = 0 if torch.cuda.is_available() else -1
gptModel = GPT2Model.from_pretrained("ytu-ce-cosmos/turkish-gpt2").to(device)
gptTokenizer = GPT2Tokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2")
gptTokenizer.pad_token = "[PAD]"

bertModel = BertModel.from_pretrained("ytu-ce-cosmos/turkish-medium-bert-uncased")
bertTokenizer = BertTokenizer.from_pretrained("ytu-ce-cosmos/turkish-medium-bert-uncased")

# Load the dataset
data = pd.read_excel('soru_cevap.xlsx')
data.drop('tercih', axis=1, inplace=True) #Redundant column
df = data.sample(n=100) #Random sampling
df.reset_index(drop=True, inplace=True) 

#Calculate embeddings for each cell in the dataset and put into a matrix
def calculate_embeddings(df,model,tokenizer,embeddingSize):
    embeddings = np.zeros((len(df), 3,embeddingSize)) # Embedding size is the vector size of model's tokenizer. 768 -> GPT-2 , 512 -> BERT
    
    for i, row in df.iterrows():
        print(i)
        for j, cell_value in enumerate(row):
            inputs = tokenizer(cell_value, return_tensors='pt', padding=True, truncation=True, max_length=embeddingSize)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[i][j] = embedding
    return embeddings

gptEmbeddings = calculate_embeddings(df,gptModel,gptTokenizer,768)
bertEmbeddings = calculate_embeddings(df,bertModel,bertTokenizer,512)
    
# Reducing and visualizing data embeddings to 2 dimensions
def visualize_embeddings(embeddings,title):
    # Get embeddings for each column
    question_embeddings = embeddings[:, 0, :]
    human_answer_embeddings = embeddings[:, 1, :]
    machine_answer_embeddings = embeddings[:, 2, :]

    # Concatenation of embeddings
    all_embeddings = np.concatenate((question_embeddings, human_answer_embeddings, machine_answer_embeddings), axis=0)

    # Create tags (0 for questions, 1 for human answers, 2 for machine answers)
    labels = np.repeat([0, 1, 2], len(df))
    
    # Apply TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings.reshape(all_embeddings.shape[0], -1))

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.title(title)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Soru', 'İnsan Cevabı', 'Makine Cevabı'])
    plt.show()

visualize_embeddings(gptEmbeddings,"GPT")
visualize_embeddings(bertEmbeddings,"BERT")

# Calculate similarity using cosine sim. and count top1 and top5 scores
def getTopAccuracy(embedding1,embedding2,columnName):
    # Calculate all similarities at once
    similarities = cosine_similarity(embedding1, embedding2)
    top1 = 0
    top5 = 0
    for index, element in enumerate(df[columnName]):
        # Get similarities for element at index
        similarity_scores = similarities[index]
        # Sort similarities then take first 5
        most_similar_indices = similarity_scores.argsort()[::-1][:5]
        top1 += 1 if df[columnName][most_similar_indices[0]] == element else 0
        top5 += 1 if element in [df[columnName][index] for index in most_similar_indices] else 0
    # Convert to a number between 0 and 1
    return top1 / len(df), top5 / len(df)


# Separate the initially calculated embeddings for the question, human answer and machine answer from the matrix
gpt_question_embeddings = gptEmbeddings[:, 0, :]
gpt_human_answer_embeddings = gptEmbeddings[:, 1, :]
gpt_machine_answer_embeddings = gptEmbeddings[:, 2, :]

#Calculate similarities and top-1/top-5 accuracies for questions to human answers
top1_human_accuracy, top5_human_accuracy = getTopAccuracy(gpt_question_embeddings,gpt_human_answer_embeddings,"insan cevabı")

# Calculate similarities and top-1/top-5 accuracies for questions to machine answers
top1_machine_accuracy, top5_machine_accuracy = getTopAccuracy(gpt_question_embeddings,gpt_machine_answer_embeddings,"makine cevabı")

# Calculate similarities and top-1/top-5 accuracies for human answers to questions
top1_human_question_accuracy, top5_human_question_accuracy = getTopAccuracy(gpt_human_answer_embeddings,gpt_question_embeddings,"soru")

# Calculate similarities and top-1/top-5 accuracies for machine answers to questions
top1_machine_question_accuracy, top5_machine_question_accuracy = getTopAccuracy(gpt_machine_answer_embeddings,gpt_question_embeddings,"soru")

# Visualize top1 and top5 accuracies of GPT model
accuracies = [
    top1_human_accuracy, top5_human_accuracy,
    top1_machine_accuracy, top5_machine_accuracy,
    top1_human_question_accuracy, top5_human_question_accuracy,
    top1_machine_question_accuracy, top5_machine_question_accuracy
]

# Naming columns
labels = [
    'Top 1 Question - Human Ans.', 'Top 5 Question - Human Ans.',
    'Top 1 Question - Machine Ans.', 'Top 5 Question - Machine Ans.',
    'Top 1 Human Ans. - Question', 'Top 5 Human Ans. - Question',
    'Top 1 Machine Ans. - Question', 'Top 5 Machine Ans. - Question'
]

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, accuracies, color='skyblue')
plt.xlabel('Accuracy Type')
plt.ylabel('Accuracy')
plt.title('GPT Accuracy Scores')
plt.ylim(0, 1) 
plt.xticks(rotation=45)
plt.tight_layout()
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{accuracy:.3f}', ha='center', va='bottom')
plt.show()


# Separate the initially calculated embeddings for the question, human answer and machine answer from the matrix
bert_question_embeddings = bertEmbeddings[:, 0, :]
bert_human_answer_embeddings = bertEmbeddings[:, 1, :]
bert_machine_answer_embeddings = bertEmbeddings[:, 2, :]

#Calculate similarities and top-1/top-5 accuracies for questions to human answers
top1_human_accuracy, top5_human_accuracy = getTopAccuracy(bert_question_embeddings,bert_human_answer_embeddings,"insan cevabı")

# Calculate similarities and top-1/top-5 accuracies for questions to machine answers
top1_machine_accuracy, top5_machine_accuracy = getTopAccuracy(bert_question_embeddings,bert_machine_answer_embeddings,"makine cevabı")

# Calculate similarities and top-1/top-5 accuracies for human answers to questions
top1_human_question_accuracy, top5_human_question_accuracy = getTopAccuracy(bert_human_answer_embeddings,bert_question_embeddings,"soru")

# Calculate similarities and top-1/top-5 accuracies for machine answers to questions
top1_machine_question_accuracy, top5_machine_question_accuracy = getTopAccuracy(bert_machine_answer_embeddings,bert_question_embeddings,"soru")

#Visualize top1 and top5 accuracies of BERT model
accuracies = [
    top1_human_accuracy, top5_human_accuracy,
    top1_machine_accuracy, top5_machine_accuracy,
    top1_human_question_accuracy, top5_human_question_accuracy,
    top1_machine_question_accuracy, top5_machine_question_accuracy
]

# Naming Columns
labels = [
    'Top 1 Question - Human Ans.', 'Top 5 Question - Human Ans.',
    'Top 1 Question - Machine Ans.', 'Top 5 Question - Machine Ans.',
    'Top 1 Human Ans. - Question', 'Top 5 Human Ans. - Question',
    'Top 1 Machine Ans. - Question', 'Top 5 Machine Ans. - Question'
]

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, accuracies, color='skyblue')
plt.xlabel('Accuracy Type')
plt.ylabel('Accuracy')
plt.title('BERT Accuracy Scores')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05, f'{accuracy:.3f}', ha='center', va='bottom')
plt.show()