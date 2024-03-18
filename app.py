from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim import models
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt

# load model
app = Flask(__name__)
lda_model = models.LdaMulticore.load("model/model_104_(9, 0.01, 0.04)")
dictionary = Dictionary.load("model/dictionary")

# preprocessing function
def case_folding(input_text):
    case_folded = input_text.lower()
    return case_folded

def remove_punctuation(input_text):
    traslation_table = str.maketrans("", "", string.punctuation)
    removed = input_text.translate(traslation_table)
    return removed

def tokenization(input_text):
    result = word_tokenize(input_text)
    return result

def stopword_removal(input_text):
    stopword_remover = StopWordRemoverFactory()
    result = [token for token in input_text if token.lower() not in stopword_remover.get_stop_words()]
    return result

def stemming(input_text):
    stemmer = StemmerFactory().create_stemmer()
    result = [stemmer.stem(token) for token in input_text]
    return result

def pp_text(text):
    pp = case_folding(text)
    pp = remove_punctuation(pp)
    pp = tokenization(pp)
    pp = stopword_removal(pp)
    pp = stemming(pp)
    return pp

# home
@app.route('/')
def home():
    return render_template('index.html')

# Post
@app.route('/predict',methods=['POST'])
def predict():
    try:
        text = [request.form.get("review")]
        text = text[0]

        text_pp = pp_text(text)
        text_pp2 = [text_pp]

        other_corpus = [dictionary.doc2bow(text) for text in text_pp2]
        unseen_doc = other_corpus[0]

        topic_distribution = lda_model.get_document_topics(unseen_doc)

        topic_id = ([key for key, value in topic_distribution])
        for i in range(9):
            if(i not in topic_id):
                topic_distribution = topic_distribution + [(i, 0)]

        # get topic and their top word
        topics_best = lda_model.show_topics(num_topics=-1, num_words=10, formatted=False)

        cluster_topics_best = {}

        # Iterate over the topics and store top words for each cluster
        for topic_id, topic_words in topics_best:
            cluster_topics_best[topic_id] = [word for word, _ in topic_words]

        # find highest topic distribution
        max_tuple = max(topic_distribution, key=lambda x: x[1])
        max_key = max_tuple[0]

        # word = {', '.join(cluster_topics_best[max_key])}

        topic = ["Pola makan dan gizi", "Kehamilan", "Rambut", "Kesehatan Umum", "Kesehatan Wanita", "Penanganan Luka", "Kulit", "Mulut dan Pencernaan", "Anak"]
        word = topic[max_key]

        # Plot the dynamic horizontal bar chart
        plt.figure(figsize=(12, 6))
        plt.barh([key for key, value in topic_distribution], [value for key, value in topic_distribution], color='skyblue', alpha=0.7)
        # plt.barh([key for key in topic], [value for key, value in topic_distribution], color='skyblue', alpha=0.7)
        plt.xlabel('Weight')
        plt.ylabel('Cluster')
        plt.title('Dynamic Topic Distribution for Documents')

        # Set the number of clusters on the y-axis
        # plt.yticks(range(9), [f'Cluster {i}' for i in range(9)])
        plt.yticks(range(9), [f'{i}' for i in topic])
        plt.show

        #save_path = ''
        plt.savefig('./static/images/result.jpg')

        return render_template('predict.html', image_position='{}'.format("<img src=\"/static/images/result.jpg\" alt=\"Image\" width=\"1200\" height=\"600\">"), max_key=max_key, word=word)
    except ValueError:
        print(ValueError)
        print("Oops!  That was no valid number.  Try again...")
        return render_template('index.html', input_text='Terjadi kesalahan')


if __name__ == "__main__":
    app.run(debug=True)