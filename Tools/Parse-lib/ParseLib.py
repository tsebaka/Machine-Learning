import numpy as np
import nltk
import bs4 as bs
import re
import urllib.request
import warnings

from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

class parse_and_prepare():
    def __init__(self, EOS=False):
        self.vocab = []
        self.texts = []
        self.text = []
        self.EOS = EOS

    def parse(self, sites):
        for site in sites:
            scrapped_data = urllib.request.urlopen(site)
            article = scrapped_data.read()

            parsed_article = bs.BeautifulSoup(article, 'lxml')
            paragraphs = parsed_article.find_all('p')
            article_text = ""
            for p in paragraphs:
                article_text += p.text

            processed_article = article_text.lower()
            if self.EOS:
                processed_article = re.sub('[^a-zA-Z.]', ' ', processed_article)
            else:
                processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)

            processed_article = re.sub(r'\s+', ' ', processed_article)
            all_sentences = nltk.sent_tokenize(processed_article)

            new_text = [nltk.word_tokenize(sent) for sent in all_sentences]
            self.texts.append(new_text)
            
            for sentence in new_text:
                for word in sentence:
                    if word not in stopwords.words('english'):
                        self.text.append(word)

        self.vocab = np.unique(self.text)
        return self.text, self.texts, self.vocab
