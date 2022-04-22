import praw,pandas,json
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
import ssl
import ETM
import numpy,scipy
import datetime
#from simpletransformers.language_representation import RepresentationModel

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from pprint import pprint
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

class Reddit:

    def __init__(self,
                date=datetime.datetime.now().strftime("%m_%d_%Y"),
                user="Scraper_FinancialIndices_Alexandre_1.0",
                topic="Ukraine",
                top=100,
                attributes=['headlines','id','author','created_utc','score','upvote_ratio','url']):
        self.user_agent=user

        if topic==None:
            return

        self.date=date

        if self.date<datetime.datetime.now().strftime("%m_%d_%Y"):
            self.load()

        with open('/Users/alexandreprofessional2/Desktop/key/Reddit/credentials.json', 'r') as f:
            credentials = json.load(f)
        

        reddit=praw.Reddit(
            client_id=credentials['client_id'],
            client_secret=credentials['secret_id'],
            user_agent=user
        )

        dict_data={column:list() for column in attributes}

        for submission in reddit.subreddit(topic).hot(limit=top):
            dict_data['headlines'].append(submission.title)
            dict_data['id'].append(submission.id)
            dict_data['author'].append(submission.author)
            dict_data['created_utc'].append(submission.created_utc)
            dict_data['score'].append(submission.score)
            dict_data['upvote_ratio'].append(submission.upvote_ratio)
            dict_data['url'].append(submission.url)

        self.redit_data=pandas.DataFrame(dict_data)
        
    def save(self):
        name=[n for n,v in globals().items() if v == self][0]
        date = datetime.datetime.now().strftime("%m_%d_%Y")
        self.redit_data.to_csv('./Data/Class/Reddit/'+name+'_'+date+'.csv',header=True,encoding='utf-8',index=False)

    def load(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.redit_data=pandas.read_csv(filepath_or_buffer='./Data/Class/Reddit/'+name+'_'+self.date+'.csv',header=True,encoding='utf-8',index=False)

    
    def topic_model_ETM(self,num_topics = 10):
        pass
    
    def topic_model_LDA(self,num_topics = 10):
        """
        Original version found on towardsdatascience from Shashank Kapadia
        https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
        """
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use','russia','russian','ukraine','ukranian'])

        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        data = self.redit_data['headlines'].tolist()

        data_words = list(sent_to_words(data))
        data_words = remove_stopwords(data_words)

        id2word = corpora.Dictionary(data_words)
        texts = data_words
        corpus = [id2word.doc2bow(text) for text in texts]

        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics)

        self.topic_model=lda_model

        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]

        self.topic_list=lda_model.print_topics()

        aad=list()
        for topic in doc_lda:
            aad.append(topic)
        
        for i in range(num_topics):
            self.redit_data['Topic '+str(i+1)]=[0]*self.redit_data.shape[0]

        for j in range(len(aad)):
            for x in aad[j]:
                ind=list(self.redit_data.columns).index('Topic '+str(x[0]+1))
                self.redit_data.iloc[j,ind]=x[1]
        
        

    def sentiment_analysis(self):
        sia=SIA()
        self.redit_data['positive']=[0]*self.redit_data.shape[0]
        self.redit_data['neutral']=[0]*self.redit_data.shape[0]
        self.redit_data['negative']=[0]*self.redit_data.shape[0]
        for i in range(self.redit_data.shape[0]):
            pol_score=sia.polarity_scores(self.redit_data.at[i,'headlines'])
            ind_pos=list(self.redit_data.columns).index('positive')
            ind_neu=list(self.redit_data.columns).index('neutral')
            ind_neg=list(self.redit_data.columns).index('negative')
            self.redit_data.iloc[i,ind_pos]=pol_score['pos']
            self.redit_data.iloc[i,ind_neu]=pol_score['neu']
            self.redit_data.iloc[i,ind_neg]=pol_score['neg']      

    def get_neutral(self):
        self.neutral_indices=self.redit_data[self.redit_data['neutral']>0.8]

    def language_model(self):
        import torch
        from transformers import BertTokenizer, BertModel

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        marked_text= ' '.join(["[CLS] " + text + " [SEP]" for text in self.redit_data['headlines'].values])

        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        """print(tokenized_text)
        print(indexed_tokens)
        print(segments_ids)
        exit()"""

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True)  
        model.eval()

        print(model.eval())

        with torch.no_grad():

            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        layer_i = 0
        print ("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0
        print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0
        print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

        token_vecs_cat = []
        token_embeddings = torch.stack(hidden_states, dim=0)
        print(token_embeddings.size())
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        print(token_embeddings.size())
        token_embeddings = token_embeddings.permute(1,0,2)
        print(token_embeddings.size())

        for token in token_embeddings:
            print(token)
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec)

        print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

    def map_domains(self,type='GPE'):
        import spacy
        from spacy import displacy
        from collections import Counter
        nlp = spacy.load("en_core_web_sm")
        domains,domains_inv={},{}
        for id,text in enumerate(self.redit_data['headlines'].values.tolist()):
            doc=nlp(text)
            domains[id]=[X.text for X in doc.ents if str(X.label_)==type]
        for key,value in domains.items():
            for domain in value:
                if domain in domains_inv.keys():
                    domains_inv[domain].add(key)
                else:
                    domains_inv[domain]=set([key])
        self.domains_inv=domains_inv
        

    def Analyse_on_several_days(self,reddit):
        int_val=({topic:val for topic,val in self.topic_model.print_topics()})
        ext_val=({topic:val for topic,val in reddit.topic_model.print_topics()})

        assert(len(int_val)==len(ext_val))
        n=len(int_val)

        def dist(x_,y_):
            def vectorize(z):
                tmpr=z.split("*")
                model=language_model()
                return(tmpr[0]*model.encode_word(tmpr[1]))
            x,y=x_.split("+"),y_.split("+")
            x_embedding,y_embedding=map(vectorize,x),map(vectorize,y)
            spatial.distance.cosine(x_embedding, y_embedding)

        dist_matrix=numpy.zeros(shape=(n,n))
        for i,int_value in int_val.items():
            for j,ext_value in ext_val.items():
                dist_matrix[i,j]=dist(int_value,ext_value)
        
        return(scipy.optimize.linear_sum_assignment(dist_matrix))



    def print_map(self):
        import googlemaps
        from datetime import datetime

        with open('/Users/alexandreprofessional2/Desktop/key/GCP/credentials_API.json', 'r') as f:
            credentials_API = json.load(f)
            print(credentials_API)

        gmaps = googlemaps.Client(key=credentials_API['key'])
        geocode_result = gmaps.geocode(self.domains_inv.keys[0])

if __name__=='__main__':
    Ukraine=Reddit(topic="Ukraine",
                   top=100,
                   attributes=['headlines','id','author','created_utc','score','upvote_ratio','url'])
    Ukraine.topic_model_LDA()
    Ukraine.sentiment_analysis()
    Ukraine.map_domains()
    Ukraine.language_model()
    Ukraine.save()
