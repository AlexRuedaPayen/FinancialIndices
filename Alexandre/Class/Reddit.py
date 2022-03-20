import praw,pandas,json
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from pprint import pprint
#nltk.download('stopwords')
#from nltk.corpus import stopwords

class Reddit:

    def __init__(self,
                user="Scraper_FinancialIndices_Alexandre_1.0",
                topic="Ukraine",
                top=100,
                attributes=['headlines','id','author','created_utc','score','upvote_ratio','url']):
        self.user_agent=user

        if topic==None:
            return

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

        print(dict_data['created_utc'])
        
    def save(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.redit_data.to_csv('./Data/Reddit/'+name+'.csv',header=True,encoding='utf-8',index=False)

    def load(self):
        name=[n for n,v in globals().items() if v == self][0]
        self.redit_data=pandas.read_csv(filepath_or_buffer='./Data/Reddit/'+name+'.csv',header=True,encoding='utf-8',index=False)

    def topic_model_LDA(self,num_topics = 10):
        
        """
        Original version found on towardsdatascience from Shashank Kapadia
        https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
        """
        #stop_words = stopwords.words('english')
        #stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        #print(self.redit_data.columns)
        data = self.redit_data['headlines'].tolist()
        #print(data)

        data_words = list(sent_to_words(data))
        #data_words = remove_stopwords(data_words)

        id2word = corpora.Dictionary(data_words)
        texts = data_words
        corpus = [id2word.doc2bow(text) for text in texts]

        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics)
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]

        aad=list()
        for topic in doc_lda:
            aad.append(topic)
        
        for i in range(num_topics):
            self.redit_data["Topic "+str(i+1)]=0
            for j in range(len(aad)):
                for x in aad[j]:
                    #(self.redit_data).set_value(j,"Topic "+str(x[0]+1), x[1])
                    (self.redit_data).at[j,"Topic "+str(x[0]+1)]=x[1]
                    #print((self.redit_data)["Topic "+str(x[0]+1)][j])
                    #print("Topic "+str(x[0]+1),j,x[1])
                    #exit()

        import matplotlib.pyplot as plt
        import datetime

        self.redit_data
        
        t=(list(map(lambda x:datetime.datetime.fromtimestamp(x),self.redit_data['created_utc'].tolist())))
        serie=[ self.redit_data['Topic 3'][i]*self.redit_data['score'][i] for i in range(self.redit_data.shape[0]) ]

        plt.plot_date(t,serie,linestyle='solid')
        plt.show()
        
        

    def sentiment_analysis(self):
        pass



Ukraine=Reddit()
Ukraine.save()
Ukraine.topic_model_LDA()
Ukraine.save()

