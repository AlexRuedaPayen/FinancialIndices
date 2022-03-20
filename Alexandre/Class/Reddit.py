import praw,pandas,json

class Reddit:
    def __init__(self,
                user="Scraper_FinancialIndices_Alexandre_1.0",
                topic="Ukraine",
                top=100,
                attributes=['headlines','id','author','created_utc','score','upvote_ratio','url']):
        self.user_agent=user

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
        self.redit_data.to_csv('')

Ukraine=Reddit('/Users/alexandreprofessional2/')
