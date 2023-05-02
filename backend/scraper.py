import twint

def webscraper(input):
    # Configure
    c = twint.Config()
    c.Search = input
    c.Pandas = True
    c.Lang = "en"
    c.Limit = 20
    c.Hide_output = True

    # Run
    twint.run.Search(c)

    tweets_df = twint.storage.panda.Tweets_df

    return tweets_df

if __name__ == "__main__":
    
    print('Enter your text you want scrapped:')

    custom_input = input()

    # custom_input = "#TLOU"

    tweets_df = webscraper(custom_input)

    print(tweets_df['tweet'])