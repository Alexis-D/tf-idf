# What?

This is a quick 'n' dirty implementation of tf*idf using python, nltk & redis.

# Usage

1st you need to `pip install -r requirements.txt` (in a virtualenv :)).

To populate Redis:

    for i in nltk_data/corpora/gutenberg/*; do python tfidf.py add $i; done

And then you can search for stuff like this:

    python tfidf.py search 'hamlet shakespeare'
    ('/home/alexis/search/nltk_data/corpora/gutenberg/shakespeare-hamlet.txt', 0.004095621599867187)
    ('/home/alexis/search/nltk_data/corpora/gutenberg/chesterton-ball.txt', 4.947557430566511e-05)
    ('/home/alexis/search/nltk_data/corpora/gutenberg/austen-sense.txt', 1.4857007758374496e-05)
    ('/home/alexis/search/nltk_data/corpora/gutenberg/melville-moby_dick.txt', 1.3480923177778739e-05)

