import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == '__main__':
    with pd.HDFStore('data/train-clarin-1-normalized.hdf5', mode='r') as store:
        clarin = store['references']

    with pd.HDFStore('data/train-jurisdic-1-normalized.hdf5', mode='r') as store:
        jurisdic = store['references']

    vectorizer = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        analyzer='char_wb'
    )
    corpus = [' '.join(clarin.transcript),
              ' '.join(jurisdic.transcript)]
    X = vectorizer.fit_transform(corpus)
    names = vectorizer.get_feature_names()
    df = pd.DataFrame(X.T.todense(), columns=['clarin', 'jurisdic'], index=names)
    df /= df.sum(axis=0)
