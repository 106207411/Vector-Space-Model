import argparse
import os
from VectorSpace import *
import time


def main(query):
    documents = []
    path = './EnglishNews/'
    files = os.listdir(path)
    for file in files:
        doc = " ".join(open(f"{path}{file}", "r").read().splitlines())
        documents.append(doc)
    
    rank_files(files, query, documents, representation_mode="TF", relevance_mode="Cosine Similarity")
    rank_files(files, query, documents, representation_mode="TF", relevance_mode="Euclidean Distance")
    rank_files(files, query, documents, representation_mode="TF-IDF", relevance_mode="Cosine Similarity")
    rank_files(files, query, documents, representation_mode="TF-IDF", relevance_mode="Euclidean Distance")
    rank_files(files, query, documents, representation_mode="TF-IDF", relevance_mode="Cosine Similarity", pseudo_feedback=True)


def rank_files(files, query, documents, representation_mode="TF", relevance_mode="Cosine Similarity", pseudo_feedback=False):
    start = time.time()
    print("-"*24)
    if pseudo_feedback:
        print(f"Feedback Queries + {representation_mode} Weighting + {relevance_mode}\n")
    else:
        print(f"{representation_mode} Weighting + {relevance_mode}\n")
    print("{:<16s}{}".format("NewsID","Score"))
    print("{:<16s}{}".format("----------","--------"))

    vectorSpace = VectorSpace(documents, representation_mode, relevance_mode)
    ratings = vectorSpace.search([query])
    scores = dict(zip(files, ratings))
    if relevance_mode == "Cosine Similarity":
        sorted_scores = sorted(scores.items(), key=lambda s: s[1], reverse=True)

        #pseudo feedback
        if pseudo_feedback:
            top_newsid, top_score = sorted_scores[0]
            path = './EnglishNews/'
            feedbackdoc = " ".join(open(f"{path}{top_newsid}", "r").read().splitlines())
            ratings = vectorSpace.feedback_search([query], feedbackdoc)
            scores = dict(zip(files, ratings))
            sorted_scores = sorted(scores.items(), key=lambda s: s[1], reverse=True)

    if relevance_mode == "Euclidean Distance":
        sorted_scores = sorted(scores.items(), key=lambda s: s[1], reverse=False)

    for newsid, score in sorted_scores[:10]:
        print("{:<16s}{:.6f}".format(newsid[:-4], score))
    print(f"Data Size: {len(files)}\n")
    stop = time.time()
    print(f'Execution Time: {(stop - start):.2f} seconds\n')  




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Please enter your query.')
    parser.add_argument('--query')
    args = parser.parse_args()
    main(args.query)