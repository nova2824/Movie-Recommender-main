from evaluate_recommender import evaluate_recommender
from streamlit_app import load_and_prepare

def main():
    print("\n===== Loading data and model =====")
    processed, vectorizer, tfidf_matrix = load_and_prepare()

    print("===== Running evaluation =====")
    results = evaluate_recommender(processed, tfidf_matrix, ks=[5, 10, 20])

    print("\n===== Evaluation Results =====")
    print(results.to_string(index=False))

    results.to_csv("evaluation_results.csv", index=False)
    print("\nSaved: evaluation_results.csv")

if __name__ == "__main__":
    main()
