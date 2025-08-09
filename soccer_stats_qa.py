from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

ROLES = {
    "defender": ["CB"],
    "midfielder": ["CM", "CDM", "CAM"],
    "attacker": ["ST", "RW", "LW"],
}

question_map = {
    "Who has the most goals": lambda df: df.loc[df["goals"].idxmax()],
    "Who has the most assists": lambda df: df.loc[df["assists"].idxmax()],
    "Who has the least goals": lambda df: df.loc[df["goals"].idxmin()],
    "Who has the least assists": lambda df: df.loc[df["assists"].idxmin()],
    "Which defender has the most assists": lambda df: df[df["position"].isin(ROLES["defender"])]
    .sort_values(by="assists", ascending=False)
    .head(1),
    "Which defender has the most goals": lambda df: df[df["position"].isin(ROLES["defender"])]
    .sort_values(by="goals", ascending=False)
    .head(1),
    "Which defender has the least assists": lambda df: df[df["position"].isin(ROLES["defender"])]
    .sort_values(by="assists", ascending=True)
    .head(1),
    "Which defender has the least goals": lambda df: df[df["position"].isin(ROLES["defender"])]
    .sort_values(by="goals", ascending=True)
    .head(1),
    "Which midfielder has the most goals": lambda df: df[df["position"].isin(ROLES["midfielder"])]
    .sort_values(by="goals", ascending=False)
    .head(1),
    "Which midfielder has the most assists": lambda df: df[df["position"].isin(ROLES["midfielder"])]
    .sort_values(by="assists", ascending=False)
    .head(1),
    "Which midfielder has the least goals": lambda df: df[df["position"].isin(ROLES["midfielder"])]
    .sort_values(by="goals", ascending=True)
    .head(1),
    "Which midfielder has the least assists": lambda df: df[df["position"].isin(ROLES["midfielder"])]
    .sort_values(by="assists", ascending=True)
    .head(1),
    "Which attacker has the most assists": lambda df: df[df["position"].isin(ROLES["attacker"])]
    .sort_values(by="assists", ascending=False)
    .head(1),
    "Which attacker has the most goals": lambda df: df[df["position"].isin(ROLES["attacker"])]
    .sort_values(by="goals", ascending=False)
    .head(1),
    "Which attacker has the least assists": lambda df: df[df["position"].isin(ROLES["attacker"])]
    .sort_values(by="assists", ascending=True)
    .head(1),
    "Which attacker has the least goals": lambda df: df[df["position"].isin(ROLES["attacker"])]
    .sort_values(by="goals", ascending=True)
    .head(1),
}

players = [
    "Mo Salah", "Kevin De Bruyne", "Harry Kane", "Virgil van Dijk",
    "Bruno Fernandes", "Sadio Mané", "Declan Rice", "Jack Grealish",
    "Bukayo Saka", "James Maddison",
]

for player in players:
    question_map[f"What position does {player} play"] = (
        lambda df, player=player: df[df["name"] == player]["position"].values[0]
    )
    question_map[f"How many goals does {player} have"] = (
        lambda df, player=player: df[df["name"] == player]["goals"].values[0]
    )
    question_map[f"How many assists does {player} have"] = (
        lambda df, player=player: df[df["name"] == player]["assists"].values[0]
    )

for n in range(1, 11):
    question_map[f"Top {n} goal scorers"] = (
        lambda df, n=n: df.sort_values(by="goals", ascending=False).head(n)
    )
    question_map[f"Top {n} assist makers"] = (
        lambda df, n=n: df.sort_values(by="assists", ascending=False).head(n)
    )
    question_map[f"Bottom {n} goal scorers"] = (
        lambda df, n=n: df.sort_values(by="goals", ascending=True).head(n)
    )
    question_map[f"Bottom {n} assist makers"] = (
        lambda df, n=n: df.sort_values(by="assists", ascending=True).head(n)
    )

example_questions = list(question_map.keys())
example_embeddings = model.encode(example_questions, convert_to_tensor=True)

data = {
    "name": [
        "Mo Salah", "Kevin De Bruyne", "Harry Kane", "Virgil van Dijk",
        "Bruno Fernandes", "Sadio Mané", "Declan Rice", "Jack Grealish",
        "Bukayo Saka", "James Maddison",
    ],
    "goals": [22, 8, 21, 3, 10, 18, 2, 5, 14, 7],
    "assists": [10, 16, 4, 1, 9, 7, 3, 8, 11, 6],
    "position": ["RW", "CM", "ST", "CB", "CAM", "LW", "CDM", "LW", "RW", "CAM"],
}

df = pd.DataFrame(data)


while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, example_embeddings)[0]
    best_match_index = scores.argmax()

    if scores[best_match_index] > 0.6:
        best_question = example_questions[best_match_index]
        answer = question_map[best_question](df)
        print("\nAnswer:\n", answer)
    else:
        print("Sorry, I didn’t understand.")
