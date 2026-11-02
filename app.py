import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush, heappop
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score

# =========================================================
# ORACLE — AI INVESTIGATION SYSTEM
# Syllabus concepts: Search, Bayesian reasoning, HMM,
# Minimax, ML, Regression, MDP, Value Iteration, Q-learning
# =========================================================

st.set_page_config(
    page_title="ORACLE AI Investigation System",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 ORACLE — AI Investigation System")
st.caption(
    "A fictional investigation project demonstrating AI and ML concepts."
)

# =========================================================
# 1. INVESTIGATION STATE-SPACE GRAPH
# =========================================================

G = nx.DiGraph()

edges = [
    ("Crime Scene", "Window", 2),
    ("Crime Scene", "CCTV Room", 3),
    ("Crime Scene", "Garden", 4),
    ("Window", "Footprints", 2),
    ("Window", "Room A", 3),
    ("CCTV Room", "Camera Recording", 1),
    ("Garden", "Broken Watch", 2),
    ("Garden", "Room B", 3),
    ("Footprints", "Alex", 3),
    ("Camera Recording", "Blake", 2),
    ("Broken Watch", "Casey", 2),
]

for source, target, cost in edges:
    G.add_edge(source, target, cost=cost)

heuristic = {
    "Crime Scene": 6,
    "Window": 4,
    "CCTV Room": 3,
    "Garden": 4,
    "Footprints": 2,
    "Camera Recording": 1,
    "Broken Watch": 2,
    "Room A": 3,
    "Room B": 3,
    "Alex": 0,
    "Blake": 0,
    "Casey": 0,
}


# =========================================================
# 2. SEARCH ALGORITHMS
# =========================================================

def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None


def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)

            for neighbor in reversed(list(graph.neighbors(node))):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    return None


def dls(graph, start, goal, limit):
    def search(node, path, depth):
        if node == goal:
            return path

        if depth >= limit:
            return None

        for neighbor in graph.neighbors(node):
            if neighbor not in path:
                result = search(
                    neighbor,
                    path + [neighbor],
                    depth + 1
                )

                if result:
                    return result

        return None

    return search(start, [start], 0)


def ucs(graph, start, goal):
    heap = [(0, start, [start])]
    best_cost = {}

    while heap:
        cost, node, path = heappop(heap)

        if node == goal:
            return path

        if node in best_cost and best_cost[node] <= cost:
            continue

        best_cost[node] = cost

        for neighbor in graph.neighbors(node):
            edge_cost = graph[node][neighbor]["cost"]

            heappush(
                heap,
                (cost + edge_cost, neighbor, path + [neighbor])
            )

    return None


def greedy_search(graph, start, goal):
    heap = [(heuristic.get(start, 0), start, [start])]
    visited = set()

    while heap:
        _, node, path = heappop(heap)

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for neighbor in graph.neighbors(node):
            heappush(
                heap,
                (
                    heuristic.get(neighbor, 0),
                    neighbor,
                    path + [neighbor]
                )
            )

    return None


def astar(graph, start, goal):
    heap = [(heuristic.get(start, 0), 0, start, [start])]
    best_cost = {}

    while heap:
        _, cost, node, path = heappop(heap)

        if node == goal:
            return path

        if node in best_cost and best_cost[node] <= cost:
            continue

        best_cost[node] = cost

        for neighbor in graph.neighbors(node):
            new_cost = cost + graph[node][neighbor]["cost"]
            score = new_cost + heuristic.get(neighbor, 0)

            heappush(
                heap,
                (score, new_cost, neighbor, path + [neighbor])
            )

    return None


def path_cost(graph, path):
    if not path or len(path) < 2:
        return 0

    return sum(
        graph[path[i]][path[i + 1]]["cost"]
        for i in range(len(path) - 1)
    )


# =========================================================
# 3. BAYESIAN REASONING
# =========================================================

suspects = ["Alex", "Blake", "Casey"]
priors = np.array([0.33, 0.33, 0.34])

evidence_likelihoods = {
    "Footprints match": np.array([0.80, 0.20, 0.25]),
    "CCTV match": np.array([0.15, 0.85, 0.20]),
    "Watch belongs to suspect": np.array([0.20, 0.25, 0.90]),
}


def bayesian_update(selected_evidence):
    posterior = priors.copy()

    for evidence in selected_evidence:
        posterior *= evidence_likelihoods[evidence]

    total = posterior.sum()

    if total == 0:
        return priors

    return posterior / total


# =========================================================
# 4. MACHINE LEARNING CLASSIFICATION
# =========================================================

ml_data = pd.DataFrame({
    "physical_evidence": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "camera_match":      [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    "witness_match":     [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    "relevant":          [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
})

X = ml_data[
    ["physical_evidence", "camera_match", "witness_match"]
]
y = ml_data["relevant"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

ml_accuracy = accuracy_score(
    y_test,
    classifier.predict(X_test)
)


# =========================================================
# 5. REGRESSION
# =========================================================

regression_data = pd.DataFrame({
    "clue_complexity": [1, 2, 3, 4, 5, 2, 4, 3, 5, 1],
    "number_of_leads": [1, 2, 2, 4, 5, 3, 3, 4, 6, 1],
    "minutes":         [10, 20, 25, 45, 60, 28, 40, 38, 70, 12],
})

reg_X = regression_data[
    ["clue_complexity", "number_of_leads"]
]
reg_y = regression_data["minutes"]

regression_model = LinearRegression()
regression_model.fit(reg_X, reg_y)


# =========================================================
# 6. MINIMAX ALGORITHM DEMONSTRATION
# =========================================================

def minimax(node, depth, maximizing, tree):
    if depth == 0 or node not in tree:
        return tree.get(node, 0)

    if maximizing:
        best = float("-inf")

        for child in tree[node]:
            best = max(
                best,
                minimax(child, depth - 1, False, tree)
            )

        return best

    best = float("inf")

    for child in tree[node]:
        best = min(
            best,
            minimax(child, depth - 1, True, tree)
        )

    return best


game_tree = {
    "Root": ["A", "B"],
    "A": ["A1", "A2"],
    "B": ["B1", "B2"],
    "A1": 3,
    "A2": 5,
    "B1": 2,
    "B2": 9,
}


# =========================================================
# 7. HIDDEN MARKOV MODEL — FORWARD ALGORITHM
# =========================================================

def hmm_forward(observations):
    states = ["Calm", "Suspicious"]

    transition = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    emission = {
        "No clue": np.array([0.8, 0.3]),
        "Clue found": np.array([0.2, 0.7])
    }

    initial = np.array([0.6, 0.4])

    alpha = initial * emission[observations[0]]

    for observation in observations[1:]:
        alpha = (alpha @ transition) * emission[observation]

    probability = alpha.sum()

    if probability == 0:
        return 0

    return probability


# =========================================================
# 8. MDP VALUE ITERATION
# =========================================================

def value_iteration():
    states = ["Start", "Clue", "Exit"]
    actions = {
        "Start": ["Investigate", "Leave"],
        "Clue": ["Collect", "Leave"],
        "Exit": []
    }

    transitions = {
        ("Start", "Investigate"): ("Clue", 2),
        ("Start", "Leave"): ("Exit", 0),
        ("Clue", "Collect"): ("Exit", 10),
        ("Clue", "Leave"): ("Exit", 1),
    }

    gamma = 0.9
    values = {state: 0.0 for state in states}

    for _ in range(100):
        new_values = values.copy()

        for state in states:
            if state == "Exit":
                new_values[state] = 0
                continue

            possible_values = []

            for action in actions[state]:
                next_state, reward = transitions[(state, action)]

                possible_values.append(
                    reward + gamma * values[next_state]
                )

            new_values[state] = max(possible_values)

        values = new_values

    policy = {}

    for state in states:
        if state == "Exit":
            policy[state] = "Terminal"
            continue

        best_action = None
        best_value = float("-inf")

        for action in actions[state]:
            next_state, reward = transitions[(state, action)]
            score = reward + gamma * values[next_state]

            if score > best_value:
                best_value = score
                best_action = action

        policy[state] = best_action

    return values, policy


# =========================================================
# 9. Q-LEARNING DEMONSTRATION
# =========================================================

def q_learning_demo():
    states = ["Start", "Clue", "Exit"]
    actions = ["Investigate", "Leave", "Collect"]

    q_table = {
        state: {action: 0.0 for action in actions}
        for state in states
    }

    rewards = {
        ("Start", "Investigate"): ("Clue", 2),
        ("Start", "Leave"): ("Exit", 0),
        ("Clue", "Collect"): ("Exit", 10),
        ("Clue", "Leave"): ("Exit", 1),
    }

    alpha = 0.1
    gamma = 0.9

    for _ in range(100):
        state = "Start"

        while state != "Exit":
            valid_actions = [
                action for (s, action) in rewards if s == state
            ]

            action = np.random.choice(valid_actions)
            next_state, reward = rewards[(state, action)]

            max_next = max(q_table[next_state].values())

            q_table[state][action] += alpha * (
                reward
                + gamma * max_next
                - q_table[state][action]
            )

            state = next_state

    return q_table


# =========================================================
# 10. STREAMLIT USER INTERFACE
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Search",
    "🧠 Bayesian",
    "🤖 Machine Learning",
    "🎮 Minimax",
    "📊 HMM & MDP",
    "🗺️ Graph"
])


# ---------------- SEARCH TAB ----------------

with tab1:
    st.subheader("AI Search — Investigation Route")

    start = st.selectbox(
        "Starting location",
        list(G.nodes),
        index=0
    )

    goal = st.selectbox(
        "Target",
        ["Alex", "Blake", "Casey"]
    )

    algorithm = st.selectbox(
        "Search algorithm",
        [
            "BFS",
            "DFS",
            "DLS",
            "UCS",
            "Greedy Best-First",
            "A*"
        ]
    )

    depth_limit = st.slider(
        "Depth limit for DLS",
        1,
        10,
        5
    )

    if st.button("Find Route"):
        if algorithm == "BFS":
            path = bfs(G, start, goal)
        elif algorithm == "DFS":
            path = dfs(G, start, goal)
        elif algorithm == "DLS":
            path = dls(G, start, goal, depth_limit)
        elif algorithm == "UCS":
            path = ucs(G, start, goal)
        elif algorithm == "Greedy Best-First":
            path = greedy_search(G, start, goal)
        else:
            path = astar(G, start, goal)

        if path:
            st.success(" → ".join(path))
            st.write(f"Route cost: **{path_cost(G, path)}**")
        else:
            st.warning("No route found with this algorithm/settings.")


# ---------------- BAYESIAN TAB ----------------

with tab2:
    st.subheader("Bayesian Evidence Analysis")

    st.warning(
        "Fictional demonstration only. The likelihoods are invented "
        "for learning and do not establish real-world guilt."
    )

    selected_evidence = st.multiselect(
        "Select evidence found",
        list(evidence_likelihoods.keys())
    )

    posterior = bayesian_update(selected_evidence)

    bayes_df = pd.DataFrame({
        "Suspect": suspects,
        "Posterior Probability": posterior
    })

    st.dataframe(
        bayes_df.style.format({
            "Posterior Probability": "{:.1%}"
        }),
        use_container_width=True
    )

    fig, ax = plt.subplots()
    ax.bar(suspects, posterior)
    ax.set_ylabel("Posterior probability")
    ax.set_ylim(0, 1)
    ax.set_title("Bayesian Evidence Update")
    st.pyplot(fig)


# ---------------- MACHINE LEARNING TAB ----------------

with tab3:
    st.subheader("Clue Relevance — Logistic Regression")

    st.write(
        "A supervised learning model trained on a small fictional dataset."
    )

    st.metric("Test accuracy", f"{ml_accuracy:.0%}")

    physical = st.selectbox(
        "Physical evidence present?",
        [0, 1]
    )

    camera = st.selectbox(
        "Camera match?",
        [0, 1]
    )

    witness = st.selectbox(
        "Witness match?",
        [0, 1]
    )

    sample = pd.DataFrame([{
        "physical_evidence": physical,
        "camera_match": camera,
        "witness_match": witness
    }])

    prediction = classifier.predict(sample)[0]
    probability = classifier.predict_proba(sample)[0][1]

    if prediction == 1:
        st.write("Prediction: **Potentially relevant clue**")
    else:
        st.write("Prediction: **Low relevance clue**")

    st.write(f"Model probability: **{probability:.1%}**")
    st.caption("Synthetic training data; results are for demonstration.")

    st.divider()
    st.subheader("Regression — Estimate Investigation Effort")

    complexity = st.slider("Clue complexity", 1, 5, 3)
    leads = st.slider("Number of leads", 1, 6, 3)

    estimated_minutes = regression_model.predict(
        pd.DataFrame([{
            "clue_complexity": complexity,
            "number_of_leads": leads
        }])
    )[0]

    st.metric(
        "Estimated effort",
        f"{max(0, estimated_minutes):.0f} minutes"
    )


# ---------------- MINIMAX TAB ----------------

with tab4:
    st.subheader("Minimax — Decision Making")

    st.write(
        "Minimax selects the best move assuming the opponent also "
        "chooses the best move."
    )

    result = minimax("Root", 2, True, game_tree)

    st.metric("Best minimax value", result)

    st.write("Example game tree:")
    st.json(game_tree)


# ---------------- HMM & MDP TAB ----------------

with tab5:
    st.subheader("Hidden Markov Model")

    observations = st.multiselect(
        "Observed events in sequence",
        ["No clue", "Clue found"],
        default=["No clue", "Clue found"]
    )

    if observations:
        probability = hmm_forward(observations)
        st.metric(
            "Probability of observation sequence",
            f"{probability:.4f}"
        )
    else:
        st.info("Select at least one observation.")

    st.divider()
    st.subheader("MDP — Value Iteration")

    values, policy = value_iteration()

    st.write("State values:")
    st.dataframe(
        pd.DataFrame({
            "State": list(values.keys()),
            "Value": list(values.values())
        }),
        use_container_width=True
    )

    st.write("Optimal policy:")
    st.dataframe(
        pd.DataFrame({
            "State": list(policy.keys()),
            "Action": list(policy.values())
        }),
        use_container_width=True
    )

    st.divider()
    st.subheader("Q-Learning")

    q_table = q_learning_demo()

    q_df = pd.DataFrame(q_table).T
    st.write("Learned Q-values:")
    st.dataframe(q_df, use_container_width=True)


# ---------------- GRAPH TAB ----------------

with tab6:
    st.subheader("Investigation State-Space Graph")

    positions = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(12, 7))

    nx.draw_networkx(
        G,
        positions,
        ax=ax,
        with_labels=True,
        node_size=1800,
        font_size=8,
        arrows=True
    )

    edge_labels = nx.get_edge_attributes(G, "cost")

    nx.draw_networkx_edge_labels(
        G,
        positions,
        edge_labels=edge_labels,
        ax=ax,
        font_size=8
    )

    ax.set_title("ORACLE Investigation Graph")
    ax.axis("off")

    st.pyplot(fig)

    st.write("Graph nodes:")
    st.write(list(G.nodes))

    st.write("Graph edges:")
    st.write(list(G.edges))


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "ORACLE is an educational AI/ML demonstration using fictional "
    "data and simplified models."
)