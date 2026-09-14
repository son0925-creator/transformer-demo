print("=" * 60)
print("TRANSFORMER DEMO")
print("Religious Education Graduate Class")
print("=" * 60)

# STEP 1: Original Sentence
sentence = "God loves every learner."

print("\n[1] Original Sentence")
print(sentence)

# STEP 2: Tokenization
tokens = sentence.split()

print("\n[2] Tokenization")

for i, token in enumerate(tokens):
    print(f"Token {i + 1}: {token}")

print("\nSentence -> Tokens")
print(sentence, "->", tokens)

import numpy as np

print("\n" + "=" * 60)
print("[3] WORDS BECOME VECTORS")
print("=" * 60)

# 교육용으로 단순화한 임베딩 벡터
embeddings = np.array([
    [1.0, 0.8, 0.2],   # God
    [0.8, 1.0, 0.3],   # loves
    [0.2, 0.4, 0.8],   # every
    [0.3, 0.5, 1.0]    # learner
])

clean_tokens = ["God", "loves", "every", "learner"]

for token, vector in zip(clean_tokens, embeddings):
    print(f"{token:10} -> {vector}")

print("\nInterpretation:")
print("Each word is represented as a vector.")
print("Transformer processes these numerical representations.")

print("\n" + "=" * 60)
print("[4] SELF-ATTENTION")
print("=" * 60)

# 교육용 단순화:
# 실제 Transformer에서는 학습된 가중치 행렬을 이용해
# Query(Q), Key(K), Value(V)를 각각 만듭니다.
# 여기서는 원리를 쉽게 보기 위해 같은 embedding을 사용합니다.

Q = embeddings
K = embeddings
V = embeddings

# 벡터의 차원
d_k = K.shape[1]

# 1. Query와 Key의 유사도 계산
scores = np.matmul(Q, K.T) / np.sqrt(d_k)

print("\nAttention Scores:")
print(np.round(scores, 3))


# 2. Softmax 함수
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 3. Attention Weight 계산
attention_weights = softmax(scores)

print("\nAttention Weights:")
print(np.round(attention_weights, 3))


# 4. Value와 결합하여 새로운 문맥 벡터 생성
context = np.matmul(attention_weights, V)

print("\nContextual Representations:")

for token, vector in zip(clean_tokens, context):
    print(f"{token:10} -> {np.round(vector, 3)}")

    import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("[5] VISUALIZING ATTENTION")
print("=" * 60)

plt.figure(figsize=(7, 6))

plt.imshow(attention_weights)

plt.xticks(
    range(len(clean_tokens)),
    clean_tokens
)

plt.yticks(
    range(len(clean_tokens)),
    clean_tokens
)

plt.xlabel("Key words")
plt.ylabel("Query words")

plt.title("Self-Attention Weights")

# 각 칸에 숫자 표시
for i in range(len(clean_tokens)):
    for j in range(len(clean_tokens)):
        plt.text(
            j,
            i,
            f"{attention_weights[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.colorbar(label="Attention Weight")

plt.tight_layout()
plt.show()


print("\n" + "=" * 60)
print("[6] REAL PRETRAINED TRANSFORMER")
print("=" * 60)

from transformers import pipeline

print("\nLoading pretrained Transformer model...")
print("Please wait...\n")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentences = [
    "I love studying religious education.",
    "Learning together gives me hope.",
    "I hate this class.",
    "Religious education can help people understand one another."
]

for sentence in sentences:
    result = classifier(sentence)[0]

    print("Sentence:")
    print(sentence)

    print("Prediction:")
    print(
        f"{result['label']} "
        f"(confidence: {result['score']:.3f})"
    )

    print("-" * 50)