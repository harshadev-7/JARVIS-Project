# J.A.R.V.I.S. Custom Transformer Architecture Specification
## Phase 2: Mathematical Design, Tensor Dimensionality & Hardware Sizing

**Project:** J.A.R.V.I.S. — Independent Local Artificial Intelligence System  
**Author:** AI Research & Machine Learning Engineering Team  
**Target Hardware:** Intel Core i3 (4 physical cores, 8 threads), 8 GB RAM, Windows 11  
**Execution Paradigm:** Pure CPU Execution (PyTorch CPU, no CUDA, no external APIs)

---

## 1. High-Level Architecture Overview

J.A.R.V.I.S. uses a **decoder-only autoregressive Transformer** inspired by the GPT architecture (Radford et al., Vaswani et al.), but redesigned from the ground up for lightweight, CPU-efficient local training and inference. 

Every component—from tokenization and positional encoding to Query/Key/Value projections, scaled dot-product causal attention, residual connections, and language-model projection heads—is explicitly formulated without high-level black-box wrappers.

```
                    ┌───────────────────────────────┐
                    │      Input Tokens x_t         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   Token Embedding (E_tok) + Positional Embedding (E_pos)│
       └────────────────────────────┬───────────────────────────┘
                                    │  h_0 ∈ ℝ^(B × T × d_model)
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │                 Transformer Block 1                    │
       │  ┌──────────────────────────────────────────────────┐  │
       │  │ Residual Connection                              │  │
       │  │    h' = h + Dropout(MaskedSelfAttention(LN(h)))  │  │
       │  └──────────────────────────────────────────────────┘  │
       │  ┌──────────────────────────────────────────────────┐  │
       │  │ Residual Connection                              │  │
       │  │    h_out = h' + Dropout(FeedForward(LN(h')))      │  │
       │  └──────────────────────────────────────────────────┘  │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                 × L Layers
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │            Final Layer Normalization (LN_f)            │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │        Language Model Head (W_lm ∈ ℝ^(d_model × |V|))   │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Logits Z ∈ ℝ^(B × T × |V|) │
                    └───────────────────────────────┘
```

---

## 2. Mathematical Component Formulations

### 2.1. Token & Positional Embeddings
Let $x = (x_1, x_2, \dots, x_T) \in \{0, 1, \dots, |V|-1\}^T$ be a sequence of token IDs of length $T$, where $|V|$ is the vocabulary size.

1. **Token Embedding Matrix**:
   $$E_{\text{tok}} \in \mathbb{R}^{|V| \times d_{\text{model}}}$$
   For each token $x_t$, the token embedding is $e_t^{\text{tok}} = E_{\text{tok}}[x_t] \in \mathbb{R}^{d_{\text{model}}}$.

2. **Learnable Positional Embedding Matrix**:
   $$E_{\text{pos}} \in \mathbb{R}^{T_{\text{max}} \times d_{\text{model}}}$$
   For position index $t \in \{0, 1, \dots, T-1\}$, the positional vector is $e_t^{\text{pos}} = E_{\text{pos}}[t] \in \mathbb{R}^{d_{\text{model}}}$.

3. **Combined Input Representation**:
   $$h_0 = \text{Dropout}(e^{\text{tok}} + e^{\text{pos}}, p_{\text{drop}}) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
   where $B$ is the mini-batch size.

---

### 2.2. Pre-Layer Normalization (Pre-LN)
We adopt the **Pre-LN** formulation rather than Post-LN. Pre-LN places normalization *inside* the residual stream, preventing gradient vanishing or exploding during CPU backpropagation without needing aggressive learning-rate warmups.

For vector $z \in \mathbb{R}^{d_{\text{model}}}$:
$$\mu = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} z_i, \quad \sigma^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (z_i - \mu)^2$$

$$\text{LN}(z) = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
where:
* $\gamma \in \mathbb{R}^{d_{\text{model}}}$ is the learnable gain (initialized to $\mathbf{1}$).
* $\beta \in \mathbb{R}^{d_{\text{model}}}$ is the learnable bias (initialized to $\mathbf{0}$).
* $\epsilon = 10^{-5}$ prevents division by zero.

---

### 2.3. Handcrafted Masked Multi-Head Self-Attention (MHA)
Rather than relying on high-level library functions, MHA is constructed from explicit matrix projections.

Let $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ be the normalized layer input.

1. **Query, Key, Value Projections**:
   $$Q = X W_Q + b_Q \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
   $$K = X W_K + b_K \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
   $$V = X W_V + b_V \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
   where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

2. **Multi-Head Partitioning**:
   The hidden dimension $d_{\text{model}}$ is split across $h$ heads, each with dimension:
   $$d_k = \frac{d_{\text{model}}}{h}$$
   $Q, K, V$ are reshaped and transposed to:
   $$Q_i, K_i, V_i \in \mathbb{R}^{B \times h \times T \times d_k}$$

3. **Scaled Dot-Product Raw Attention Scores**:
   $$S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times h \times T \times T}$$
   Dividing by $\sqrt{d_k}$ stabilizes the variance of the dot product to $1.0$, preventing softmax saturation.

4. **Autoregressive Causal Masking**:
   Tokens must never attend to future tokens. The causal mask $M \in \mathbb{R}^{T \times T}$ is defined as:
   $$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
   $$\tilde{S} = S + M$$

5. **Softmax & Context Aggregation**:
   $$A = \text{Softmax}(\tilde{S}, \text{dim}=-1) \in \mathbb{R}^{B \times h \times T \times T}$$
   $$\text{head}_i = A_i V_i \in \mathbb{R}^{B \times h \times T \times d_k}$$

6. **Head Concatenation & Output Projection**:
   $$\text{Concat}(\text{head}_1, \dots, \text{head}_h) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
   $$\text{MHA}(X) = \text{Concat}(\text{heads}) W_O + b_O$$
   where $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

---

### 2.4. Position-Wise Feed-Forward Network (FFN / MLP)
Following the attention sub-layer, each token vector passes through a two-layer multi-layer perceptron with an expansion factor of 4:
$$d_{\text{ff}} = 4 \times d_{\text{model}}$$

$$\text{FFN}(z) = \left( \text{GELU}(z W_1 + b_1) \right) W_2 + b_2$$
where:
* $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
* $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
* $\text{GELU}(a) = 0.5a \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( a + 0.044715 a^3 \right) \right) \right)$ provides smooth gradient flow across near-zero activations.

---

### 2.5. Full Transformer Decoder Block
The $l$-th decoder block combines MHA, FFN, LayerNorm, and residual connections:

$$h^{(l)'} = h^{(l-1)} + \text{Dropout}\Big( \text{MHA}\big( \text{LN}_1(h^{(l-1)}) \big) \Big)$$
$$h^{(l)} = h^{(l)'} + \text{Dropout}\Big( \text{FFN}\big( \text{LN}_2(h^{(l)'}) \big) \Big)$$

This maintains a direct gradient highway through $h^{(0)} \to h^{(1)} \to \dots \to h^{(L)}$, critical for shallow CPU training.

---

### 2.6. Language Model Output Head & Cross-Entropy Loss
1. **Final Normalization & Projection**:
   $$\bar{h} = \text{LN}_{\text{final}}(h^{(L)}) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
   $$Z = \bar{h} W_{\text{lm}} + b_{\text{lm}} \in \mathbb{R}^{B \times T \times |V|}$$

2. **Next-Token Prediction Objective**:
   Given input sequence $(x_1, \dots, x_{T-1})$, the target sequence is $(x_2, \dots, x_T)$.
   The loss is the mean Cross-Entropy over non-padding tokens:
   $$\mathcal{L} = -\frac{1}{N_{\text{valid}}} \sum_{b=1}^B \sum_{t=1}^{T-1} \mathbf{1}_{\{x_{t+1} \neq \text{PAD}\}} \log \left( \frac{\exp(Z_{b, t, x_{t+1}})}{\sum_{k=1}^{|V|} \exp(Z_{b, t, k})} \right)$$

---

## 3. Autoregressive Sampling Algorithm

During inference, tokens are generated one by one:
1. Pass input prompt tokens $[x_1, \dots, x_t]$ through the network.
2. Extract the logits for the final step: $z_t = Z[:, -1, :] \in \mathbb{R}^{|V|}$.
3. Scale by temperature $\tau$:
   $$z_t' = \frac{z_t}{\tau}$$
4. Apply **Top-k Truncation**:
   Keep only the $k$ highest logit values; set all others to $-\infty$.
5. Convert to probability distribution:
   $$P(x_{t+1} = k) = \frac{\exp(z'_{t, k})}{\sum_j \exp(z'_{t, j})}$$
6. Sample next token ID $x_{t+1} \sim P(x_{t+1})$.
7. If $x_{t+1} = \text{EOS}$ or $t \ge T_{\text{max}}$, terminate generation. Otherwise, append $x_{t+1}$ and repeat.

---

## 4. Hardware Configuration & Parameter Sizing

Targeting an **Intel Core i3 with 8 GB RAM**, we design two balanced model profiles:

### 4.1. Profile A: "JARVIS-Micro" (Educational & Rapid CPU Verification)
* Vocabulary Size ($|V|$): $1,024$ tokens
* Context Window ($T_{\text{max}}$): $64$ tokens
* Embedding Dimension ($d_{\text{model}}$): $64$
* Attention Heads ($h$): $4$ ($d_k = 16$)
* Layers ($L$): $2$
* Feedforward Dimension ($d_{\text{ff}}$): $256$
* **Parameter Breakdown:**
  * Token Embedding: $1024 \times 64 = 65,536$
  * Position Embedding: $64 \times 64 = 4,096$
  * Per-Layer MHA ($Q, K, V, O$): $4 \times (64 \times 64 + 64) = 16,640$
  * Per-Layer FFN: $(64 \times 256 + 256) + (256 \times 64 + 64) = 33,088$
  * Per-Layer LN ($2 \times 2 \times 64$): $256$
  * Total Per Layer: $49,984 \times 2 \text{ layers} = 99,968$
  * Final LN: $128$
  * Output LM Head: $64 \times 1024 + 1024 = 66,560$
* **Total Parameters:** **$\approx 236,288$ parameters** ($\approx 0.94 \text{ MB}$ FP32 memory)
* **Peak RAM During Training:** $< 120 \text{ MB}$
* **Epoch Time on Core i3:** $\approx 5 - 15 \text{ seconds}$

### 4.2. Profile B: "JARVIS-Base" (Domain Offline Assistant)
* Vocabulary Size ($|V|$): $2,048$ tokens
* Context Window ($T_{\text{max}}$): $128$ tokens
* Embedding Dimension ($d_{\text{model}}$): $128$
* Attention Heads ($h$): $4$ ($d_k = 32$)
* Layers ($L$): $4$
* Feedforward Dimension ($d_{\text{ff}}$): $512$
* **Total Parameters:** **$\approx 1,385,000$ parameters** ($\approx 5.5 \text{ MB}$ FP32 memory)
* **Peak RAM During Training:** $< 250 \text{ MB}$
* **Epoch Time on Core i3:** $\approx 45 - 90 \text{ seconds}$

Both profiles are well within the physical limits of 8 GB RAM and operate without GPU or cloud compute.

---

## 5. Tensor Shape Trace (Forward Pass)

The following contract guides our Phase 3 modular code implementation:

| Step / Sub-Layer | Input Shape | Operation | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input IDs** | $(B, T)$ | Token indices | $(B, T)$ |
| **Embedding** | $(B, T)$ | $E_{\text{tok}}[x] + E_{\text{pos}}[p]$ | $(B, T, d_{\text{model}})$ |
| **LayerNorm 1** | $(B, T, d_{\text{model}})$ | Pre-LN before attention | $(B, T, d_{\text{model}})$ |
| **$Q, K, V$ Projections** | $(B, T, d_{\text{model}})$ | Linear projections & head reshape | $(B, h, T, d_k)$ |
| **Causal Attention** | $(B, h, T, d_k)$ | $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$ | $(B, h, T, d_k)$ |
| **Head Concat & Out Proj** | $(B, h, T, d_k)$ | Concat heads & project with $W_O$ | $(B, T, d_{\text{model}})$ |
| **Residual 1** | $(B, T, d_{\text{model}})$ | $h + \text{MHA}(\text{LN}(h))$ | $(B, T, d_{\text{model}})$ |
| **LayerNorm 2** | $(B, T, d_{\text{model}})$ | Pre-LN before MLP | $(B, T, d_{\text{model}})$ |
| **FFN / MLP** | $(B, T, d_{\text{model}})$ | $W_1 \to \text{GELU} \to W_2$ | $(B, T, d_{\text{model}})$ |
| **Residual 2** | $(B, T, d_{\text{model}})$ | $h' + \text{FFN}(\text{LN}(h'))$ | $(B, T, d_{\text{model}})$ |
| **Final LayerNorm** | $(B, T, d_{\text{model}})$ | $\text{LN}_{\text{final}}$ after $L$ blocks | $(B, T, d_{\text{model}})$ |
| **LM Head** | $(B, T, d_{\text{model}})$ | Linear projection to $|V|$ | $(B, T, |V|)$ |

---

## Summary

This design satisfies all requirements of Section 3 of your blueprint:
1. No high-level black-box wrappers; full implementation from mathematical first principles.
2. Explicit causal masking and multi-head decomposition.
3. Pre-LN residual architecture for numerical stability on CPU.
4. Memory footprint under 250 MB, tailored for Intel Core i3 and 8 GB RAM.
